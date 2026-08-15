"""完整即時推理管線(多人,continual inference)。

主迴圈:
    讀影格(cv2)→ 每 N 幀取樣(目標 8-10 fps 特徵率)
    → 人物偵測 + ByteTrack 追蹤
    → 每 track 裁上半身 ROI(EMA 平滑框)
    → 所有 track 的 ROI 合成一個 batch,backbone 只前向一次
    → 各自 push 進 per-track DualScaleBuffer
    → 時序頭(僅重算頭,不重算 backbone = continual inference)
    → 狀態機 + 單週期分數 → EMA 累積 + 雙門檻警報
    → 畫面疊加(框、track ID、P_t 置信度條、警報狀態),可存 mp4

用法:
    python -m inference.pipeline --source 0                # webcam
    python -m inference.pipeline --source video.mp4 --model-ckpt ckpt.pt
    (--no-model 時只跑偵測+追蹤,對應里程碑 M1 的骨架驗證)
"""
import argparse
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from models.full_model import build_model
from models.ring_buffer import DualScaleBuffer
from tracking.detector import PersonDetector
from tracking.tracker import PersonTracker
from tracking.roi import ROISmoother, crop_upper_body
from inference.state_machine import (StageStateMachine, cycle_score,
                                     HandToMouthCounter, LoiterDetector,
                                     MovementGate, PresenceClassifier)
from inference.skeleton import (SkeletonStageEstimator, draw_skeleton,
                                L_WRI, R_WRI)
from inference.alarm import AlarmManager
from inference import methods as methods_registry
from inference import verifier as verify_mod
from utils import load_config, resolve_device

# ImageNet 正規化(與訓練一致)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_roi(roi_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 ROI → 正規化 float tensor (3, H, W)。"""
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


@dataclass
class TrackState:
    """單一 track 的推理狀態集合。"""
    buffer: DualScaleBuffer
    state_machine: StageStateMachine
    counter: HandToMouthCounter
    move_gate: Optional[MovementGate] = None
    moving: bool = False         # 視窗內移動 ≥ N 倍身高
    presence: Optional[PresenceClassifier] = None
    presence_state: str = "unknown"   # 經過 / 徘徊 / 等待
    wander_notified: bool = False     # 徘徊已通報(每個 track 只報一次)
    wait_notified: bool = False       # 等待已通報(同上)
    wait_gate_logged: bool = False    # 「型態不符」只提示一次
    skeleton: Optional[SkeletonStageEstimator] = None
    loiter: Optional[LoiterDetector] = None
    last_seen: float = 0.0
    last_stage: int = 3          # background
    last_P: float = 0.0
    alarm_active: bool = False
    loitering: bool = False
    unverified: bool = False     # 背向且網路分數高 → 無法確認警示
    phone: bool = False          # 手舉超過 max_dwell 未放下 → 講電話姿態
    last_hardcase_t: float = float("-inf")  # hard case 存檔冷卻
    # 「接近未達標」診斷:手靠近臉但沒過 S2 門檻的一段動作
    approach_active: bool = False
    approach_start_t: float = 0.0
    approach_last_t: float = 0.0
    approach_min_d: float = float("inf")
    approach_saw_s2: bool = False
    last_kpts: Optional[np.ndarray] = None
    last_dnorm: Optional[float] = None
    ori_hist: deque = field(default_factory=deque)  # (t, orientation)
    # 節點滾動緩衝(t, kpts):第二階段複核的輸入來源。
    # 17×3 float @10fps 每人每分鐘約 120 KB,成本近零,所以恆常累積,
    # 不管當下有沒有要複核 —— 警報觸發時證據已經在過去,來不及回頭錄
    pose_hist: deque = field(default_factory=deque)
    verify: Optional["verify_mod.VerifyResult"] = None
    verify_pending: bool = False

    def back_fraction(self, now: float, window: float) -> float:
        """近期視窗內判定為背向的幀比例。"""
        while self.ori_hist and now - self.ori_hist[0][0] > window:
            self.ori_hist.popleft()
        if len(self.ori_hist) < 3:
            return 0.0
        return sum(1 for _, o in self.ori_hist if o == "back") \
            / len(self.ori_hist)


class SmokingDetectionPipeline:
    """多人抽菸行為偵測管線。"""

    def __init__(self, infer_cfg: dict, model_cfg: Optional[dict] = None,
                 ckpt_path: Optional[str] = None, use_model: bool = True,
                 method=None):
        # 判定方法(見 inference/methods.py):決定 stage1 怎麼算分、
        # 要不要開骨架分支、要不要載外觀網路、以及有沒有第二階段複核。
        # 沒指定時由 use_model 反推,舊呼叫端的行為完全不變。
        if method is None:
            method = methods_registry.get("hybrid" if use_model else "rule")
        elif isinstance(method, str):
            method = methods_registry.get(method)
        self.method = method
        if method.needs_appearance and model_cfg is None:
            raise ValueError(
                f"方法「{method.name}」需要外觀網路,請提供 model_cfg")
        # 方法自己決定骨架分支的開關,不受設定檔挑錯影響
        infer_cfg = method.apply(infer_cfg)
        self.cfg = infer_cfg
        self.use_model = (method.needs_appearance and use_model
                          and model_cfg is not None)

        det = infer_cfg["detector"]
        self.device = resolve_device(det.get("device", "auto"))

        # 骨架分支:啟用時改用 pose 模型(一次前向同時取得框與關鍵點)
        skel = infer_cfg.get("skeleton", {})
        self.skeleton_enabled = bool(skel.get("enabled", False))
        self.skel_cfg = skel
        if self.skeleton_enabled:
            from tracking.pose_detector import PoseDetector
            self.detector = PoseDetector(
                skel.get("model", "yolov8s-pose.pt"), det["conf"],
                det.get("device", "auto"))
        else:
            self.detector = PersonDetector(det["model"], det["conf"],
                                           det.get("device", "auto"))
        trk = infer_cfg["tracker"]
        self.tracker = PersonTracker(**trk)

        self.roi_cfg = infer_cfg["roi"]
        self.smoother = ROISmoother(beta=self.roi_cfg["ema_beta"],
                                    jump_threshold=self.roi_cfg["jump_threshold"])

        self.model = None
        if self.use_model:
            self.model_cfg = model_cfg
            self.model = build_model(model_cfg).to(self.device).eval()
            if ckpt_path:
                state = torch.load(ckpt_path, map_location=self.device,
                                   weights_only=True)
                self.model.load_state_dict(state.get("model", state))
                print(f"[管線] 已載入權重:{ckpt_path}")
            else:
                print("[管線] 未指定權重,使用隨機初始化(僅供管線驗證)")

        sm = infer_cfg["state_machine"]
        self.sm_cfg = sm
        al = infer_cfg["alarm"]
        # 通報次數門檻:手到嘴事件 ≥ 此次數才允許紅色警報(GUI 可調)
        self.min_events = int(al.get("min_events", 3))
        # 次數主導模式:次數達標且無任何排除條件 → 直接把分數推向觸發區
        # (否則 0.4×次數分上限 0.32 永遠過不了 0.6 觸發線,
        #  次數形同虛設,警報實際被網路分數綁架)
        self.count_driven = bool(al.get("count_driven", True))
        # 事件結算通知(GUI 掛載後可顯示「停留幾秒/是否計入/原因」)
        self.on_event = None
        # 自由格式診斷訊息(接近未達標等)
        self.on_log = None
        # 徘徊通報(GUI 可開關,預設關):徘徊不是抽菸,走橘色警示而非
        # 紅色警報,也不碰 P_t —— 兩者是不同語意的事件,不該混在一起
        self.presence_cfg = infer_cfg.get("presence", {})
        self.wander_alert_enabled = bool(
            self.presence_cfg.get("alert_wandering", False))
        # 等待通報:與徘徊同一類(在場型態的橘色警示),不影響抽菸警報。
        # 站著不動的人才是抽菸的主要對象,所以這個開關比徘徊更常用得到
        self.wait_alert_enabled = bool(
            self.presence_cfg.get("alert_waiting", False))
        # 抽菸警報總開關:關掉只做偵測與在場型態,不發紅色警報
        self.smoking_alarm_enabled = True
        # 只有「等待」才判抽菸:抽菸是站定了才做的事,經過與徘徊的人
        # 手部擺動很像手到嘴。這比移動排除嚴格——移動排除看的是 10 秒內
        # 的位移,一個剛進畫面、型態還在「判定中」的人只要當下沒走動就
        # 攔不住;這一條看的是在場型態本身。
        self.smoking_requires_waiting = bool(
            self.presence_cfg.get("smoking_requires_waiting", True))
        self.on_presence = None   # (track_id, 在場秒數, 累積路徑)
        self._dwell_override = None  # GUI 即時調整停留窗口用

        # 移動排除(可開關):走動中(累積移動 ≥ N 倍身高)不視為抽菸
        mg = infer_cfg.get("move_gate", {})
        self.mg_cfg = mg
        self.move_gate_enabled = bool(mg.get("enabled", True))
        self.alarm = AlarmManager(
            ema_alpha=al["ema_alpha"],
            trigger_threshold=al["trigger_threshold"],
            release_threshold=al["release_threshold"],
            sustain_sec=al["sustain_sec"],
            snapshot_dir=al["snapshot_dir"])

        # 第二階段複核(方法不含 stage2 時為 None)。
        # 複核在背景執行緒跑:它是離線工作,不該讓即時路徑等它。
        vcfg = infer_cfg.get("verify", {})
        self.verifier = verify_mod.build(method, infer_cfg)
        self.verify_window_sec = float(vcfg.get("window_sec", 90.0))
        self.on_verify = None      # (track_id, VerifyResult)
        self._verify_pool = None
        if self.verifier is not None:
            from concurrent.futures import ThreadPoolExecutor
            self._verify_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="verify")
        target_fps = float(self.cfg["sampling"]["target_fps"])
        # 緩衝格數留 1.2 倍餘裕:來源 fps 抖動時不會把窗頭吃掉
        self._pose_maxlen = max(16, int(self.verify_window_sec
                                        * target_fps * 1.2))

        self.recycle_sec = infer_cfg["track_recycle_sec"]
        self._tracks: Dict[int, TrackState] = {}

    # ---------- track 狀態管理 ----------

    def _get_track_state(self, tid: int) -> TrackState:
        if tid not in self._tracks:
            mc = self.model_cfg if self.use_model else None
            if mc is not None:
                feat = mc["feature"]
                buf = mc["buffer"]
                buffer = DualScaleBuffer(
                    C=feat["C"], H=feat["H"], W=feat["W"],
                    short_T=buf["short"]["T"], short_stride=buf["short"]["stride"],
                    long_T=buf["long"]["T"], long_stride=buf["long"]["stride"])
            else:
                buffer = None
            skeleton, loiter = None, None
            if self.skeleton_enabled:
                skeleton = SkeletonStageEstimator(
                    near_ratio=self.skel_cfg.get("near_ratio", 0.6),
                    move_ratio=self.skel_cfg.get("move_ratio", 0.35),
                    kpt_conf=self.skel_cfg.get("kpt_conf", 0.3),
                    nose_conf=self.skel_cfg.get("nose_conf", 0.5),
                    min_scale_px=self.skel_cfg.get("min_scale_px", 24.0),
                    kpt_err_px=self.skel_cfg.get("kpt_err_px", 4.0),
                    rise_margin=self.skel_cfg.get("rise_margin", 0.5),
                    fps=self.cfg["sampling"]["target_fps"])
                lo = self.cfg.get("loiter", {})
                if lo.get("enabled", True):
                    loiter = LoiterDetector(
                        min_duration=lo.get("min_duration", 20.0),
                        move_ratio=lo.get("move_ratio", 0.6),
                        wrist_vis_max=lo.get("wrist_vis_max", 0.1))
            esc = self.cfg.get("escalation", {})
            lv = esc.get("levels", {"low": 0.2, "mid": 0.5, "high": 0.8})
            pr = self.presence_cfg
            self._tracks[tid] = TrackState(
                buffer=buffer,
                state_machine=StageStateMachine(
                    window_sec=self.sm_cfg["window_sec"],
                    s2_min_dwell=self.sm_cfg["s2_min_dwell"]),
                move_gate=MovementGate(
                    max_heights=self.mg_cfg.get("max_heights", 3.0),
                    window_sec=self.mg_cfg.get("window_sec", 10.0)),
                presence=PresenceClassifier(
                    window_sec=pr.get("window_sec", 60.0),
                    short_stay=pr.get("short_stay", 8.0),
                    long_stay=pr.get("long_stay", 20.0),
                    pass_path=pr.get("pass_path", 1.0),
                    wander_path=pr.get("wander_path", 3.0),
                    run_speed=pr.get("run_speed", 1.5)),
                pose_hist=deque(maxlen=self._pose_maxlen),
                counter=HandToMouthCounter(
                    window_sec=esc.get("window_sec", 90.0),
                    min_dwell=(self._dwell_override[0] if self._dwell_override
                               else esc.get("min_dwell", 2.0)),
                    max_dwell=(self._dwell_override[1] if self._dwell_override
                               else esc.get("max_dwell", 5.0)),
                    min_gap=esc.get("min_gap", 2.0),
                    gap_tolerance=esc.get("gap_tolerance", 0.5),
                    levels=((1, lv["low"]), (2, lv["mid"]), (3, lv["high"]))),
                skeleton=skeleton, loiter=loiter)
        return self._tracks[tid]

    def _recycle_stale(self, now: float) -> None:
        """回收消失超過 recycle_sec 的 track(buffer、狀態機、警報、平滑器)。"""
        stale = [tid for tid, st in self._tracks.items()
                 if now - st.last_seen > self.recycle_sec]
        for tid in stale:
            st = self._tracks.pop(tid)
            if st.buffer is not None:
                st.buffer.reset()
            self.smoother.remove(tid)
            self.alarm.remove(tid)

    # ---------- 單次取樣步 ----------

    @torch.no_grad()
    def step(self, frame: np.ndarray, timestamp: float) -> Dict[int, dict]:
        """處理一張取樣影格,回傳每個 track 的狀態摘要(供繪圖)。"""
        if self.skeleton_enabled:
            dets, all_kpts = self.detector.detect(frame)
        else:
            dets = self.detector.detect(frame)
            all_kpts = None
        tracked = self.tracker.update(dets)

        results: Dict[int, dict] = {}
        rois, tids, boxes = [], [], []
        for tid, bbox in tracked:
            st = self._get_track_state(tid)
            st.last_seen = timestamp
            # 骨架:以原始偵測框 IoU 配對關鍵點
            if all_kpts is not None:
                k = _best_iou(bbox, dets[:, :4])
                st.last_kpts = all_kpts[k] if k is not None else None
                # 節點進滾動緩衝(複製一份:偵測器可能重用同一塊記憶體)
                st.pose_hist.append(
                    (timestamp, None if st.last_kpts is None else
                     np.asarray(st.last_kpts, np.float32).copy()))
            smoothed = self.smoother.update(tid, bbox)
            # 移動量恆常累計(排除與否由開關決定,開關切換即時反映)
            if st.move_gate is not None:
                st.moving = st.move_gate.update(timestamp, smoothed)
            # 在場型態(經過/徘徊/等待)恆常更新:純軌跡幾何,
            # 與骨架分支是否啟用無關
            if st.presence is not None:
                st.presence_state = st.presence.update(timestamp, smoothed)
                # 徘徊通報:每個 track 只報一次。不因狀態在
                # 徘徊/等待門檻附近抖動而重複通報 —— 人還在原地,
                # 一次就夠;離場後 track 回收,重新進場才會再報
                for kind, on, done in (
                        (PresenceClassifier.WANDERING,
                         self.wander_alert_enabled, "wander_notified"),
                        (PresenceClassifier.WAITING,
                         self.wait_alert_enabled, "wait_notified")):
                    if (on and not getattr(st, done)
                            and st.presence_state == kind):
                        setattr(st, done, True)
                        if self.on_presence is not None:
                            stay, path, _s, _v = st.presence.stats()
                            self.on_presence(tid, stay, path, kind)
            boxes.append(smoothed)
            tids.append(tid)
            if self.use_model:
                roi = crop_upper_body(
                    frame, smoothed,
                    aspect_ratio=self.roi_cfg["aspect_ratio"],
                    upper_body_ratio=self.roi_cfg["upper_body_ratio"],
                    out_size=self.roi_cfg["out_size"])
                rois.append(preprocess_roi(roi))

        # 骨架分支:規則推斷階段 → 餵狀態機與事件計數器
        # (背向時估計器內部棄權:回報背景、不產生 S2)
        if self.skeleton_enabled:
            for i, tid in enumerate(tids):
                st = self._tracks[tid]
                stage, d, ori = st.skeleton.update(st.last_kpts)
                st.last_stage = stage
                st.last_dnorm = d
                st.ori_hist.append((timestamp, ori))
                st.state_machine.push(stage, timestamp)
                # 移動排除開啟且移動中:不計手到嘴事件(走動時手部
                # 擺動易誤判;餵背景讓進行中的停留正常結算)
                excluded = self.move_gate_enabled and st.moving
                episode = st.counter.update(3 if excluded else stage,
                                            timestamp)
                if episode is not None and self.on_event is not None:
                    dwell, counted, reason = episode
                    self.on_event(tid, dwell, counted, reason)
                self._track_approach(tid, st, stage, d, timestamp)
                # 逗留偵測:手腕可見度 + 位移
                if st.loiter is not None:
                    k = st.last_kpts
                    conf = self.skel_cfg.get("kpt_conf", 0.3)
                    wrist_vis = (k is not None and
                                 (k[L_WRI, 2] >= conf or k[R_WRI, 2] >= conf))
                    st.loitering = st.loiter.update(timestamp, boxes[i],
                                                    wrist_vis)

        net_scores = None
        if self.use_model and rois:
            # 所有 track 的 ROI 合成一個 batch,backbone 只前向一次
            batch = torch.stack(rois).to(self.device)
            feats = self.model.extract_feature(batch).cpu()  # (N, C, H', W')

            shorts, longs = [], []
            for i, tid in enumerate(tids):
                st = self._tracks[tid]
                st.buffer.push(feats[i])
                shorts.append(st.buffer.get_short())
                longs.append(st.buffer.get_long())

            # continual inference:僅重算時序頭
            out = self.model.forward_buffers(
                torch.stack(shorts).to(self.device),
                torch.stack(longs).to(self.device))
            net_scores = out["cycle_score"].cpu()

            if not self.skeleton_enabled:
                # 無骨架時,狀態機/計數器以網路階段頭為階段來源
                stage_ids = out["stage_logits"].argmax(dim=1).cpu()
                for i, tid in enumerate(tids):
                    st = self._tracks[tid]
                    st.last_stage = int(stage_ids[i])
                    st.state_machine.push(st.last_stage, timestamp)
                    st.counter.update(st.last_stage, timestamp)

        # 融合與警報:cycle = w_sm × 次數警戒分數 + w_net × 網路分數
        # 背向時:骨架已棄權(無事件),且紅色警報被閘門擋下——
        # 網路分數若仍偏高,分流為橘色「無法確認」警示 + hard case 存檔
        w = self.sm_cfg["weights"]
        unv = self.cfg.get("unverified", {})
        mode = self.method.stage1
        for i, tid in enumerate(tids):
            st = self._tracks[tid]
            if mode == "rule":
                cyc = st.counter.score()          # 純規則:只看次數警戒
            elif mode == "network":
                if net_scores is None:
                    continue
                cyc = float(net_scores[i])        # 純網路:只看外觀分數
            elif net_scores is not None:
                cyc = cycle_score(st.counter.score(), float(net_scores[i]),
                                  w["state_machine"], w["network"])
            elif self.skeleton_enabled:
                cyc = st.counter.score()  # 純骨架模式(無模型)
            else:
                continue

            is_back = (st.back_fraction(timestamp,
                                        unv.get("window_sec", 5.0))
                       >= unv.get("back_frac", 0.6))
            # 講電話姿態:手舉著超過 max_dwell 仍未放下(即時判定)
            st.phone = (st.counter.max_dwell is not None
                        and st.counter.ongoing_dwell(timestamp)
                        > st.counter.max_dwell)
            # 紅色警報條件:非背向、非移動中(若開啟排除)、
            # 非講電話姿態、事件次數達門檻、P 持續超過觸發線
            # count_gate=False 的方法(純外觀網路基準線)不套次數規則,
            # 否則「純網路」實際上是被規則綁著跑,失去對照的意義
            enough_events = (st.counter.count() >= self.min_events
                             if self.method.count_gate else True)
            waiting_ok = (not self.smoking_requires_waiting
                          or st.presence_state == PresenceClassifier.WAITING)
            allow = (self.smoking_alarm_enabled
                     and waiting_ok
                     and not is_back
                     and not (self.move_gate_enabled and st.moving)
                     and not st.phone
                     and enough_events)
            # 講清楚為什麼沒通報:次數夠了卻卡在型態,不說的話看起來像
            # 偵測壞掉(等待要在場滿 long_stay 秒才成立,預設 20 秒)
            if (self.smoking_alarm_enabled and enough_events
                    and not waiting_ok and self.on_log is not None
                    and not st.wait_gate_logged):
                st.wait_gate_logged = True
                self.on_log(
                    f"track {tid} 次數已達標,但在場型態是"
                    f"「{PRESENCE_NAMES.get(st.presence_state) or '判定中'}」"
                    f",只有「等待」才判抽菸 → 不通報")
            # 次數主導:條件全過時分數直接推滿,P 快速進入觸發區,
            # 經 sustain 秒確認後通報(事件過期後 P 自然衰退解除)
            if self.count_driven and allow and self.method.count_gate:
                cyc = 1.0
            was_active = st.alarm_active
            st.last_P, st.alarm_active = self.alarm.update(
                tid, cyc, timestamp, frame, allow_trigger=allow)
            # 新觸發 → 送第二階段複核(背景執行緒);解除 → 清掉舊結果
            if st.alarm_active and not was_active:
                self._submit_verify(tid, st, timestamp)
            elif was_active and not st.alarm_active:
                st.verify = None
                st.verify_pending = False

            st.unverified = (is_back and net_scores is not None
                             and float(net_scores[i])
                             >= unv.get("net_score_min", 0.7))
            # hard case 存檔:每 track 有冷卻時間,避免同一情境重複存
            if st.unverified and (timestamp - st.last_hardcase_t
                                  >= unv.get("save_cooldown_sec", 30.0)):
                st.last_hardcase_t = timestamp
                self._save_hard_case(tid, float(net_scores[i]),
                                     timestamp, frame)

        for i, tid in enumerate(tids):
            st = self._tracks[tid]
            ori = st.ori_hist[-1][1] if st.ori_hist else "unknown"
            results[tid] = {
                "bbox": boxes[i], "stage": st.last_stage,
                "P": st.last_P, "alarm": st.alarm_active,
                "kpts": st.last_kpts, "d_norm": st.last_dnorm,
                "events": st.counter.count(),
                "level": st.counter.score(),
                "loiter": st.loitering,
                "presence": st.presence_state,
                # 徘徊警示:只在使用者開啟時成立(橘色,不影響紅色警報)
                "wander_alert": (self.wander_alert_enabled
                                 and st.presence_state
                                 == PresenceClassifier.WANDERING),
                "wait_alert": (self.wait_alert_enabled
                               and st.presence_state
                               == PresenceClassifier.WAITING),
                "orientation": ori,
                "unverified": st.unverified,
                "moving": self.move_gate_enabled and st.moving,
                "phone": st.phone,
                # 第二階段複核:complete 前為 pending,無 stage2 的方法為 None
                "verify": ("pending" if st.verify_pending else
                           st.verify.status if st.verify else None),
                "verify_top": st.verify.top if st.verify else None,
                "verify_smoking": st.verify.smoking if st.verify else None,
            }

        self._recycle_stale(timestamp)
        return results

    # ---------- 第二階段複核 ----------

    def _submit_verify(self, tid: int, st: TrackState,
                       timestamp: float) -> None:
        """把這個 track 的節點滾動窗丟給複核器(背景執行緒)。

        取的是**觸發時刻往前**的窗,不等後續影格:紅色警報要 ≥3 次手到嘴
        事件加 sustain 秒才成立,證據早就落在過去了。等未來幾秒只會延後
        結果,換不到新資訊。
        """
        if self.verifier is None or self._verify_pool is None:
            return
        kpts, span = verify_mod.pose_window(
            list(st.pose_hist), timestamp, self.verify_window_sec,
            float(self.cfg["sampling"]["target_fps"]))
        st.verify_pending = True
        st.verify = None
        fps = float(self.cfg["sampling"]["target_fps"])

        def work():
            try:
                res = self.verifier.verify(kpts, fps, span_sec=span)
            except Exception as e:      # 複核失敗絕不可影響第一階段警報
                res = verify_mod.VerifyResult(
                    status=verify_mod.ABSTAIN, span_sec=span,
                    reason=f"複核發生錯誤:{e}")
            st.verify = res
            st.verify_pending = False
            if self.on_verify is not None:
                self.on_verify(tid, res)

        self._verify_pool.submit(work)

    def close(self) -> None:
        """釋放背景資源。GUI 每次「開始」都會建一條新管線,不收的話
        每停一次就留下一個閒置的複核執行緒。"""
        if self._verify_pool is not None:
            self._verify_pool.shutdown(wait=False)
            self._verify_pool = None

    def _track_approach(self, tid: int, st: TrackState, stage: int,
                        d: Optional[float], timestamp: float) -> None:
        """「接近未達標」診斷:手靠近臉(d < 門檻+0.5)但整段都沒觸發 S2
        → 結束時記錄最近距離與門檻差距,回答「為什麼沒開始計時」。"""
        near = self.skel_cfg.get("near_ratio", 0.9)
        if d is not None and d < near + 0.5:
            if not st.approach_active:
                st.approach_active = True
                st.approach_start_t = timestamp
                st.approach_min_d = float("inf")
                st.approach_saw_s2 = False
            st.approach_last_t = timestamp
            st.approach_min_d = min(st.approach_min_d, d)
            st.approach_saw_s2 |= (stage == 1)
        elif st.approach_active and \
                timestamp - st.approach_last_t > 0.8:
            duration = st.approach_last_t - st.approach_start_t
            if (not st.approach_saw_s2 and duration >= 0.5
                    and self.on_log is not None):
                if st.approach_min_d < near:
                    self.on_log(
                        f"track {tid} 手在臉部 {duration:.1f} 秒但未經舉手"
                        f"動作(疑似手被遮擋、腕點誤定位)→ 不採信")
                else:
                    self.on_log(
                        f"track {tid} 手接近臉 {duration:.1f} 秒,"
                        f"最近距離 {st.approach_min_d:.2f} 未達門檻 {near:.2f}"
                        f" → 未開始計時")
            st.approach_active = False

    def set_dwell_window(self, min_dwell: float, max_dwell: float) -> None:
        """即時調整停留窗口(套用到現有與之後建立的所有 track)。"""
        self._dwell_override = (float(min_dwell), float(max_dwell))
        for st in self._tracks.values():
            st.counter.min_dwell = float(min_dwell)
            st.counter.max_dwell = float(max_dwell)

    def _save_hard_case(self, tid: int, net_score: float,
                        timestamp: float, frame: np.ndarray) -> None:
        """背向高分案例存檔:人工檢視 + 自錄資料集的 hard case 來源。"""
        from utils import imwrite
        save_dir = self.cfg.get("unverified", {}).get("save_dir",
                                                      "./hard_cases")
        os.makedirs(save_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = os.path.join(save_dir,
                            f"back_track{tid}_{stamp}_net{net_score:.2f}.jpg")
        if imwrite(path, frame):
            print(f"[無法確認] track {tid} 背向且網路分數 {net_score:.2f},"
                  f"已存 hard case:{path}")

    # ---------- 主迴圈 ----------

    def run(self, source, display: bool = True,
            save_video: Optional[str] = None) -> None:
        """讀取影片/攝影機/RTSP 串流並持續推理。

        source:檔案路徑、攝影機編號,或 rtsp:// URL
        (RTSP 憑證特殊字元自動編碼、TCP 傳輸、斷線自動重連)。
        """
        from inference.stream import VideoSource
        target_fps = self.cfg["sampling"]["target_fps"]
        # 告訴來源我們實際要幾 fps:HLS 的佇列會在收幀時就抽稀到這個
        # 量級,不然整段 30 fps 全存進記憶體
        vs = VideoSource(source, sample_fps=target_fps,
                         **self.cfg.get("stream", {}))

        sample_every = max(1, round(vs.fps / target_fps))  # 檔案:幀計數取樣
        min_interval = 1.0 / target_fps                     # live:時間取樣
        print(f"[管線] 來源={vs.kind} fps={vs.fps:.1f},"
              f"特徵率目標 {target_fps} fps"
              + ("(live:時間取樣 + 只取最新影格)" if vs.is_live else
                 f"(每 {sample_every} 幀取樣)"))

        writer = None
        frame_idx = 0
        last_proc = float("-inf")
        results: Dict[int, dict] = {}
        try:
            while True:
                frame, ts = vs.read()
                if frame is None:
                    if vs.is_live:
                        continue  # 讀取逾時(重連中),持續等待
                    break         # 檔案播畢

                if getattr(vs, "pre_sampled", False):
                    do_step = True      # 來源已抽稀到 target_fps,不再濾
                elif vs.is_live:
                    do_step = ts - last_proc >= min_interval
                else:
                    do_step = frame_idx % sample_every == 0
                if do_step:
                    results = self.step(frame, ts)
                    last_proc = ts

                vis = draw_overlay(frame, results)
                if save_video and writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                        vs.fps, (w, h))
                if writer is not None:
                    writer.write(vis)
                if display:
                    cv2.imshow("smoking-detection", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                frame_idx += 1
        finally:
            vs.release()
            self.close()
            if writer is not None:
                writer.release()
                print(f"[管線] 影片已存:{save_video}")
            if display:
                cv2.destroyAllWindows()


_STAGE_NAMES = {0: "S1 raise", 1: "S2 mouth", 2: "S3 lower", 3: "bg"}


def _best_iou(box: np.ndarray, cands: np.ndarray,
              min_iou: float = 0.3) -> Optional[int]:
    """回傳與 box IoU 最大且 ≥ min_iou 的候選索引(無則 None)。"""
    if len(cands) == 0:
        return None
    x1 = np.maximum(box[0], cands[:, 0])
    y1 = np.maximum(box[1], cands[:, 1])
    x2 = np.minimum(box[2], cands[:, 2])
    y2 = np.minimum(box[3], cands[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = ((cands[:, 2] - cands[:, 0]) * (cands[:, 3] - cands[:, 1]))
    iou = inter / np.clip(area_a + area_b - inter, 1e-9, None)
    k = int(iou.argmax())
    return k if iou[k] >= min_iou else None


def alert_level_name(level: float) -> str:
    """警戒分數 → 等級名稱(畫面用英文,GUI 面板用中文)。"""
    if level >= 0.8:
        return "HIGH"
    if level >= 0.5:
        return "MID"
    if level >= 0.2:
        return "LOW"
    return ""


# 在場型態的畫面標籤(畫面用英文,GUI 面板與記錄用中文)
_PRESENCE_TAGS = {"passing": "PASS", "running": "RUN",
                  "wandering": "WANDER", "waiting": "WAIT"}
PRESENCE_NAMES = {"passing": "經過", "running": "經過(跑)",
                  "wandering": "徘徊", "waiting": "等待",
                  "unknown": ""}


def draw_overlay(frame: np.ndarray, results: Dict[int, dict],
                 show_skeleton: bool = True) -> np.ndarray:
    """畫面疊加:框、ID、階段、次數警戒等級、逗留標記、P_t 條、骨架。"""
    vis = frame.copy()
    for tid, r in results.items():
        if show_skeleton and r.get("kpts") is not None:
            draw_skeleton(vis, r["kpts"], stage_id=r["stage"],
                          d_norm=r.get("d_norm"))
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        level = r.get("level", 0.0)
        # 降級不否決:第二階段判「非抽菸」時警報**仍在**,只是由紅轉橘
        # 待人工複查;複核中或棄權一律維持紅色
        downgraded = r["alarm"] and r.get("verify") == "review"
        if r["alarm"] and not downgraded:
            color = (0, 0, 255)          # 紅:警報
        elif (downgraded or r.get("unverified") or r.get("loiter")
              or r.get("wander_alert") or r.get("wait_alert")):
            color = (0, 165, 255)   # 橘:無法確認 / 逗留 / 徘徊 / 等待警示
        elif level >= 0.5:
            color = (0, 200, 255)        # 黃:中警戒
        else:
            color = (0, 200, 0)          # 綠:一般
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        label = f"ID{tid} {_STAGE_NAMES.get(r['stage'], '?')}"
        if r.get("orientation") == "back":
            label += "(back)"
        n = r.get("events", 0)
        lv = alert_level_name(level)
        if n:
            label += f" x{n}"
        if lv:
            label += f" [{lv}]"
        if r.get("unverified"):
            label += " UNVERIFIED"
        if r.get("loiter"):
            label += " LOITER"
        if r.get("moving"):
            label += " MOVING"
        tag = _PRESENCE_TAGS.get(r.get("presence"))
        if tag:
            label += f" {tag}"
        if r.get("phone"):
            label += " PHONE"
        if r["alarm"]:
            label += " SMOKING!"
        v = r.get("verify")
        if r["alarm"] and v:
            label += {"pending": " VERIFYING", "confirmed": " CONFIRMED",
                      "review": " REVIEW", "abstain": " NO-SKEL"}.get(v, "")
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # P_t 置信度條(框左側豎條)
        bar_h = max(1, y2 - y1)
        fill = int(bar_h * min(1.0, max(0.0, r["P"])))
        cv2.rectangle(vis, (x1 - 10, y1), (x1 - 4, y2), (80, 80, 80), 1)
        cv2.rectangle(vis, (x1 - 10, y2 - fill), (x1 - 4, y2), color, -1)
        cv2.putText(vis, f"{r['P']:.2f}", (x1 - 10, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def main():
    parser = argparse.ArgumentParser(description="抽菸行為偵測即時管線")
    parser.add_argument("--source", default="0", help="影片路徑或攝影機編號")
    parser.add_argument("--infer-config", default="configs/inference.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--model-ckpt", default=None, help="模型權重路徑")
    parser.add_argument("--no-model", action="store_true",
                        help="只跑偵測+追蹤(M1 骨架驗證)")
    parser.add_argument(
        "--method", default=methods_registry.DEFAULT_KEY,
        choices=methods_registry.keys() + ["list"],
        help="判定方法(見 inference/methods.py);list = 只列出可選項目")
    parser.add_argument("--save-video", default=None, help="輸出 mp4 路徑")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    if args.method == "list":
        for m in methods_registry.METHODS:
            mark = "" if m.available else f"  [缺 {', '.join(m.missing())}]"
            print(f"{m.key:18} {m.name}{mark}\n{'':18} {m.desc}")
        return

    method = methods_registry.get(args.method)
    if method.missing():
        parser.error(f"方法「{method.name}」缺少權重:"
                     f"{', '.join(method.missing())}")
    if args.no_model and method.needs_appearance:
        parser.error(f"--no-model 與方法「{method.name}」衝突,"
                     f"要不用外觀網路請改 --method rule")
    infer_cfg = load_config(args.infer_config)
    # 要不要載外觀網路由方法決定,不再由旗標決定
    model_cfg = (load_config(args.model_config)
                 if method.needs_appearance else None)
    pipeline = SmokingDetectionPipeline(
        infer_cfg, model_cfg, ckpt_path=args.model_ckpt,
        use_model=method.needs_appearance, method=method)
    print(f"[管線] 判定方法:{pipeline.method.key} — {pipeline.method.name}")
    pipeline.run(args.source, display=not args.no_display,
                 save_video=args.save_video)


if __name__ == "__main__":
    main()
