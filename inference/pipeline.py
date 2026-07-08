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
                                     MovementGate)
from inference.skeleton import (SkeletonStageEstimator, draw_skeleton,
                                L_WRI, R_WRI)
from inference.alarm import AlarmManager
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
    last_kpts: Optional[np.ndarray] = None
    last_dnorm: Optional[float] = None
    ori_hist: deque = field(default_factory=deque)  # (t, orientation)

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
                 ckpt_path: Optional[str] = None, use_model: bool = True):
        self.cfg = infer_cfg
        self.use_model = use_model and model_cfg is not None

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
                    fps=self.cfg["sampling"]["target_fps"])
                lo = self.cfg.get("loiter", {})
                if lo.get("enabled", True):
                    loiter = LoiterDetector(
                        min_duration=lo.get("min_duration", 20.0),
                        move_ratio=lo.get("move_ratio", 0.6),
                        wrist_vis_max=lo.get("wrist_vis_max", 0.1))
            esc = self.cfg.get("escalation", {})
            lv = esc.get("levels", {"low": 0.2, "mid": 0.5, "high": 0.8})
            self._tracks[tid] = TrackState(
                buffer=buffer,
                state_machine=StageStateMachine(
                    window_sec=self.sm_cfg["window_sec"],
                    s2_min_dwell=self.sm_cfg["s2_min_dwell"]),
                move_gate=MovementGate(
                    max_heights=self.mg_cfg.get("max_heights", 3.0),
                    window_sec=self.mg_cfg.get("window_sec", 10.0)),
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
            smoothed = self.smoother.update(tid, bbox)
            # 移動量恆常累計(排除與否由開關決定,開關切換即時反映)
            if st.move_gate is not None:
                st.moving = st.move_gate.update(timestamp, smoothed)
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
        for i, tid in enumerate(tids):
            st = self._tracks[tid]
            if net_scores is not None:
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
            allow = (not is_back
                     and not (self.move_gate_enabled and st.moving)
                     and not st.phone
                     and st.counter.count() >= self.min_events)
            # 次數主導:條件全過時分數直接推滿,P 快速進入觸發區,
            # 經 sustain 秒確認後通報(事件過期後 P 自然衰退解除)
            if self.count_driven and allow:
                cyc = 1.0
            st.last_P, st.alarm_active = self.alarm.update(
                tid, cyc, timestamp, frame, allow_trigger=allow)

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
                "orientation": ori,
                "unverified": st.unverified,
                "moving": self.move_gate_enabled and st.moving,
                "phone": st.phone,
            }

        self._recycle_stale(timestamp)
        return results

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
        vs = VideoSource(source, **self.cfg.get("stream", {}))

        target_fps = self.cfg["sampling"]["target_fps"]
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

                if vs.is_live:
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


def draw_overlay(frame: np.ndarray, results: Dict[int, dict]) -> np.ndarray:
    """畫面疊加:框、ID、階段、次數警戒等級、逗留標記、P_t 條、骨架。"""
    vis = frame.copy()
    for tid, r in results.items():
        if r.get("kpts") is not None:
            draw_skeleton(vis, r["kpts"], stage_id=r["stage"],
                          d_norm=r.get("d_norm"))
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        level = r.get("level", 0.0)
        if r["alarm"]:
            color = (0, 0, 255)          # 紅:警報
        elif r.get("unverified") or r.get("loiter"):
            color = (0, 165, 255)        # 橘:無法確認 / 逗留警告
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
        if r.get("phone"):
            label += " PHONE"
        if r["alarm"]:
            label += " SMOKING!"
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
    parser.add_argument("--save-video", default=None, help="輸出 mp4 路徑")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    infer_cfg = load_config(args.infer_config)
    model_cfg = None if args.no_model else load_config(args.model_config)
    pipeline = SmokingDetectionPipeline(
        infer_cfg, model_cfg, ckpt_path=args.model_ckpt,
        use_model=not args.no_model)
    pipeline.run(args.source, display=not args.no_display,
                 save_video=args.save_video)


if __name__ == "__main__":
    main()
