"""骨架輔助分支:以手腕-鼻子距離時序,用規則推斷動作階段。

計畫書 §4.5:對每個 track 抽取關鍵點,計算 wrist-to-nose 距離
(以肩寬正規化,對距離/解析度不敏感),依規則推斷:

    S1 舉手:距離在短時間內明顯縮小(手朝臉靠近)
    S2 嘴部停留:距離 < near_ratio × 肩寬
    S3 放下:距離自嘴部附近明顯拉大

推斷出的階段餵入既有的 StageStateMachine 檢查 S1→S2→S3 順序與
S2 停留時長,分數與網路 cycle score 做 late fusion——
可解釋(畫面直接看得到手到嘴的線)且不需階段標籤訓練。
"""
from collections import deque
from typing import Optional, Tuple

import numpy as np

# COCO 關鍵點索引
NOSE, L_SHO, R_SHO, L_WRI, R_WRI = 0, 5, 6, 9, 10
L_HIP, R_HIP = 11, 12

# 繪圖用骨架連線(上半身為主)
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),        # 手臂
    (5, 6), (5, 11), (6, 12), (11, 12),     # 軀幹
    (0, 5), (0, 6),                          # 頭-肩
    (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
]


def estimate_orientation(kpts: np.ndarray, kpt_conf: float = 0.3,
                         nose_conf: float = 0.5) -> str:
    """由關鍵點推斷人物朝向:"front" / "back" / "unknown"。

    線索(依可靠度排序):
    1. 肩膀左右順序:COCO 關鍵點有左右語意——正對鏡頭時,
       人的右肩(kpt 6)在畫面 x 較小側;背對時左右顛倒。
       需左右肩分離夠大(側面時符號不穩,不採用)。
    2. 鼻點置信度:背對時鼻子多為低置信的幻覺點。
    """
    sho_ok = (kpts[L_SHO, 2] >= kpt_conf and kpts[R_SHO, 2] >= kpt_conf)
    if sho_ok:
        dx = float(kpts[R_SHO, 0] - kpts[L_SHO, 0])  # 背對時 > 0
        # 分離度基準:肩寬與軀幹高取大(側面肩寬被壓縮)
        scale = float(abs(dx))
        if kpts[L_HIP, 2] >= kpt_conf and kpts[R_HIP, 2] >= kpt_conf:
            mid_sho = (kpts[L_SHO, :2] + kpts[R_SHO, :2]) / 2
            mid_hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
            scale = max(scale, 0.55 * float(np.linalg.norm(mid_sho - mid_hip)))
        if scale > 1e-3 and dx > 0.3 * scale:
            return "back"      # 肩序顛倒且分離明確 → 背對(優先於鼻點)
    if kpts[NOSE, 2] >= nose_conf:
        return "front"
    return "unknown"


class SkeletonStageEstimator:
    """單一 track 的骨架階段推斷器。

    背向棄權:背對鏡頭時,腕-鼻 2D 距離失去意義(深度不可知,
    打鍵盤/撥頭髮都會投影在頭附近),一律回報 background 且不產生
    S2 訊號——「沒證據」不等於「有反證」,也不等於「有證據」。

    Args:
        near_ratio: 腕-鼻距離 < 此比例×身體尺度 視為「手在臉部」(S2)
        move_ratio: 0.6 秒內距離變化超過此比例 視為舉手/放下
        kpt_conf: 一般關鍵點最低置信度
        nose_conf: 鼻點專用門檻(較嚴,背面幻覺鼻點過不了)
        fps: 取樣率(決定回看視窗)
    """

    def __init__(self, near_ratio: float = 0.6, move_ratio: float = 0.35,
                 kpt_conf: float = 0.3, nose_conf: float = 0.5,
                 min_scale_px: float = 24.0, kpt_err_px: float = 4.0,
                 rise_margin: float = 0.5, fps: float = 10.0):
        self.near = near_ratio
        self.move = move_ratio
        self.kpt_conf = kpt_conf
        self.nose_conf = nose_conf
        # 距離自適應:身體尺度太小(太遠)→ 關鍵點雜訊佔比過大,棄權;
        # 門檻加上「像素誤差 ÷ 尺度」餘裕,越遠的人相對門檻自動放寬
        self.min_scale_px = min_scale_px
        self.kpt_err_px = kpt_err_px
        # 「由遠而近」武裝機制:S2 只在手曾離臉 ≥ near+rise_margin 後採信。
        # 手被遮擋(背在身後/插口袋)時,姿態模型會以高置信度把腕點
        # 幻覺在身體輪廓上(衣領附近,d≈0.7),恰好落在門檻內——
        # 但幻覺點永遠不會「先遠離再靠近」,真的舉手一定會。
        self.rise_margin = rise_margin
        self._armed = False    # 手已離臉夠遠,下次進入臉部區域可信
        self._in_s2 = False    # 進行中的可信 S2
        # 腕點連續不可見(手出畫面/垂下)≥ 此幀數亦視為離臉 → 武裝;
        # 幻覺腕點恆定「可見」,不會走這條路
        self._none_arm_frames = max(2, int(0.5 * fps))
        self._none_count = 0   # 連續「正面但量不到腕點」幀數(武裝用)
        # 手離開臉部區域(d ≥ near_eff)持續這麼多幀也可武裝,不必退到
        # near+rise_margin 那麼遠。手肘撐著、手在胸口與嘴之間小幅來回的
        # 抽菸姿勢永遠退不到 1.4 身體尺度,原本第一口之後就再也武裝不了,
        # 次數卡在 1、到不了 min_events。幻覺腕點的 d 恆定落在 near 以內,
        # 走不到這條路,所以放寬不會把它放進來
        self._above_arm_frames = max(2, int(1.0 * fps))
        self._above_count = 0
        self._gap_count = 0    # 連續無有效距離量測幀數(任何棄權路徑)
        self.lookback = max(2, int(0.6 * fps))
        self.lookback_sec = 0.6
        # (t, d_norm, 用的是哪隻腕點)。存時間而不是只存值:索引式回看
        # 假設呼叫頻率固定,但等待閘門會整段跳過 update(),轉回候選時
        # 「6 格前」可能是好幾秒前。存腕點編號是因為 d 取左右較近者——
        # 換手時 delta 比較的是兩隻不同的手,會憑空生出 S1/S3
        self._hist: deque = deque(maxlen=int(3 * fps))

    def _body_scale(self, kpts: np.ndarray) -> Optional[float]:
        """身體尺度基準:max(肩寬, 0.55×軀幹高)。

        側面視角時肩寬被透視壓縮(可縮到近 0),但軀幹高
        (肩中點→髖中點)對水平旋轉不變,兩者取大以涵蓋正面與側面。
        """
        if kpts[L_SHO, 2] < self.kpt_conf or kpts[R_SHO, 2] < self.kpt_conf:
            return None
        shoulder_w = float(np.linalg.norm(kpts[L_SHO, :2] - kpts[R_SHO, :2]))
        scale = shoulder_w
        if kpts[L_HIP, 2] >= self.kpt_conf and kpts[R_HIP, 2] >= self.kpt_conf:
            mid_sho = (kpts[L_SHO, :2] + kpts[R_SHO, :2]) / 2
            mid_hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
            torso_h = float(np.linalg.norm(mid_sho - mid_hip))
            scale = max(scale, 0.55 * torso_h)
        return scale if scale > 1e-3 else None

    def _wrist_nose_dist(self, kpts: np.ndarray):
        """正規化腕-鼻距離(取左右腕較近者)與用到的腕點編號。

        鼻點採較嚴門檻 nose_conf:S2 的語意依賴鼻子位置可信。
        回傳 (d, wrist) 或 (None, None)——wrist 給呼叫端判斷有沒有換手。
        """
        if kpts[NOSE, 2] < self.nose_conf:
            return None, None
        scale = self._body_scale(kpts)
        if scale is None:
            return None, None
        dists = [
            (float(np.linalg.norm(kpts[w, :2] - kpts[NOSE, :2])) / scale, w)
            for w in (L_WRI, R_WRI) if kpts[w, 2] >= self.kpt_conf
        ]
        return min(dists) if dists else (None, None)

    def update(self, kpts: Optional[np.ndarray],
               timestamp: Optional[float] = None
               ) -> Tuple[int, Optional[float], str]:
        """推入一幀關鍵點 (17, 3),回傳 (stage_id, d_norm, orientation)。

        stage_id 與全專案一致:0=S1、1=S2、2=S3、3=背景。
        kpts 為 None 或判定背向時棄權:回報背景、不產生 S2。

        timestamp 省略時退回索引式回看(呼叫頻率固定才成立);傳入時
        S1/S3 的「0.6 秒前」以真實時間回查,呼叫被跳過也不會錯亂。
        """
        self._now = timestamp
        if kpts is None:
            # 無姿態(track 偵測閃爍):可武裝——真動作中斷後重現
            # 手在嘴仍該續判;幻覺案例的姿態是恆定存在的
            self._abstain(arm=True)
            return 3, None, "unknown"

        ori = estimate_orientation(kpts, self.kpt_conf, self.nose_conf)
        if ori == "back":
            # 背向棄權:2D 腕-鼻距離不可信,不推 S2(也不倒扣);
            # 不武裝——背向時手的真實位置未知
            self._abstain(arm=False)
            return 3, None, ori

        # 太遠棄權:身體尺度不足,關鍵點量化誤差會淹沒真實距離
        scale = self._body_scale(kpts)
        if scale is not None and scale < self.min_scale_px:
            self._abstain(arm=False)
            return 3, None, ori

        d, wrist = self._wrist_nose_dist(kpts)
        if d is None:
            # 正面但量不到腕-鼻距離(手出畫面/垂下未偵測):
            # 持續一段時間即可武裝——手顯然不在臉部
            self._abstain(arm=True)
            return 3, None, ori
        self._none_count = 0
        self._gap_count = 0
        self._hist.append((timestamp, d, wrist))

        # 距離自適應門檻:加上關鍵點像素誤差的相對餘裕
        # (近距離 scale 大 → 餘裕趨近 0;遠距離自動放寬)
        near_eff = self.near + self.kpt_err_px / scale

        # S2:手在嘴部——僅在「由遠而近」(armed)時採信;
        # 每口之間手須放回遠處(≥ near+rise_margin)才能再武裝
        if d < near_eff:
            self._above_count = 0
            if self._in_s2 or self._armed:
                self._in_s2 = True
                self._armed = False
                return 1, d, ori
            return 3, d, ori   # 手在臉部但未經舉手:視為遮擋誤定位,不採信
        self._in_s2 = False
        if d >= near_eff + self.rise_margin:
            self._armed = True      # 手已離臉夠遠(強證據,立刻武裝)
            self._above_count = 0
        else:
            # 只是離開臉部區域:持續夠久也算放下過(見 __init__ 的說明)
            self._above_count += 1
            if self._above_count >= self._above_arm_frames:
                self._armed = True

        # 與 0.6 秒前比較,判斷靠近(S1)或離開(S3)
        past = self._past_dist(timestamp, wrist)
        if past is not None:
            delta = past - d
            if delta > self.move and d < near_eff + 2 * self.move:
                return 0, d, ori   # 明顯縮小且已接近 → S1 舉手
            if -delta > self.move and past < near_eff + self.move:
                return 2, d, ori   # 自嘴部附近明顯拉大 → S3 放下
        return 3, d, ori

    def _past_dist(self, now: Optional[float], wrist: int) -> Optional[float]:
        """取約 0.6 秒前、**同一隻手**的距離;取不到回傳 None。

        換手不比較:d 取左右腕較近者,兩隻手的距離序列混在一起時,
        「較近的手換了一隻」會被讀成手快速靠近或遠離,憑空生出 S1/S3。
        托腮(一手固定在臉旁)同時另一手抬起,正好構成這個情境。
        """
        if now is None:
            # 沒有時間資訊:退回索引式回看(舊行為)
            if len(self._hist) <= self.lookback:
                return None
            t_p, d_p, w_p = self._hist[-1 - self.lookback]
            return d_p if (d_p is not None and w_p == wrist) else None
        # 由新到舊找第一個夠久遠的樣本;太久遠(> 2 倍回看)不採信,
        # 那多半是閘門或追蹤中斷造成的空窗,不是連續動作
        for t_p, d_p, w_p in reversed(self._hist):
            if t_p is None or d_p is None:
                continue
            age = now - t_p
            if age >= self.lookback_sec:
                if age > 2 * self.lookback_sec or w_p != wrist:
                    return None
                return d_p
        return None

    def _abstain(self, arm: bool) -> None:
        """棄權路徑的共同狀態維護。

        - 任何棄權持續超過容忍幀數 → 清 _in_s2(進行中的 S2 視為中斷,
          之後須重新武裝;否則殘留的 _in_s2 會讓幻覺腕點繞過武裝機制)
        - arm=True 的路徑(正面但量不到手)累積夠久 → 武裝
        """
        self._hist.append((getattr(self, "_now", None), None, -1))
        self._above_count = 0
        self._gap_count += 1
        if self._gap_count >= self._none_arm_frames:
            self._in_s2 = False
        if arm:
            self._none_count += 1
            if self._none_count >= self._none_arm_frames:
                self._armed = True
                self._in_s2 = False
        else:
            self._none_count = 0

    def reset(self) -> None:
        self._hist.clear()
        self._armed = False
        self._in_s2 = False
        self._none_count = 0
        self._gap_count = 0


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray,
                  stage_id: int = 3, d_norm: Optional[float] = None,
                  kpt_conf: float = 0.3) -> None:
    """在影格上就地畫骨架與腕-鼻連線(S2 時紅色高亮)。"""
    import cv2
    pts = kpts[:, :2].astype(int)
    vis = kpts[:, 2] >= kpt_conf

    for a, b in SKELETON_EDGES:
        if vis[a] and vis[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 200, 0), 2)
    for i in range(len(pts)):
        if vis[i]:
            cv2.circle(frame, tuple(pts[i]), 3, (0, 255, 255), -1)

    # 腕-鼻距離線:一般黃色、S2(手在嘴部)紅色加粗
    if vis[NOSE]:
        wrists = [w for w in (L_WRI, R_WRI) if vis[w]]
        if wrists:
            w = min(wrists, key=lambda k: np.linalg.norm(
                pts[k] - pts[NOSE]))
            color = (0, 0, 255) if stage_id == 1 else (0, 255, 255)
            cv2.line(frame, tuple(pts[w]), tuple(pts[NOSE]), color,
                     3 if stage_id == 1 else 1)
            if d_norm is not None:
                mid = ((pts[w] + pts[NOSE]) // 2)
                cv2.putText(frame, f"{d_norm:.2f}", tuple(mid),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
