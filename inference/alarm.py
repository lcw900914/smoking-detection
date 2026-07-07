"""EMA 置信度累積 + 雙門檻(hysteresis)警報。

- P_t = α·P_{t-1} + (1-α)·cycle_score,per-track 獨立維護
- P_t > trigger 且持續 sustain_sec 秒 → 觸發警報
- P_t < release → 解除警報(雙門檻避免抖動)
- 觸發時執行 callback:預設 console log + 截圖存檔
  (介面保留,之後可替換為通知服務)
"""
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class _TrackAlarmState:
    """單一 track 的警報狀態。"""
    P: float = 0.0                 # EMA 置信度
    above_since: Optional[float] = None  # 首次越過觸發門檻的時間
    active: bool = False           # 警報是否觸發中


def default_alarm_callback(track_id: int, P: float, timestamp: float,
                           frame: Optional[np.ndarray],
                           snapshot_dir: str = "./alarms") -> None:
    """預設警報 callback:console log + 截圖存檔。"""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    print(f"[警報] track {track_id} 疑似抽菸行為!"
          f"P_t={P:.3f} t={timestamp:.1f}s")
    if frame is not None:
        from utils import imwrite
        os.makedirs(snapshot_dir, exist_ok=True)
        path = os.path.join(snapshot_dir,
                            f"alarm_track{track_id}_{stamp}.jpg")
        if imwrite(path, frame):
            print(f"[警報] 截圖已存:{path}")


class AlarmManager:
    """多 track 警報管理器。

    Args:
        ema_alpha: EMA 係數 α(預設 0.9)
        trigger_threshold / release_threshold: 雙門檻(0.75 / 0.4)
        sustain_sec: 觸發需持續秒數(預設 3.0)
        callback: 觸發時呼叫 (track_id, P, timestamp, frame)
    """

    def __init__(self, ema_alpha: float = 0.9,
                 trigger_threshold: float = 0.75,
                 release_threshold: float = 0.4,
                 sustain_sec: float = 3.0,
                 snapshot_dir: str = "./alarms",
                 callback: Optional[Callable] = None):
        assert release_threshold < trigger_threshold
        self.alpha = ema_alpha
        self.trigger = trigger_threshold
        self.release = release_threshold
        self.sustain_sec = sustain_sec
        self.snapshot_dir = snapshot_dir
        self.callback = callback or (
            lambda tid, P, t, fr: default_alarm_callback(
                tid, P, t, fr, snapshot_dir=self.snapshot_dir))
        self._states: Dict[int, _TrackAlarmState] = {}

    def update(self, track_id: int, cycle_score: float, timestamp: float,
               frame: Optional[np.ndarray] = None,
               allow_trigger: bool = True):
        """更新一個 track 的置信度,回傳 (P_t, 警報是否觸發中)。

        allow_trigger=False 時 EMA 照常更新、解除照常,但不允許
        「新觸發」——用於背向等無法驗證的情境(呼叫端另行分流成
        橘色無法確認警示)。
        """
        st = self._states.setdefault(track_id, _TrackAlarmState())
        st.P = self.alpha * st.P + (1.0 - self.alpha) * float(cycle_score)

        if st.active:
            # 觸發中:降到解除門檻以下才解除
            if st.P < self.release:
                st.active = False
                st.above_since = None
                print(f"[警報解除] track {track_id} P_t={st.P:.3f}")
        elif not allow_trigger:
            st.above_since = None  # 不可驗證期間不累積觸發時間
        else:
            if st.P > self.trigger:
                if st.above_since is None:
                    st.above_since = timestamp
                elif timestamp - st.above_since >= self.sustain_sec:
                    st.active = True
                    self.callback(track_id, st.P, timestamp, frame)
            else:
                st.above_since = None
        return st.P, st.active

    def get_state(self, track_id: int) -> _TrackAlarmState:
        return self._states.setdefault(track_id, _TrackAlarmState())

    def remove(self, track_id: int) -> None:
        """track 回收時移除其狀態(含 EMA 重置)。"""
        self._states.pop(track_id, None)

    def reset(self) -> None:
        self._states.clear()
