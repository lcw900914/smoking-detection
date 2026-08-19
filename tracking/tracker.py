"""ByteTrack 多目標追蹤 wrapper(使用 supervision 套件實作)。

參考文獻
────────
  Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P.,
  Liu, W., Wang, X. (2022). ByteTrack: Multi-Object Tracking by
  Associating Every Detection Box. ECCV 2022.
  https://doi.org/10.1007/978-3-031-20047-2_1

本專案只是呼叫 supervision 的實作,沒有改動演算法。追蹤在這裡是
**感測器**不是判定器——它決定「這幾幀是同一個人」,不決定「這個人
在不在抽菸」(見 inference/methods.py 開頭)。
"""
from typing import List, Tuple

import numpy as np


class PersonTracker:
    """ByteTrack 追蹤器,輸入偵測框、輸出帶 track ID 的框。

    Args:
        track_activation_threshold / lost_track_buffer /
        minimum_matching_threshold: 對應 supervision.ByteTrack 參數
        frame_rate: 來源影片 fps(影響 lost buffer 換算)
    """

    def __init__(self, track_activation_threshold: float = 0.25,
                 lost_track_buffer: int = 30,
                 minimum_matching_threshold: float = 0.8,
                 frame_rate: int = 30):
        import supervision as sv  # 延遲載入

        self._sv = sv
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, detections: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """更新追蹤狀態。

        Args:
            detections: (N, 5) [x1, y1, x2, y2, conf](PersonDetector 輸出)

        Returns:
            list of (track_id, bbox_xyxy(4,))
        """
        sv = self._sv
        if len(detections) == 0:
            dets = sv.Detections.empty()
        else:
            dets = sv.Detections(
                xyxy=detections[:, :4],
                confidence=detections[:, 4],
                class_id=np.zeros(len(detections), dtype=int),
            )
        tracked = self.tracker.update_with_detections(dets)
        out = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            out.append((tid, tracked.xyxy[i].copy()))
        return out

    def reset(self) -> None:
        self.tracker.reset()
