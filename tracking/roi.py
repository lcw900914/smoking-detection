"""ROI 平滑與裁切。

- 對每個 track 的框做 EMA 平滑(β=0.8)
- 以人物中心為錨、固定 3:4(寬:高)長寬比,取上半身(框上緣至 60% 高)
- 框中心跳動超過閾值視為異常(偵測失誤或 ID switch 前兆),沿用前一幀平滑框
"""
from typing import Dict, Optional

import cv2
import numpy as np


class ROISmoother:
    """逐 track 的框 EMA 平滑器。

    smoothed = β * prev + (1-β) * new;跳動異常時忽略新框。
    """

    def __init__(self, beta: float = 0.8, jump_threshold: float = 0.5):
        """
        Args:
            beta: EMA 係數(越大越平滑)
            jump_threshold: 新框中心位移超過(前一框對角線 × 此值)視為異常
        """
        self.beta = beta
        self.jump_threshold = jump_threshold
        self._state: Dict[int, np.ndarray] = {}  # track_id → 平滑框 xyxy

    def update(self, track_id: int, bbox: np.ndarray) -> np.ndarray:
        """更新並回傳平滑後的框 (x1, y1, x2, y2)。"""
        bbox = np.asarray(bbox, dtype=np.float32)
        prev = self._state.get(track_id)
        if prev is None:
            self._state[track_id] = bbox.copy()
            return bbox.copy()

        # 跳動檢查:中心位移相對前一框對角線
        pc = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
        nc = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        diag = float(np.hypot(prev[2] - prev[0], prev[3] - prev[1]))
        if diag > 0 and float(np.linalg.norm(nc - pc)) > self.jump_threshold * diag:
            # 異常跳動:沿用前一幀平滑框
            return prev.copy()

        smoothed = self.beta * prev + (1.0 - self.beta) * bbox
        self._state[track_id] = smoothed
        return smoothed.copy()

    def remove(self, track_id: int) -> None:
        """track 回收時移除其平滑狀態。"""
        self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()


def upper_body_box(bbox: np.ndarray, aspect_ratio: float = 0.75,
                   upper_body_ratio: float = 0.6) -> np.ndarray:
    """由人物全身框計算上半身 ROI 框(固定長寬比、人物中心錨定)。

    Args:
        bbox: 全身框 (x1, y1, x2, y2)
        aspect_ratio: 寬/高(3:4 = 0.75)
        upper_body_ratio: 取框上緣至此比例高度

    Returns:
        上半身 ROI 框 (x1, y1, x2, y2),可能超出影像邊界(裁切時補邊)。
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h = y2 - y1
    cx = (x1 + x2) / 2  # 以人物中心 x 為錨

    roi_h = h * upper_body_ratio
    roi_w = roi_h * aspect_ratio
    return np.array([cx - roi_w / 2, y1, cx + roi_w / 2, y1 + roi_h],
                    dtype=np.float32)


def crop_roi(frame: np.ndarray, roi: np.ndarray,
             out_size: int = 224) -> np.ndarray:
    """依 ROI 框裁切影格並 resize 到 out_size×out_size。

    超出影像邊界的部分以黑邊補齊,維持長寬比例不變形失準。
    """
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_size, out_size, 3), dtype=frame.dtype)

    # 邊界內的實際裁切區
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(W, x2), min(H, y2)

    patch = np.zeros((y2 - y1, x2 - x1, 3), dtype=frame.dtype)
    if ix2 > ix1 and iy2 > iy1:
        patch[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = frame[iy1:iy2, ix1:ix2]
    return cv2.resize(patch, (out_size, out_size),
                      interpolation=cv2.INTER_LINEAR)


def crop_upper_body(frame: np.ndarray, bbox: np.ndarray,
                    aspect_ratio: float = 0.75, upper_body_ratio: float = 0.6,
                    out_size: int = 224) -> np.ndarray:
    """一步完成:全身框 → 上半身 ROI → 裁切 resize。"""
    roi = upper_body_box(bbox, aspect_ratio, upper_body_ratio)
    return crop_roi(frame, roi, out_size)
