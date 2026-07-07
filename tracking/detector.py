"""YOLOv8 人物偵測 wrapper(只取 person 類)。"""
from typing import Optional

import numpy as np


class PersonDetector:
    """Ultralytics YOLOv8 人物偵測器。

    Args:
        model: 權重路徑或模型名(預設 yolov8s.pt,首次使用自動下載)
        conf: 置信度門檻(預設 0.4)
        device: auto / cuda / cpu
    """

    PERSON_CLASS_ID = 0  # COCO person

    def __init__(self, model: str = "yolov8s.pt", conf: float = 0.4,
                 device: str = "auto"):
        from ultralytics import YOLO  # 延遲載入,避免測試環境依賴

        self.model = YOLO(model)
        self.conf = conf
        self.device = None if device == "auto" else device

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """偵測單張 BGR 影格中的人物。

        Returns:
            (N, 5) ndarray,每列 [x1, y1, x2, y2, conf];無人時 shape (0, 5)。
        """
        results = self.model.predict(
            frame, conf=self.conf, classes=[self.PERSON_CLASS_ID],
            device=self.device, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()[:, None]
        return np.concatenate([xyxy, conf], axis=1).astype(np.float32)
