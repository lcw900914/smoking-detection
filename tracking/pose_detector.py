"""YOLOv8-pose 人物偵測 + 關鍵點抽取(骨架分支用)。

與 PersonDetector 介面相容(回傳框),額外回傳 COCO 17 關鍵點,
使偵測與姿態共用一次前向,不增加額外偵測模型。
"""
from typing import Tuple

import numpy as np


class PoseDetector:
    """YOLOv8-pose:一次前向同時取得人物框與 17 個 COCO 關鍵點。

    關鍵點索引:0 鼻、5/6 肩、7/8 肘、9/10 腕、11/12 髖。
    """

    def __init__(self, model: str = "yolov8s-pose.pt", conf: float = 0.4,
                 device: str = "auto"):
        from ultralytics import YOLO  # 延遲載入

        self.model = YOLO(model)
        self.conf = conf
        self.device = None if device == "auto" else device

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """偵測人物與關鍵點。

        Returns:
            boxes: (N, 5) [x1, y1, x2, y2, conf]
            kpts:  (N, 17, 3) [x, y, conf];與 boxes 同順序
        """
        results = self.model.predict(frame, conf=self.conf,
                                     device=self.device, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 17, 3), dtype=np.float32))
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()[:, None]
        boxes = np.concatenate([xyxy, conf], axis=1).astype(np.float32)
        kpts = r.keypoints.data.cpu().numpy().astype(np.float32)
        return boxes, kpts
