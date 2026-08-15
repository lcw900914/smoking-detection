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

    @staticmethod
    def _parse(r) -> Tuple[np.ndarray, np.ndarray]:
        """一筆 ultralytics 結果 → (boxes, kpts)。"""
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 5), dtype=np.float32),
                    np.zeros((0, 17, 3), dtype=np.float32))
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()[:, None]
        boxes = np.concatenate([xyxy, conf], axis=1).astype(np.float32)
        kpts = r.keypoints.data.cpu().numpy().astype(np.float32)
        return boxes, kpts

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """偵測人物與關鍵點。

        Returns:
            boxes: (N, 5) [x1, y1, x2, y2, conf]
            kpts:  (N, 17, 3) [x, y, conf];與 boxes 同順序
        """
        results = self.model.predict(frame, conf=self.conf,
                                     device=self.device, verbose=False)
        return self._parse(results[0])

    def detect_batch(self, frames) -> list:
        """一次推論多張影格,回傳與輸入等長、同順序的結果串列。

        存在的理由是**固定開銷**:實測每次 predict() 約 28ms,而且把模型
        換成 yolov8n、解析度降到 320 都還是 28ms——成本不在計算,在每次
        呼叫的啟動與同步。批次 4 攤掉之後每幀降到約 14.7ms(2 倍)。
        再大反而更慢:6GB VRAM 到批次 8 就開始塞不下。

        只有離線分析用得上(它一開始就知道要跑哪些幀);即時管線一次只
        拿得到一張,沒有東西可以批。
        """
        if not frames:
            return []
        results = self.model.predict(list(frames), conf=self.conf,
                                     device=self.device, verbose=False)
        return [self._parse(r) for r in results]
