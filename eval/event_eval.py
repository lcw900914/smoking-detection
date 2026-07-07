"""事件級評估:在長測試影片上跑完整 pipeline,計算——

- 事件級 precision / recall(預測警報區間與真值區間
  IoU-in-time ≥ 0.3 算命中)
- 每小時誤報數(FP/h)
- 警報延遲中位數(真值行為開始 → 警報觸發的秒數)

真值 json 格式:
    {"events": [{"start": 12.0, "end": 45.0, "track_id": 1}, ...]}
    (track_id 可省略;省略時僅以時間區間匹配)

用法:
    python -m eval.event_eval --video test.mp4 --gt test_events.json \
        --ckpt checkpoints/e2e_best.pt
"""
import argparse
import json
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import load_config


def temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """兩時間區間的 IoU。"""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match_events(pred: List[dict], gt: List[dict],
                 iou_thresh: float = 0.3) -> dict:
    """貪婪匹配預測與真值事件,計算 P/R 與警報延遲。

    pred/gt 元素:{"start", "end", 可選 "track_id"}
    """
    matched_gt = set()
    delays = []
    tp = 0
    for p in sorted(pred, key=lambda e: e["start"]):
        best_iou, best_j = 0.0, None
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            iou = temporal_iou((p["start"], p["end"]),
                               (g["start"], g["end"]))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j is not None and best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp += 1
            delays.append(p["start"] - gt[best_j]["start"])

    fp = len(pred) - tp
    fn = len(gt) - tp
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "median_delay_sec": float(np.median(delays)) if delays else None,
    }


def collect_alarm_events(pipeline, video_path: str) -> Tuple[List[dict], float]:
    """跑完整 pipeline,收集每個 track 的警報區間(觸發→解除)。

    回傳 (事件列表, 影片總時長秒數)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片:{video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_every = max(1, round(
        src_fps / pipeline.cfg["sampling"]["target_fps"]))

    events: List[dict] = []
    active: dict = {}  # track_id → start_time
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / src_fps
        if frame_idx % sample_every == 0:
            results = pipeline.step(frame, t)
            for tid, r in results.items():
                if r["alarm"] and tid not in active:
                    active[tid] = t
                elif not r["alarm"] and tid in active:
                    events.append({"start": active.pop(tid), "end": t,
                                   "track_id": tid})
        frame_idx += 1
    duration = frame_idx / src_fps
    # 影片結束仍觸發中的警報
    for tid, start in active.items():
        events.append({"start": start, "end": duration, "track_id": tid})
    cap.release()
    return events, duration


def main():
    parser = argparse.ArgumentParser(description="事件級評估")
    parser.add_argument("--video", required=True, help="長測試影片")
    parser.add_argument("--gt", required=True, help="真值事件 json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--infer-config", default="configs/inference.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    args = parser.parse_args()

    from inference.pipeline import SmokingDetectionPipeline
    pipeline = SmokingDetectionPipeline(
        load_config(args.infer_config), load_config(args.model_config),
        ckpt_path=args.ckpt)

    with open(args.gt, "r", encoding="utf-8") as f:
        gt_events = json.load(f)["events"]

    pred_events, duration = collect_alarm_events(pipeline, args.video)
    metrics = match_events(pred_events, gt_events, args.iou_thresh)
    hours = duration / 3600.0
    metrics["fp_per_hour"] = metrics["fp"] / hours if hours > 0 else None
    metrics["video_duration_sec"] = duration
    metrics["num_pred_events"] = len(pred_events)
    metrics["num_gt_events"] = len(gt_events)

    print("\n===== 事件級評估結果 =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
