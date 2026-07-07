"""影片 → ROI jpg 序列離線解碼。

輸入:影片資料夾 + 標註 json(schema 見下),輸出每 clip 一個資料夾的
jpg 序列(已做上半身 ROI 裁切)+ label.json,供 dataset 直接讀圖,
避免訓練時 GPU 因即時解碼而閒置。

標註 json schema(每 clip 一個 .json,或一個檔案內含 clip 物件列表):
{
  "clip_id": "...",
  "video": "相對於 --videos 的影片檔名(省略時用 clip_id 推測)",
  "label": "smoking | drinking | eating | phone | wiping | background",
  "frames": [{"idx": 0, "stage": "S1|S2|S3|S4|none",
              "bbox": [x, y, w, h], "track_id": 1}]
}

用法:
    python -m data.preprocess --videos raw/videos --annotations raw/labels \
        --out datasets/processed/train
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from tracking.roi import upper_body_box, crop_roi
from data import STAGE_MAP
from utils import imwrite


def _find_video(videos_dir: Path, ann: dict) -> Path:
    """依標註推測影片檔路徑。"""
    if "video" in ann:
        return videos_dir / ann["video"]
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        cand = videos_dir / f"{ann['clip_id']}{ext}"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"找不到 clip {ann['clip_id']} 的影片(請在標註加 video 欄位)")


def process_clip(ann: dict, videos_dir: Path, out_root: Path,
                 out_size: int = 224, aspect_ratio: float = 0.75,
                 upper_body_ratio: float = 0.6, jpg_quality: int = 95) -> None:
    """處理單一 clip:逐幀依 bbox 裁上半身 ROI 存 jpg,並輸出 label.json。"""
    clip_dir = out_root / ann["clip_id"]
    clip_dir.mkdir(parents=True, exist_ok=True)

    video_path = _find_video(videos_dir, ann)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片:{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 依 idx 排序的幀標註
    frames = sorted(ann["frames"], key=lambda f: f["idx"])
    wanted = {f["idx"]: f for f in frames}
    max_idx = frames[-1]["idx"]

    out_frames: List[dict] = []
    idx = 0
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            fr = wanted[idx]
            x, y, w, h = fr["bbox"]  # 標註為 xywh
            bbox_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            roi_box = upper_body_box(bbox_xyxy, aspect_ratio, upper_body_ratio)
            roi = crop_roi(frame, roi_box, out_size)

            fname = f"img_{len(out_frames):06d}.jpg"
            if not imwrite(clip_dir / fname, roi, jpg_quality=jpg_quality):
                raise RuntimeError(f"寫圖失敗:{clip_dir / fname}")
            out_frames.append({
                "file": fname,
                "src_idx": idx,
                "stage": fr.get("stage", "none"),
                "stage_id": STAGE_MAP.get(fr.get("stage", "none"), 3),
                "track_id": fr.get("track_id", 0),
            })
        idx += 1
    cap.release()

    label = {
        "clip_id": ann["clip_id"],
        "label": ann["label"],
        "fps": fps,
        "num_frames": len(out_frames),
        "frames": out_frames,
    }
    with open(clip_dir / "label.json", "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)
    print(f"[preprocess] {ann['clip_id']}: {len(out_frames)} 幀 → {clip_dir}")


def load_annotations(ann_path: Path) -> List[dict]:
    """讀取標註:可為單一 json 檔(物件或列表)或資料夾內多個 json。"""
    anns: List[dict] = []
    if ann_path.is_dir():
        for p in sorted(ann_path.glob("*.json")):
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            anns.extend(obj if isinstance(obj, list) else [obj])
    else:
        with open(ann_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        anns = obj if isinstance(obj, list) else [obj]
    return anns


def main():
    parser = argparse.ArgumentParser(description="影片 → ROI jpg 序列離線解碼")
    parser.add_argument("--videos", required=True, help="影片資料夾")
    parser.add_argument("--annotations", required=True,
                        help="標註 json 檔或資料夾")
    parser.add_argument("--out", required=True, help="輸出根目錄(含 split)")
    parser.add_argument("--out-size", type=int, default=224)
    parser.add_argument("--aspect-ratio", type=float, default=0.75)
    parser.add_argument("--upper-body-ratio", type=float, default=0.6)
    args = parser.parse_args()

    anns = load_annotations(Path(args.annotations))
    out_root = Path(args.out)
    print(f"[preprocess] 共 {len(anns)} 個 clip")
    for ann in anns:
        process_clip(ann, Path(args.videos), out_root,
                     out_size=args.out_size,
                     aspect_ratio=args.aspect_ratio,
                     upper_body_ratio=args.upper_body_ratio)


if __name__ == "__main__":
    main()
