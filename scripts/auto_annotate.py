"""HMDB51 等無框資料集的自動標註:YOLO 人物偵測 + ByteTrack → 標註 json。

對每段影片:
1. 依 target_fps 取樣幀(HMDB51 約 30fps → 每 3 幀取 1)
2. YOLO 偵測人物 + ByteTrack 追蹤
3. 取主要 track(出現次數最多、面積最大者);取樣幀缺偵測時沿用前一框
4. 全片無人被偵測到時退回全幀框(close-up 片段常見)
5. 輸出 data/preprocess.py 相容的標註 json(stage 一律 "none",
   僅 clip 級標籤;階段標籤需之後人工或規則補標)

同時依 --val-ratio 做 train/val 分割(依 clip 名 hash,可重現)。

用法:
    python scripts/auto_annotate.py --videos-root D:/datasets/hmdb51/videos \
        --out D:/datasets/hmdb51/annotations \
        --class-map smoke=smoking drink=drinking eat=eating chew=eating talk=background
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.detector import PersonDetector  # noqa: E402
from tracking.tracker import PersonTracker    # noqa: E402


def sanitize(name: str) -> str:
    """clip_id 檔案系統安全化。"""
    return re.sub(r"[^0-9A-Za-z_\-]", "_", name)


def annotate_video(video_path: Path, detector: PersonDetector,
                   sample_step: int, min_conf_frames: int = 3):
    """對單一影片自動標註,回傳 frames 標註列表(可能為空)。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "無法開啟"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = PersonTracker(frame_rate=30)
    per_track = {}   # track_id → list of (idx, bbox_xyxy)
    sampled = []     # 所有取樣幀 idx
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_step == 0:
            sampled.append(idx)
            dets = detector.detect(frame)
            for tid, bbox in tracker.update(dets):
                per_track.setdefault(tid, []).append((idx, bbox))
        idx += 1
    cap.release()
    if idx == 0:
        return None, "空影片"

    # 主要 track:出現幀數 × 平均面積 作排序依據
    def track_score(entries):
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for _, b in entries]
        return len(entries) * float(np.mean(areas))

    frames = []
    if per_track:
        main_tid, entries = max(per_track.items(),
                                key=lambda kv: track_score(kv[1]))
        if len(entries) >= min_conf_frames:
            by_idx = dict(entries)
            last = entries[0][1]
            for i in sampled:
                bbox = by_idx.get(i)
                if bbox is None:
                    bbox = last  # 缺偵測:沿用前一框
                last = bbox
                x1, y1, x2, y2 = [float(v) for v in bbox]
                frames.append({"idx": i, "stage": "none",
                               "bbox": [x1, y1, x2 - x1, y2 - y1],
                               "track_id": int(main_tid)})
            return frames, f"track {main_tid}({len(entries)} 偵測)"

    # 退回全幀框(close-up 或偵測失敗)
    for i in sampled:
        frames.append({"idx": i, "stage": "none",
                       "bbox": [0.0, 0.0, float(W), float(H)],
                       "track_id": 0})
    return frames, "全幀框 fallback"


def split_of(clip_id: str, val_ratio: float) -> str:
    """依 clip 名 hash 決定 train/val(可重現的隨機分割)。"""
    h = int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < val_ratio * 1000 else "train"


def main():
    parser = argparse.ArgumentParser(description="YOLO+ByteTrack 自動標註")
    parser.add_argument("--videos-root", required=True,
                        help="影片根目錄(類別子資料夾)")
    parser.add_argument("--out", required=True, help="標註輸出根目錄")
    parser.add_argument("--class-map", nargs="+", required=True,
                        help="來源類別=專案標籤,如 smoke=smoking")
    parser.add_argument("--sample-step", type=int, default=3,
                        help="每 N 幀取樣(30fps→10fps 用 3)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    class_map = dict(kv.split("=") for kv in args.class_map)
    root = Path(args.videos_root)
    out = Path(args.out)
    detector = PersonDetector(conf=args.conf)

    stats = {"total": 0, "fallback": 0}
    for src_class, label in class_map.items():
        class_dir = root / src_class
        if not class_dir.is_dir():
            print(f"[auto_annotate] 略過:{class_dir} 不存在")
            continue
        videos = sorted([p for p in class_dir.iterdir()
                         if p.suffix.lower() in (".avi", ".mp4", ".mov", ".mkv")])
        print(f"[auto_annotate] {src_class} → {label}:{len(videos)} 段")
        for vp in videos:
            clip_id = sanitize(f"{src_class}_{vp.stem}")
            frames, info = annotate_video(vp, detector, args.sample_step)
            if frames is None:
                print(f"  [跳過] {vp.name}: {info}")
                continue
            stats["total"] += 1
            if "fallback" in info:
                stats["fallback"] += 1

            split = split_of(clip_id, args.val_ratio)
            ann_dir = out / split
            ann_dir.mkdir(parents=True, exist_ok=True)
            ann = {"clip_id": clip_id,
                   "video": str(vp.relative_to(root)).replace("\\", "/"),
                   "label": label, "frames": frames}
            with open(ann_dir / f"{clip_id}.json", "w",
                      encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False)

    print(f"\n[auto_annotate] 完成:{stats['total']} 段"
          f"(全幀 fallback {stats['fallback']} 段),輸出:{out}")


if __name__ == "__main__":
    main()
