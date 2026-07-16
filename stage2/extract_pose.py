"""Phase 0 離線回抽:對已錄警報片段重跑姿態,抽出「警報對象」的節點序列。

難點:片段裡有多人,要鎖定被警報的那個人——
1. 以顏色遮罩找出各幀的紅色警報框(疊加層烙在像素上,反成定位依據)
2. 警報活躍幀:取與紅框 IoU 最大的姿態偵測為對象
3. 其餘幀:以 IoU 鏈式關聯前後傳播(同一人的框逐幀連續)

輸出(每段):annotations/pose/{clip_stem}.npz
    kpts  (T, 17, 3)  對象節點序列(缺幀以 conf=0 填)
    bbox  (T, 4)      對象框
    valid (T,)        該幀是否成功關聯
    fps, clip 相對路徑

用法:python -m stage2.extract_pose [--dir alarms/clips]
"""
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np

from tracking.pose_detector import PoseDetector


def find_red_box(frame: np.ndarray):
    """紅色警報「矩形框線」的外接框;不足回傳 None。

    紅色遮罩會同時吃到標籤文字與腕-鼻紅線——取「外接面積最大」的
    連通元件,矩形框線的延展遠大於文字塊與短線,天然勝出。
    (遠處小人物的文字若混入外接框,會撐歪框導致 IoU 匹配失敗)
    """
    b = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    r = frame[:, :, 2].astype(int)
    mask = ((r > 180) & (g < 80) & (b < 80)).astype(np.uint8)
    if mask.sum() < 150:
        return None
    # 輕度膨脹把框線像素連起來
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    best_area = 0
    for i in range(1, n):
        x, y, w, h, px = stats[i]
        if px < 100:
            continue
        if w * h > best_area:          # 外接面積(延展),非像素數
            best_area = w * h
            best = np.array([x, y, x + w, y + h], float)
    return best


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def extract_clip(path: str, detector: PoseDetector):
    """回傳 (kpts (T,17,3), bbox (T,4), valid (T,), fps)。"""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames_dets = []          # 每幀:(boxes (N,5), kpts (N,17,3))
    red_boxes = []            # 每幀:紅框或 None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, kpts = detector.detect(frame)
        frames_dets.append((boxes, kpts))
        red_boxes.append(find_red_box(frame))
    cap.release()
    T = len(frames_dets)
    if T == 0:
        return None

    # 1) 錨定:紅框幀中,挑「與紅框 IoU 最大」且最可信的一幀當起點
    anchor_t, anchor_i, best = -1, -1, 0.2
    for t, ((boxes, _), red) in enumerate(zip(frames_dets, red_boxes)):
        if red is None or len(boxes) == 0:
            continue
        for i, bb in enumerate(boxes[:, :4]):
            v = iou(bb, red)
            if v > best:
                anchor_t, anchor_i, best = t, i, v
    if anchor_t < 0:
        return None  # 整段找不到可關聯的紅框對象

    # 2) 由錨點向前後做 IoU 鏈式關聯
    sel = [-1] * T
    sel[anchor_t] = anchor_i
    for direction in (1, -1):
        prev_box = frames_dets[anchor_t][0][anchor_i, :4]
        t = anchor_t + direction
        miss = 0
        while 0 <= t < T and miss <= 15:   # 容忍 1.5 秒偵測中斷
            boxes = frames_dets[t][0]
            j_best, v_best = -1, 0.3
            for j, bb in enumerate(boxes[:, :4]):
                v = iou(bb, prev_box)
                if v > v_best:
                    j_best, v_best = j, v
            if j_best >= 0:
                sel[t] = j_best
                prev_box = boxes[j_best, :4]
                miss = 0
            else:
                miss += 1
            t += direction

    kpts_out = np.zeros((T, 17, 3), np.float32)
    bbox_out = np.zeros((T, 4), np.float32)
    valid = np.zeros(T, bool)
    for t, j in enumerate(sel):
        if j >= 0:
            kpts_out[t] = frames_dets[t][1][j]
            bbox_out[t] = frames_dets[t][0][j, :4]
            valid[t] = True
    return kpts_out, bbox_out, valid, fps


def main():
    parser = argparse.ArgumentParser(description="警報片段節點序列回抽")
    parser.add_argument("--dir", default="alarms/clips")
    parser.add_argument("--out", default="annotations/pose")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 離線回抽用低門檻:誤偵測由紅框匹配自然濾除,
    # 遠處小人物才抓得到
    detector = PoseDetector("yolov8s-pose.pt", conf=0.2)

    clips = sorted(glob.glob(os.path.join(args.dir, "*.mp4")),
                   key=os.path.getmtime)
    n_ok = 0
    for path in clips:
        stem = Path(path).stem
        dst = out / f"{stem}.npz"
        if dst.exists():
            n_ok += 1
            continue
        result = extract_clip(path, detector)
        if result is None:
            print(f"[跳過] {stem}:找不到警報對象")
            continue
        kpts, bbox, valid, fps = result
        np.savez_compressed(
            dst, kpts=kpts, bbox=bbox, valid=valid, fps=fps,
            clip=str(Path(path).as_posix()))
        n_ok += 1
        print(f"[OK] {stem}:{valid.sum()}/{len(valid)} 幀關聯成功")
    print(f"完成:{n_ok}/{len(clips)} 段,輸出 {out}")


if __name__ == "__main__":
    main()
