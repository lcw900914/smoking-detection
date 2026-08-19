"""外觀單幀對照組的資料匯出:警報片段 → 「手舉起來」那些幀的人物裁切。

為什麼要有外觀這一組
────────────────────
骨架對照組(frame_baseline.py)與主線時序模型看的都是**座標**,而座標
裡沒有菸。手停在嘴邊 1.5 秒,抽菸與喝水的骨架幾乎一模一樣——這是整個
專案誤報的根源(docs 的 57 段誤報分析)。外觀模型是唯一有機會直接看到
「手上那根白色細長物」的路徑,所以它必須在比較表上有一格,否則「時序
到底有沒有用」這個問題會被回答成「骨架到底夠不夠用」。

關於 YOLO26(講清楚,免得找錯東西)
──────────────────────────────────
Ultralytics 的 YOLO26 提供的任務是 detect / segment / pose / classify /
obb / sem,**沒有內建的「人體動作辨識」模型**。單幀判動作在這個生態系
裡對應的是 `yolo26-cls`:對裁好的人物影像做影像分類。也就是說「YOLO26
判動作」= 我們自己定義類別、自己準備裁切、拿 yolo26-cls 訓練。這支腳本
做的就是「自己準備裁切」那一步。骨架仍然來自 pose 模型,它負責決定
**哪些幀值得裁**,分類則交給外觀模型。

裁哪一塊
────────
預設裁**上半身**(鼻/雙肩/雙腕的外接框加邊),不是整個人物框。理由是
解析度:模型輸入是 224×224,整個人物框有一半的像素花在腿和桌子上,
而判別訊息全部集中在手與臉之間那一塊。要比對整框的效果可以 --region
person。

三個必須先知道的污染來源
────────────────────────
1. **疊加層烙在像素上 —— 這是目前的硬阻礙。** 2026-08-18 實測
   alarms/clips 底下每一段(含最新的 08-16)都把綠色人物框、青色骨架、
   紅色腕-鼻線與紅色警報框畫進畫面才存檔。這不只是雜訊,是**直接的
   標籤洩漏**:紅框與紅線只在規則判定成立時才畫,外觀模型只要學會
   「有沒有紅線」就贏了,完全不必看菸。
   所以這支腳本預設會擋下來(見 `overlay_ratio`),要用 --allow-overlay
   才跑得動,而那樣跑出來的數字沒有意義。
   治本:GUI 的錄影疊加開關關掉(關 = 錄乾淨影像)之後重錄一批。
2. **場景捷徑比骨架更嚴重。** 骨架至少做過身體尺度正規化;像素沒有。
   6 段抽菸正樣本全來自同一場錄影同一個機位,背景本身就足以分類。
   train_frame.py 的捷徑探針在骨架上量到 AUC 0.94,外觀只會更高。
   **在補到跨場景正樣本之前,外觀這一組的數字不能當成辨識能力。**
3. **同段的幀高度重複。** 切 train/val 一定要依段切,不能依幀切。
   manifest 記了每張圖屬於哪一段,train_frame_cls.py 據此分 fold。

用法:
    python -m stage2.export_crops                       # 上半身裁切
    python -m stage2.export_crops --region person       # 整個人物框
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from stage2.frame_baseline import candidate_frames
from stage2.kinematics import kinematic_features
from stage2.hier_dataset import load_pose_items
from stage2.taxonomy import DEEP_CLASSES, deep_index
from utils import imwrite

# 上半身區域用到的關鍵點:鼻、雙眼、雙耳、雙肩、雙肘、雙腕
UPPER_JOINTS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
PAD_RATIO = 0.25          # 外接框各邊往外擴這個比例
MIN_SIDE_PX = 48          # 比這還小的裁切放大也沒有資訊,直接丟掉


# 疊加層偵測門檻:飽和純色像素佔比。自然的辦公室畫面幾乎沒有這種
# 像素(布料與螢幕都不會同時「最亮通道 ≥200、最暗通道 ≤90」);
# 實測有疊加層的片段落在 0.5% ~ 1%,乾淨畫面應該低一個數量級。
OVERLAY_RATIO_MAX = 0.002


def overlay_ratio(frame: np.ndarray) -> float:
    """畫面裡「飽和純色」像素的比例 —— 疊加層的指紋。"""
    mx = frame.max(axis=2).astype(np.int16)
    mn = frame.min(axis=2).astype(np.int16)
    return float(((mx >= 200) & (mn <= 90) & (mx - mn >= 140)).mean())


def check_overlay(items, pose_dir: str, n_probe: int = 5) -> float:
    """抽驗幾段的疊加層比例,回傳中位數。"""
    import cv2

    vals = []
    for it in items[:n_probe]:
        cap = cv2.VideoCapture(str(it["clip"]))
        for i in range(0, 60, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                vals.append(overlay_ratio(fr))
        cap.release()
    return float(np.median(vals)) if vals else 0.0


def upper_body_box(kpts_t: np.ndarray, conf_thresh: float = 0.3):
    """單幀關鍵點 → 上半身外接框 (x1, y1, x2, y2);點太少回傳 None。"""
    pts = kpts_t[list(UPPER_JOINTS)]
    vis = pts[:, 2] >= conf_thresh
    if vis.sum() < 3:
        return None
    xy = pts[vis, :2]
    x1, y1 = xy.min(0)
    x2, y2 = xy.max(0)
    # 做成正方形再加邊:非正方形裁切送進 224×224 會被拉伸變形,
    # 而「手臂伸長多少」正是這個任務的判別量之一
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2 * (1 + PAD_RATIO)
    return np.array([cx - half, cy - half, cx + half, cy + half])


def crop(frame: np.ndarray, box) -> "np.ndarray | None":
    """框可以超出畫面;超出的部分補黑,而不是把框推回畫面內。

    推回去會讓人物在裁切裡偏心,同一個動作在畫面邊緣與中央長得不一樣,
    模型得多學一份。補黑則保持人物永遠在中心。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    out = np.zeros((y2 - y1, x2 - x1, 3), frame.dtype)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    if sx2 <= sx1 or sy2 <= sy1:
        return None
    out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return out


def export_clip(item: dict, region: str, out_dir: Path, stride: int):
    """→ [manifest 條目];來源影片不存在或讀不到時回傳空清單。"""
    import cv2

    src = Path(item["clip"])
    if not src.exists():
        print(f"[跳過] {item['stem']}:找不到來源影片 {src}")
        return []
    label = item["label"]
    cls = DEEP_CLASSES[deep_index(label)]
    kin = kinematic_features(item["kpts"], item["fps"])
    picks = sorted({t for t, _ in candidate_frames(kin)})[::stride]
    if not picks:
        return []

    want = set(picks)
    cap = cv2.VideoCapture(str(src))
    rows, t = [], 0
    dst_dir = out_dir / "pool" / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if t in want:
            box = (upper_body_box(item["kpts"][t]) if region == "upper"
                   else item["bbox"][t])
            patch = crop(frame, box) if box is not None else None
            if patch is not None:
                name = f"{item['stem']}_{t:04d}.jpg"
                if imwrite(dst_dir / name, patch):
                    rows.append({"path": f"pool/{cls}/{name}",
                                 "clip": item["stem"], "frame": t,
                                 "label": label, "class": cls})
        t += 1
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser(description="單幀外觀對照組的裁切匯出")
    ap.add_argument("--pose-dir", default="annotations/pose")
    ap.add_argument("--labels", default="annotations/clip_labels.json")
    ap.add_argument("--out", default="datasets/frame_cls")
    ap.add_argument("--region", default="upper", choices=("upper", "person"),
                    help="upper = 鼻/肩/腕外接框(預設);person = 整個人物框")
    ap.add_argument("--stride", type=int, default=1,
                    help="候選幀取樣間隔;同段的相鄰幀幾乎一樣,>1 可省空間")
    ap.add_argument("--allow-overlay", action="store_true",
                    help="明知畫面有疊加層仍要匯出(結果不能當成辨識能力)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [it for it in load_pose_items(args.pose_dir, args.labels)
             if it.get("label") and deep_index(it["label"]) is not None]

    # bbox 在 load_pose_items 裡沒帶出來,person 模式要自己補讀
    if args.region == "person":
        for it in items:
            d = np.load(Path(args.pose_dir) / f"{it['stem']}.npz",
                        allow_pickle=True)
            it["bbox"] = d["bbox"]

    # ---- 疊加層守門 ----
    ratio = check_overlay(items, args.pose_dir)
    print(f"疊加層檢查:飽和純色像素佔比中位數 {ratio:.4%}"
          f"(門檻 {OVERLAY_RATIO_MAX:.2%})")
    if ratio > OVERLAY_RATIO_MAX and not args.allow_overlay:
        raise SystemExit("\n".join([
            "",
            "這批片段的畫面上畫了人物框 / 骨架 / 腕-鼻線。",
            "紅框與紅線只在規則判定成立時才畫 —— 外觀模型會直接學它,",
            "學到的是「上一版規則怎麼判」,不是「這個人在不在抽菸」。",
            "",
            "要做外觀對照組,先把 GUI 的錄影疊加開關關掉"
            "(關 = 錄乾淨影像)重錄一批,再跑這支。",
            "真的只是想看看流程跑不跑得動,加 --allow-overlay。"]))

    rows = []
    for it in items:
        rows += export_clip(it, args.region, out, max(1, args.stride))
        print(f"[{it['stem']}] 累計 {len(rows)} 張")

    manifest = out / "manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"region": args.region, "stride": args.stride,
                   "classes": DEEP_CLASSES, "items": rows}, f,
                  ensure_ascii=False, indent=1)

    dist = Counter(r["class"] for r in rows)
    clips = len({r["clip"] for r in rows})
    print(f"\n完成:{len(rows)} 張裁切、來自 {clips} 段 → {out}")
    print("類別分佈:" + "  ".join(f"{k} {v}" for k, v in dist.items()))
    print(f"清單:{manifest}")
    print("\n下一步:python -m stage2.train_frame_cls")
    print("⚠ 提醒:6 段抽菸正樣本全來自同一場錄影,外觀模型會直接學背景。"
          "\n  這一組的數字在補到跨場景正樣本之前只能當上界看。")


if __name__ == "__main__":
    main()
