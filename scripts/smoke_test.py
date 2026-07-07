"""端到端煙霧測試:用合成資料驗證整條訓練工具鏈可跑通。

流程:
    合成影片 + 標註 json
    → data.preprocess(ROI jpg 序列)
    → training.extract_features(fp16 .npy)
    → training.train_head(2 epochs,小 batch)
    → eval.clip_eval

全程 CPU 可跑(backbone 不載預訓練權重,避免下載);
輸出全部放在 ./smoke_run/,結束時印出 PASS / FAIL。

用法(專案根目錄):
    python scripts/smoke_test.py [--keep](保留 smoke_run 供檢查)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

RUN = ROOT / "smoke_run"
N_FRAMES = 100
W_VID, H_VID = 320, 240


def make_synthetic_clip(videos_dir: Path, ann_dir: Path, clip_id: str,
                        label: str, seed: int) -> None:
    """產生一段合成影片(移動矩形模擬人物)與對應標註。"""
    rng = np.random.RandomState(seed)
    videos_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    path = videos_dir / f"{clip_id}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (W_VID, H_VID))
    # 人物框:輕微漂移
    x, y, w, h = 120.0, 40.0, 80.0, 160.0
    frames = []
    for i in range(N_FRAMES):
        x += rng.uniform(-1.5, 1.5)
        y += rng.uniform(-0.8, 0.8)
        frame = np.full((H_VID, W_VID, 3), 40, dtype=np.uint8)
        color = (0, 180, 0) if label == "smoking" else (180, 0, 0)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)),
                      color, -1)
        # smoking 用亮點模擬手部週期動作(讓兩類影像有統計差異)
        if label == "smoking" and (i % 25) < 12:
            cv2.circle(frame, (int(x + w / 2), int(y + 30)), 8,
                       (255, 255, 255), -1)
        writer.write(frame)

        # 階段標籤:smoking 走 S1(5)-S2(12)-S3(5)-none(3) 週期
        if label == "smoking":
            phase = i % 25
            stage = ("S1" if phase < 5 else
                     "S2" if phase < 17 else
                     "S3" if phase < 22 else "none")
        else:
            stage = "S2" if (i % 40) < 3 else "none"  # hard negative:短暫 S2
        frames.append({"idx": i, "stage": stage,
                       "bbox": [x, y, w, h], "track_id": 1})
    writer.release()

    with open(ann_dir / f"{clip_id}.json", "w", encoding="utf-8") as f:
        json.dump({"clip_id": clip_id, "video": path.name,
                   "label": label, "frames": frames}, f)


def run_module(mod: str, *args: str) -> None:
    """以子行程執行 `python -m mod args...`,失敗即中止。"""
    cmd = [sys.executable, "-m", mod, *args]
    print(f"\n>>> {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(f"[煙霧測試 FAIL] {mod} 退出碼 {r.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true",
                        help="結束後保留 smoke_run/")
    args = parser.parse_args()

    if RUN.exists():
        shutil.rmtree(RUN)

    # ---------- 1. 合成資料 ----------
    print("=== 1/5 產生合成影片與標註 ===")
    specs = [("smoke", "smoking"), ("drink", "drinking"),
             ("bg", "background")]
    for split, n in (("train", 2), ("val", 1)):
        for stem, label in specs:
            for k in range(n):
                make_synthetic_clip(
                    RUN / "raw" / split / "videos",
                    RUN / "raw" / split / "labels",
                    f"{stem}_{split}_{k}", label,
                    seed=hash((stem, split, k)) % 10000)

    # ---------- 2. 前處理 ----------
    print("=== 2/5 前處理(影片 → ROI jpg)===")
    for split in ("train", "val"):
        run_module("data.preprocess",
                   "--videos", str(RUN / "raw" / split / "videos"),
                   "--annotations", str(RUN / "raw" / split / "labels"),
                   "--out", str(RUN / "processed" / split))

    # ---------- 3. 煙霧測試用設定(不載預訓練、小 batch)----------
    model_cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text(
        encoding="utf-8"))
    model_cfg["backbone"]["pretrained"] = False  # 避免下載權重
    (RUN / "model_smoke.yaml").write_text(
        yaml.safe_dump(model_cfg, allow_unicode=True), encoding="utf-8")

    train_cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text(
        encoding="utf-8"))
    train_cfg["paths"].update({
        "ckpt_dir": str(RUN / "checkpoints"),
        "log_dir": str(RUN / "runs"),
    })
    train_cfg["head"].update({"epochs": 2, "batch_size": 8,
                              "num_workers": 0})
    (RUN / "train_smoke.yaml").write_text(
        yaml.safe_dump(train_cfg, allow_unicode=True), encoding="utf-8")

    # ---------- 4. 特徵抽取 + 階段一訓練 ----------
    print("=== 3/5 離線特徵抽取 ===")
    for split in ("train", "val"):
        run_module("training.extract_features",
                   "--data", str(RUN / "processed" / split),
                   "--out", str(RUN / "features" / split),
                   "--model-config", str(RUN / "model_smoke.yaml"),
                   "--batch-size", "32")

    print("=== 4/5 階段一訓練(2 epochs)===")
    run_module("training.train_head",
               "--train-features", str(RUN / "features" / "train"),
               "--val-features", str(RUN / "features" / "val"),
               "--model-config", str(RUN / "model_smoke.yaml"),
               "--train-config", str(RUN / "train_smoke.yaml"),
               "--tag", "smoke")

    ckpt = RUN / "checkpoints" / "smoke_last.pt"
    assert ckpt.exists(), "訓練未產生 checkpoint"

    # ---------- 5. clip 級評估 ----------
    print("=== 5/5 clip 級評估 ===")
    run_module("eval.clip_eval",
               "--features", str(RUN / "features" / "val"),
               "--ckpt", str(ckpt),
               "--model-config", str(RUN / "model_smoke.yaml"))

    print("\n[煙霧測試 PASS] 前處理 → 特徵抽取 → 訓練 → 評估 全鏈路可跑")
    if not args.keep:
        shutil.rmtree(RUN, ignore_errors=True)
        print("(smoke_run/ 已清理,--keep 可保留)")


if __name__ == "__main__":
    main()
