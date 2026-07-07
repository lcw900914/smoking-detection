"""Clip 級評估:accuracy、macro-F1、各階段 P/R、混淆矩陣。

混淆矩陣以原始行為類別列出(smoking vs 各 hard negative 分開),
用於分析誤報主要來自哪類混淆動作。

用法:
    python -m eval.clip_eval --features datasets/features/val \
        --ckpt checkpoints/head_best.pt
    (--data 模式則用影像端到端評估)
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.full_model import build_model
from data import STAGE_NAMES, LABEL_SET
from utils import load_config, resolve_device


@torch.no_grad()
def run_eval(model, loader, device, use_features: bool,
             clip_meta: dict) -> dict:
    """收集預測並計算指標。clip_meta: clip_id → 原始行為標籤。"""
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix)
    model.eval()
    stage_true, stage_pred = [], []
    clip_true, clip_pred, clip_labels_raw = [], [], []

    for batch in loader:
        if use_features:
            out = model.forward_features(batch["short_feats"].to(device),
                                         batch["long_feats"].to(device))
        else:
            out = model(batch["short_images"].to(device),
                        batch["long_images"].to(device))
        stage_true += batch["stage_label"].tolist()
        stage_pred += out["stage_logits"].argmax(1).cpu().tolist()
        clip_true += batch["clip_label"].tolist()
        clip_pred += (torch.sigmoid(out["fusion_logit"]).cpu() > 0.5
                      ).float().tolist()
        clip_labels_raw += [clip_meta.get(cid, "background")
                            for cid in batch["clip_id"]]

    # clip 級
    acc = accuracy_score(clip_true, clip_pred)
    macro_f1 = f1_score(clip_true, clip_pred, average="macro")

    # 階段 P/R
    p, r, f, _ = precision_recall_fscore_support(
        stage_true, stage_pred, labels=list(range(len(STAGE_NAMES))),
        zero_division=0)

    # smoking vs 各 hard negative:每個原始類別的預測為 smoking 的比例
    per_label = defaultdict(lambda: [0, 0])  # label → [判為 smoking 數, 總數]
    for raw, pred in zip(clip_labels_raw, clip_pred):
        per_label[raw][0] += int(pred == 1.0)
        per_label[raw][1] += 1

    cm = confusion_matrix(clip_true, clip_pred, labels=[0, 1])

    return {
        "clip_accuracy": acc, "clip_macro_f1": macro_f1,
        "stage_precision": dict(zip(STAGE_NAMES, p.round(4))),
        "stage_recall": dict(zip(STAGE_NAMES, r.round(4))),
        "stage_f1": dict(zip(STAGE_NAMES, f.round(4))),
        "binary_confusion_matrix": cm.tolist(),
        "predicted_smoking_rate_per_label": {
            k: f"{v[0]}/{v[1]}" for k, v in sorted(per_label.items())},
    }


def main():
    parser = argparse.ArgumentParser(description="clip 級評估")
    parser.add_argument("--features", default=None, help="離線特徵目錄")
    parser.add_argument("--data", default=None, help="processed clip 目錄")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    assert (args.features is None) != (args.data is None), \
        "請擇一指定 --features 或 --data"

    model_cfg = load_config(args.model_config)
    device = resolve_device("auto")
    model = build_model(model_cfg).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))

    buf = model_cfg["buffer"]
    kwargs = dict(short_T=buf["short"]["T"], long_T=buf["long"]["T"],
                  long_stride=buf["long"]["stride"])
    if args.features:
        from data.feature_dataset import FeatureClipDataset
        ds = FeatureClipDataset(args.features, train=False, **kwargs)
        clip_meta = {m["clip_id"]: m["label"] for m in ds.metas}
        use_features = True
    else:
        from data.dataset import SmokingClipDataset
        ds = SmokingClipDataset(args.data, augment=None, **kwargs)
        clip_meta = {c["clip_id"]: c["label"] for c in ds.clips}
        use_features = False

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    metrics = run_eval(model, loader, device, use_features, clip_meta)

    print("\n===== Clip 級評估結果 =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
