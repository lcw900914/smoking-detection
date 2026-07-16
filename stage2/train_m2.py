"""Phase 0/1 訓練:PoseTCN 分類(k-fold 交叉驗證,小資料配方)。

Phase 0(尚無抽菸正樣本):以誤報類別互分(drinking/phone/desk_work/
other)驗證骨架特徵有辨別力;Phase 1 標籤含 smoking 後直接重跑,
自動變成五分類過濾器。

用法:python -m stage2.train_m2 [--folds 5] [--epochs 60]
"""
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stage2.dataset import PoseClipDataset
from stage2.model import PoseTCN, CLASSES
from utils import set_seed, resolve_device


def run_fold(train_items, val_items, present, args, device):
    """訓練一個 fold,回傳 (val 預測, val 真值)。"""
    train_ds = PoseClipDataset(items=train_items, window=args.window,
                               train=True)
    val_ds = PoseClipDataset(items=val_items, window=args.window,
                             train=False)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = PoseTCN(num_classes=len(CLASSES)).to(device)
    # 類別權重:反比頻率(缺席類權重 0)
    counts = Counter(it["label"] for it in train_items)
    w = torch.tensor([1.0 / counts[c] if counts.get(c) else 0.0
                      for c in range(len(CLASSES))],
                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            logits, _ = model(x.to(device))
            loss = criterion(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in val_loader:
            logits, _ = model(x.to(device))
            preds += logits.argmax(1).cpu().tolist()
            gts += y.tolist()
    return preds, gts, model


def main():
    parser = argparse.ArgumentParser(description="M2 PoseTCN k-fold 訓練")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--save", default="checkpoints/stage2_m2.pt")
    args = parser.parse_args()
    set_seed(42)
    device = resolve_device("auto")

    full = PoseClipDataset(window=args.window, train=False)
    items = full.items
    labels = [it["label"] for it in items]
    present = sorted(set(labels))
    print(f"樣本 {len(items)} 段,類別分佈:",
          {CLASSES[c]: labels.count(c) for c in present})

    # 分層 k-fold
    rng = np.random.RandomState(42)
    folds = [[] for _ in range(args.folds)]
    for c in present:
        idx = [i for i, l in enumerate(labels) if l == c]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % args.folds].append(i)

    all_pred, all_gt = [], []
    last_model = None
    for k in range(args.folds):
        val_i = set(folds[k])
        tr = [items[i] for i in range(len(items)) if i not in val_i]
        va = [items[i] for i in val_i]
        if not va or not tr:
            continue
        preds, gts, last_model = run_fold(tr, va, present, args, device)
        all_pred += preds
        all_gt += gts
        acc = np.mean(np.array(preds) == np.array(gts))
        print(f"fold {k}: val acc {acc:.3f}({len(va)} 段)")

    from sklearn.metrics import confusion_matrix, classification_report
    names = [CLASSES[c] for c in present]
    print("\n=== k-fold 匯總 ===")
    print(classification_report(
        all_gt, all_pred, labels=present, target_names=names,
        zero_division=0))
    print("混淆矩陣(列=真值):")
    print(confusion_matrix(all_gt, all_pred, labels=present))

    if last_model is not None:
        import os
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"model": last_model.state_dict(),
                    "classes": CLASSES}, args.save)
        print(f"最後一折模型已存 {args.save}(正式版請全量重訓)")


if __name__ == "__main__":
    main()
