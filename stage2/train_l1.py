"""L1 訓練:骨架拓樸圖 → 逐幀淺層基元。

不需要任何人工標籤——監督訊號由 primitives.py 的規則產生。
所以這支腳本現在就能跑,而且再多錄影片就能再變強,不用等人標。

評估看兩件事:
  1. **對規則的還原率**:在規則有把握的幀上,網路同不同意。這是下限,
     不是目標——完全等於規則代表網路只學會了複製公式。
  2. **在規則棄權的幀上的產出**:規則因為鼻點不可信而棄權的地方,
     網路仍會給答案。這正是要它學的東西,但沒有真值可以量,
     只能看基元時間軸的連續性(--dump 印出來人看)。

用法:
    python -m stage2.train_l1                 # 5-fold 交叉驗證 + 全量重訓
    python -m stage2.train_l1 --dump 3        # 另外印 3 段的基元時間軸
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from stage2.hier_dataset import PrimitiveDataset, load_pose_items
from stage2.hier_model import PrimitiveNet, count_params
from stage2.primitives import IGNORE, NUM_PRIMITIVES, PRIMITIVES, PRIM_NAMES
from utils import resolve_device, set_seed


def class_weights(items, device):
    """反比頻率的類別權重:raise 只佔 1.7%,不加權會被 free 淹掉。"""
    from stage2.hier_dataset import primitive_label_stats
    st = primitive_label_stats(items)
    counts = np.array([max(st.get(c, 0), 1) for c in range(NUM_PRIMITIVES)],
                      dtype=np.float64)
    w = counts.sum() / (NUM_PRIMITIVES * counts)
    return torch.tensor(w, dtype=torch.float32, device=device), st


def run_epoch(model, loader, criterion, device, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss, correct, total = 0.0, 0, 0
    conf_mat = np.zeros((NUM_PRIMITIVES, NUM_PRIMITIVES), np.int64)
    with torch.set_grad_enabled(train):
        for g, kin, lab in loader:
            g, kin, lab = g.to(device), kin.to(device), lab.to(device)
            logits, _ = model(g, kin)               # (B,T,2,P)
            loss = criterion(logits.reshape(-1, NUM_PRIMITIVES),
                             lab.reshape(-1))
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tot_loss += loss.item() * lab.numel()
            m = lab.reshape(-1) != IGNORE
            if m.any():
                p = logits.reshape(-1, NUM_PRIMITIVES).argmax(1)[m]
                t = lab.reshape(-1)[m]
                correct += int((p == t).sum())
                total += int(m.sum())
                for a, b in zip(t.cpu().numpy(), p.cpu().numpy()):
                    conf_mat[a, b] += 1
    return tot_loss / max(len(loader), 1), correct / max(total, 1), conf_mat


def main():
    ap = argparse.ArgumentParser(description="L1 逐幀基元訓練")
    ap.add_argument("--folds", type=int, default=5)
    # 100 epoch 不是隨便挑的:30 epoch 時 val 0.59、100 epoch 時 0.74,
    # 而且 train 與 val 幾乎相等(0.737/0.738)——這個規模是欠擬合,
    # 不是過擬合,該給的是訓練時間而不是更多正規化。
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pose-dir", default="annotations/pose")
    ap.add_argument("--save", default="checkpoints/hier_l1.pt")
    ap.add_argument("--dump", type=int, default=0,
                    help="訓練後印幾段的基元時間軸")
    args = ap.parse_args()
    set_seed(42)
    device = resolve_device("auto")

    items = load_pose_items(args.pose_dir)
    w, st = class_weights(items, device)
    tot = sum(st.values())
    print(f"樣本 {len(items)} 段,逐幀偽標籤分佈(共 {tot:,} 幀×手):")
    print(f"  {'棄權':6s} {st.get(IGNORE, 0):7,d}  "
          f"{st.get(IGNORE, 0) / tot:5.1%}")
    for c in range(NUM_PRIMITIVES):
        print(f"  {PRIM_NAMES[PRIMITIVES[c]]:6s} {st.get(c, 0):7,d}  "
              f"{st.get(c, 0) / tot:5.1%}   權重 {w[c]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    rng = np.random.RandomState(42)
    order = rng.permutation(len(items))
    folds = [order[i::args.folds] for i in range(args.folds)]

    accs, mats = [], []
    for k in range(args.folds):
        val_i = set(folds[k].tolist())
        tr = [items[i] for i in range(len(items)) if i not in val_i]
        va = [items[i] for i in sorted(val_i)]
        if not tr or not va:
            continue
        tr_ds = PrimitiveDataset(tr, args.window, train=True, seed=k)
        va_ds = PrimitiveDataset(va, args.window, train=False)
        tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=args.batch)

        model = PrimitiveNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        for _ in range(args.epochs):
            run_epoch(model, tr_ld, criterion, device, opt)
            sched.step()
        _, acc, cm = run_epoch(model, va_ld, criterion, device)
        accs.append(acc)
        mats.append(cm)
        print(f"fold {k}: 規則還原率 {acc:.3f}({len(va)} 段)")

    cm = np.sum(mats, axis=0)
    print(f"\n=== {args.folds}-fold 匯總:規則還原率 "
          f"{np.mean(accs):.3f} ± {np.std(accs):.3f} ===")
    names = [PRIM_NAMES[p] for p in PRIMITIVES]
    print(f"{'真值\\預測':10s}" + "".join(f"{n:>8s}" for n in names)
          + f"{'召回':>8s}")
    for i, n in enumerate(names):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        print(f"{n:10s}" + "".join(f"{v:8d}" for v in cm[i])
              + f"{rec:8.2f}")

    # 全量重訓當成正式權重(交叉驗證只用來估效能)
    full = PrimitiveDataset(items, args.window, train=True, seed=99)
    loader = DataLoader(full, batch_size=args.batch, shuffle=True)
    model = PrimitiveNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    for _ in range(args.epochs):
        run_epoch(model, loader, criterion, device, opt)
        sched.step()

    import os
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "primitives": PRIMITIVES,
                "params": count_params(model)}, args.save)
    print(f"\n全量重訓完成,權重存至 {args.save}"
          f"(參數量 {count_params(model):,})")

    if args.dump:
        from stage2.infer_hier import HierarchicalRecognizer
        rec = HierarchicalRecognizer(l1_ckpt=args.save, device=device)
        for it in items[:args.dump]:
            print(f"\n──── {it['stem']}(標籤 {it['label']})────")
            print(rec.explain(it["kpts"], it["fps"]))


if __name__ == "__main__":
    main()
