"""L2 訓練:淺層片段序列 → 具體動作(抽菸/喝水/講電話/扶眼鏡/抓頭髮)。

**這支腳本能不能跑得出東西,取決於標籤,不是模型。**
目前 annotations/clip_labels.json 的 87 段裡只有 6 段 smoking,其餘是
2026-07 標的粗類(other_neg / desk_work),對得上深層詞彙的只有
「抽菸」與「其他」兩類。所以現在跑出來的實際上是二分類,
腳本會據實印出類別分佈與警告,不會假裝在做六分類。

要拿到真正的六分類,先用 scripts/label_tool.py 以新的細類重標
(扶眼鏡 / 抓頭髮 / 喝水 / 講電話 各自有按鈕),再跑這支。

評估:leave-one-out 或 k-fold(段數太少,分層 k-fold 為主),
主指標是抽菸維度的 AUC 與「固定召回下的誤報過濾率」——
後者才對應實際價值(第二階段是過濾器,不是分類器)。

用法:
    python -m stage2.train_l2                      # 用規則產生片段
    python -m stage2.train_l2 --l1 checkpoints/hier_l1.pt   # 用 L1 產生
"""
import argparse
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from stage2.composition import STAT_DIM, grammar_scores
from stage2.hier_dataset import (CompositionDataset, collate_tokens,
                                 load_pose_items)
from stage2.hier_model import CompositionNet, PrimitiveNet, count_params
from stage2.kinematics import graph_features, kinematic_features
from stage2.taxonomy import DEEP_CLASSES, DEEP_NAMES, COARSE_NEGATIVE_CODES
from utils import resolve_device, set_seed


def make_l1_fns(ckpt: str, device):
    """把訓練好的 L1 包成 CompositionDataset 要的兩個回呼。"""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    net = PrimitiveNet().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    cache = {}

    def _run(it):
        key = it["stem"]
        if key not in cache:
            g = torch.from_numpy(graph_features(it["kpts"])).permute(2, 0, 1)
            kin = torch.from_numpy(kinematic_features(it["kpts"], it["fps"]))
            with torch.no_grad():
                lo, em = net(g.unsqueeze(0).to(device),
                             kin.unsqueeze(0).to(device))
            cache[key] = (lo[0].argmax(-1).cpu().numpy().astype(np.int8),
                          em[0].cpu().numpy())
        return cache[key]

    return (lambda it: _run(it)[0], lambda it: _run(it)[1], net.embed_dim)


def evaluate(model, loader, device, n_classes):
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for tk, tm, mask, st, y in loader:
            p = model(tk.to(device), tm.to(device), mask.to(device),
                      st.to(device)).softmax(-1)
            probs.append(p.cpu().numpy())
            gts.append(y.numpy())
    return np.concatenate(probs), np.concatenate(gts)


def filter_rate_at_full_recall(smoke_prob, y_smoke):
    """固定 100% 抽菸召回下,能濾掉多少非抽菸片段。

    這是第二階段的實用指標:過濾器不准漏掉真的抽菸,在這個前提下
    誤報少掉幾成才是它的價值。
    """
    pos = smoke_prob[y_smoke == 1]
    neg = smoke_prob[y_smoke == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = pos.min()
    return float((neg < thr).mean())


def main():
    ap = argparse.ArgumentParser(description="L2 片段序列分類訓練")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l1", default=None, help="L1 權重;不給則走規則路徑")
    ap.add_argument("--pose-dir", default="annotations/pose")
    ap.add_argument("--labels", default="annotations/clip_labels.json")
    ap.add_argument("--save", default="checkpoints/hier_l2.pt")
    args = ap.parse_args()
    set_seed(42)
    device = resolve_device("auto")

    items = load_pose_items(args.pose_dir, args.labels)
    prim_fn = emb_fn = None
    embed_dim = 0
    if args.l1:
        prim_fn, emb_fn, embed_dim = make_l1_fns(args.l1, device)
        print(f"淺層來源:L1 網路 {args.l1}(嵌入 {embed_dim} 維)")
    else:
        print("淺層來源:規則(未指定 --l1)")

    ds = CompositionDataset(items, prim_fn, emb_fn, embed_dim)
    labels = np.array([s["label"] for s in ds.samples])
    present = sorted(set(labels.tolist()))
    dist = Counter(DEEP_CLASSES[c] for c in labels)
    print(f"可用樣本 {len(ds)} 段,深層類別分佈:"
          + "  ".join(f"{DEEP_NAMES[k]} {v}" for k, v in dist.items()))
    coarse = sum(1 for it in items
                 if it["label"] in COARSE_NEGATIVE_CODES)
    if len(present) < len(DEEP_CLASSES):
        missing = [DEEP_NAMES[c] for i, c in enumerate(DEEP_CLASSES)
                   if i not in present]
        print(f"⚠ 缺類別:{'、'.join(missing)} —— 這些類別現在學不到,"
              f"要先用 scripts/label_tool.py 補標")
    if coarse:
        print(f"⚠ 其中 {coarse} 段是 2026-07 的粗類負樣本"
              f"(other_neg/desk_work),裡面混著扶眼鏡與抓頭髮;"
              f"要做六分類前應該複標")

    token_dim = ds.samples[0]["tokens"].shape[1]
    counts = np.array([max((labels == c).sum(), 1)
                       for c in range(len(DEEP_CLASSES))], float)
    w = torch.tensor(
        [(counts.sum() / (len(present) * counts[c])) if c in present else 0.0
         for c in range(len(DEEP_CLASSES))],
        dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # 分層 k-fold(每類各自輪流分配,類別再少也不會整折缺席)
    rng = np.random.RandomState(42)
    folds = [[] for _ in range(args.folds)]
    for c in present:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % args.folds].append(int(i))

    all_p, all_y = [], []
    for k in range(args.folds):
        val_i = folds[k]
        tr_i = [i for i in range(len(ds)) if i not in set(val_i)]
        if not val_i or not tr_i:
            continue
        tr = DataLoader(Subset(ds, tr_i), batch_size=args.batch,
                        shuffle=True, collate_fn=collate_tokens)
        va = DataLoader(Subset(ds, val_i), batch_size=args.batch,
                        collate_fn=collate_tokens)
        model = CompositionNet(token_dim=token_dim,
                               stat_dim=STAT_DIM).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        model.train()
        for _ in range(args.epochs):
            for tk, tm, mask, st, y in tr:
                loss = criterion(model(tk.to(device), tm.to(device),
                                       mask.to(device), st.to(device)),
                                 y.to(device))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
        p, y = evaluate(model, va, device, len(DEEP_CLASSES))
        all_p.append(p)
        all_y.append(y)
        print(f"fold {k}: {len(val_i)} 段,"
              f"acc {(p.argmax(1) == y).mean():.3f}")

    p = np.concatenate(all_p)
    y = np.concatenate(all_y)
    smoke_i = DEEP_CLASSES.index("smoking")
    y_smoke = (y == smoke_i).astype(int)
    print(f"\n=== {args.folds}-fold 匯總({len(y)} 段)===")
    print(f"整體正確率 {(p.argmax(1) == y).mean():.3f}")
    if y_smoke.sum() and (1 - y_smoke).sum():
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_smoke, p[:, smoke_i])
        print(f"抽菸維度 AUC {auc:.3f}"
              f"(正樣本 {int(y_smoke.sum())} 段)")
        print(f"100% 抽菸召回下的誤報過濾率 "
              f"{filter_rate_at_full_recall(p[:, smoke_i], y_smoke):.1%}")

        # 文法基線:學習版沒贏過它就沒有意義
        g = np.array([grammar_scores(s["analysis"].segments,
                                     s["analysis"].stats,
                                     s["analysis"].cycles)["smoking"]
                      for s in ds.samples])
        g_auc = roc_auc_score(y_smoke_all(ds, smoke_i), g)
        print(f"(對照)片段文法基線 AUC {g_auc:.3f}")
        if auc <= g_auc:
            print(f"\n⚠ 學習版沒有贏過文法基線({auc:.3f} ≤ {g_auc:.3f})。"
                  f"\n  正樣本只有 {int(y_smoke.sum())} 段,這個結果是資料量的"
                  f"問題,不是架構的問題——L2 有 {count_params(model):,} 個"
                  f"參數,靠 {int(y_smoke.sum())} 段正樣本學不動。"
                  f"\n  在補到足夠正樣本之前,推論請**不要**載入這個權重:"
                  f"HierarchicalRecognizer 不給 l2_ckpt 就會自動走文法,"
                  f"那目前比較準。")
        else:
            print(f"\n✓ 學習版勝過文法基線({auc:.3f} > {g_auc:.3f}),"
                  f"可以把 {args.save} 掛上 HierarchicalRecognizer。")

    from sklearn.metrics import classification_report
    names = [DEEP_NAMES[DEEP_CLASSES[c]] for c in present]
    print(classification_report(y, p.argmax(1), labels=present,
                                target_names=names, zero_division=0))

    # 全量重訓存檔
    full = DataLoader(ds, batch_size=args.batch, shuffle=True,
                      collate_fn=collate_tokens)
    model = CompositionNet(token_dim=token_dim,
                           stat_dim=STAT_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    model.train()
    for _ in range(args.epochs):
        for tk, tm, mask, st, yy in full:
            loss = criterion(model(tk.to(device), tm.to(device),
                                   mask.to(device), st.to(device)),
                             yy.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": DEEP_CLASSES,
                "token_dim": token_dim, "l1": args.l1}, args.save)
    print(f"權重存至 {args.save}(參數量 {count_params(model):,})")


def y_smoke_all(ds, smoke_i):
    return np.array([1 if s["label"] == smoke_i else 0 for s in ds.samples])


if __name__ == "__main__":
    main()
