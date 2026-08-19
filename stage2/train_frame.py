"""單幀對照組訓練 —— 量出「只看一張畫面」的天花板。

這支腳本存在的唯一理由是回答一個問題:
**多幀時序模型比單幀好多少?**

所以它不只訓練單幀模型,還在**同一批段、同一組 fold** 上把主線的兩條
路徑(片段文法、L1 網路基元 + 文法)一起算出來,最後印成一張可以直接
貼進論文的比較表。分開跑兩支腳本再把數字抄在一起是不行的:fold 不同、
可用段數不同,差距有多少是方法造成的就講不清楚。

對齊了哪些變因
──────────────
  同一份骨架      annotations/pose/*.npz(同一顆 YOLO pose 抽的)
  同一組標籤      annotations/clip_labels.json → taxonomy.deep_index
  同一套 fold     種子 42、分層 k-fold,與 stage2/train_l2.py 逐行相同
  同一個指標      抽菸維度 AUC + 100% 召回下的誤報過濾率
  唯一的變因      模型看不看得到時間軸

沒有候選幀的段怎麼算
────────────────────
「一幀手都沒舉起來」的段,單幀模型沒有輸入,抽菸分數記 0。**這些段仍
然計入評估**——主線模型在同樣的段上照樣會給分數,把它們挑掉等於幫
對照組作弊。腳本另外會印一份「只算兩邊都有輸入的段」的數字當作參考。

用法:
    python -m stage2.train_frame                        # MLP 對照組
    python -m stage2.train_frame --arch gcn             # 單幀圖網路
    python -m stage2.train_frame --l1 checkpoints/hier_l1.pt   # 連 L1 一起比
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from stage2.composition import analyze, grammar_scores
from stage2.frame_baseline import (ARCHES, FRAME_ARM_DIM,
                                   FRAME_GLOBAL_DIM, FrameDataset,
                                   MIN_TOPK, TOPK_RATIO, aggregate,
                                   build_model)
from stage2.hier_dataset import load_pose_items
from stage2.kinematics import kinematic_features
from stage2.taxonomy import (COARSE_NEGATIVE_CODES, DEEP_CLASSES,
                             DEEP_NAMES)
from utils import resolve_device, set_seed

SMOKE_I = DEEP_CLASSES.index("smoking")


# ---------------------------------------------------------------------
# 指標
# ---------------------------------------------------------------------

def filter_rate_at_full_recall(score: np.ndarray, y: np.ndarray) -> float:
    """在「一個抽菸都不漏」的門檻下,擋掉了多少比例的負樣本。

    這才是第二階段的實際價值:它是過濾器,不是分類器。AUC 好看但在
    100% 召回處擋不掉東西的模型,掛上去等於沒掛。
    """
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((neg < pos.min()).mean())


def report(name: str, score: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, score) if y.sum() and (1 - y).sum() else float("nan")
    fr = filter_rate_at_full_recall(score, y)
    return {"name": name, "auc": float(auc), "filter": fr, "n": len(y),
            "score": np.asarray(score, float), "y": np.asarray(y, int)}


def print_table(rows, title: str):
    print(f"\n{title}")
    print(f"  {'方法':<34}{'抽菸 AUC':>10}{'100%召回過濾率':>16}")
    print("  " + "─" * 60)
    for r in rows:
        fr = "—" if np.isnan(r["filter"]) else f"{r['filter']:.1%}"
        print(f"  {r['name']:<34}{r['auc']:>10.3f}{fr:>16}")


# ---------------------------------------------------------------------
# 混淆矩陣
#
# **這個任務不能看正確率。** 64 段裡 58 段是負樣本,「全部猜不是抽菸」
# 就有 90.6% 正確率,而且完全沒有用。要看的是抽菸那一列:漏了幾段
# (FN)、誤報幾段(FP)。所以下面每個操作點都把 2×2 攤開來印。
#
# 三個操作點各自回答不同的問題:
#   門檻 0.25   verifier.MIN_SMOKING,系統實際在用的那條線
#   100% 召回   一段都不漏的前提下,還剩幾個誤報 —— 第二階段是過濾器,
#               「降級不否決」是紅線,這才是它真正的價值指標
#   最佳 F1     這份資料理論上最好能到哪(會過擬合到驗證集,只作參考)
# ---------------------------------------------------------------------

OP_THRESHOLD = 0.25          # inference/verifier.py 的 MIN_SMOKING


def confusion(score: np.ndarray, y: np.ndarray, thr: float):
    """→ (tp, fn, fp, tn)。"""
    pred = score >= thr
    return (int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum()),
            int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum()))


def prf(tp: int, fn: int, fp: int, tn: int) -> dict:
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre and rec and pre + rec else 0.0
    acc = (tp + tn) / max(tp + fn + fp + tn, 1)
    return {"recall": rec, "precision": pre, "f1": f1, "acc": acc}


def best_f1_threshold(score: np.ndarray, y: np.ndarray) -> float:
    best, thr = -1.0, 0.5
    for t in np.unique(score):
        m = prf(*confusion(score, y, float(t)))
        if m["f1"] > best:
            best, thr = m["f1"], float(t)
    return thr


def operating_points(score: np.ndarray, y: np.ndarray):
    """→ [(操作點名稱, 門檻)]。"""
    pts = [(f"門檻 {OP_THRESHOLD}(verifier 實際在用)", OP_THRESHOLD)]
    pos = score[y == 1]
    if len(pos):
        pts.append(("100% 召回(一段都不漏)", float(pos.min())))
    pts.append(("最佳 F1(過擬合驗證集,參考用)",
                best_f1_threshold(score, y)))
    return pts


def print_confusion(name: str, score: np.ndarray, y: np.ndarray):
    print()
    print(f"  ── {name} " + "─" * max(0, 52 - len(name)))
    for label, thr in operating_points(score, y):
        tp, fn, fp, tn = confusion(score, y, thr)
        m = prf(tp, fn, fp, tn)
        print(f"    {label}   門檻 {thr:.3f}")
        print(f"        {'':<10}{'預測:抽菸':>10}{'預測:其他':>10}")
        print(f"        {'實際:抽菸':<10}{tp:>10}{fn:>10}"
              f"     ← 漏測 {fn}")
        print(f"        {'實際:其他':<10}{fp:>10}{tn:>10}"
              f"     ← 誤報 {fp}")
        print(f"        召回 {m['recall']:.0%}  精確 {m['precision']:.0%}  "
              f"F1 {m['f1']:.2f}  正確率 {m['acc']:.1%}")


def print_all_confusions(rows, y: np.ndarray, title: str):
    print()
    print(title)
    base = 1.0 - y.mean()
    print(f"  (基準線:全部猜「不是抽菸」= 正確率 {base:.1%}、召回 0%。"
          f"正確率在這個任務上沒有鑑別力,看漏測與誤報)")
    for r in rows:
        print_confusion(r["name"], r["score"], r["y"])


def print_class_breakdown(name: str, probs: np.ndarray, y: np.ndarray,
                          classes) -> None:
    """argmax 落在哪些類別 —— 看模型把負樣本當成什麼。"""
    pred = probs.argmax(1)
    print()
    print(f"  ── {name}:argmax 落點 " + "─" * 34)
    for real, tag in ((1, "實際抽菸"), (0, "實際其他")):
        m = y == real
        if not m.any():
            continue
        cnt = Counter(classes[c] for c in pred[m])
        line = "  ".join(f"{DEEP_NAMES.get(k, k)} {v}"
                         for k, v in cnt.most_common())
        print(f"    {tag}({int(m.sum())} 段):{line}")


# ---------------------------------------------------------------------
# 主線路徑的分數(對照用)
# ---------------------------------------------------------------------

def temporal_scores(clips, l1_ckpt=None, device=None) -> np.ndarray:
    """主線的段級抽菸分數。l1_ckpt 為 None 走純規則基元 + 片段文法。

    文法本身沒有可學參數,所以不需要分 fold——它在任何 fold 上都是
    同一組數字,直接整批算完最公平也最省事。
    """
    prim_fn = None
    if l1_ckpt:
        from stage2.hier_model import PrimitiveNet
        from stage2.kinematics import graph_features
        ck = torch.load(l1_ckpt, map_location=device, weights_only=False)
        net = PrimitiveNet().to(device)
        net.load_state_dict(ck["model"])
        net.eval()

        def prim_fn(kpts, fps):
            g = torch.from_numpy(graph_features(kpts)).permute(2, 0, 1)
            kin = torch.from_numpy(kinematic_features(kpts, fps))
            with torch.no_grad():
                lo, _ = net(g.unsqueeze(0).to(device),
                            kin.unsqueeze(0).to(device))
            return lo[0].argmax(-1).cpu().numpy().astype(np.int8)

    out = []
    for c in clips:
        prim = prim_fn(c["kpts"], c["fps"]) if prim_fn else None
        a = analyze(c["kin"], c["fps"], prim=prim)
        out.append(grammar_scores(a.segments, a.stats, a.cycles)["smoking"])
    return np.array(out, dtype=float)


# ---------------------------------------------------------------------
# 訓練
# ---------------------------------------------------------------------

def train_one(ds, train_idx, args, weights, device) -> nn.Module:
    model = build_model(args.arch, num_classes=len(DEEP_CLASSES),
                        use_global=args.global_features).to(device)
    if not train_idx:
        return model
    # pin_memory / 多 worker 在這台機器上會炸(見 lcw-windows-ml-env-quirks),
    # 樣本本來就只是 24 維向量,單進程反而快
    loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch,
                        shuffle=True, num_workers=0, pin_memory=False,
                        drop_last=len(train_idx) > args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(weight=weights)
    model.train()
    for epoch in range(args.epochs):
        ds.resample(epoch)          # 每個 epoch 重抽一輪關鍵點增強
        for x, y in loader:
            loss = crit(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def clip_probs(model, ds, clip_ids, device, ratio: float = TOPK_RATIO,
               min_k: int = MIN_TOPK) -> np.ndarray:
    """每段的各類別機率 (N, C);沒有候選幀的段走 abstain(抽菸 0)。

    回傳整個機率向量而不只是抽菸那一維:混淆矩陣要看模型把負樣本當成
    什麼(是判成「喝水」還是「其他」),只留一維就看不到了。
    """
    from stage2.frame_baseline import abstain_scores
    model.eval()
    n_cls = len(DEEP_CLASSES)
    out = np.zeros((len(clip_ids), n_cls), dtype=float)
    fallback = np.array([abstain_scores()[c] for c in DEEP_CLASSES])
    for i, ci in enumerate(clip_ids):
        feats = ds.clip_features(ci)
        if len(feats) == 0:
            out[i] = fallback
            continue
        p = model(torch.from_numpy(feats).to(device)).softmax(-1)
        out[i] = aggregate(p.cpu().numpy(), ratio, min_k)
    return out


def shortcut_probe(items, folds, y, args, device) -> float:
    """捷徑探針:**只**餵全域特徵(軀幹傾斜 / 遠近 / 朝向)訓練同一顆頭。

    這 7 維完全不含「手在做什麼」的資訊,照理說 AUC 應該貼著 0.5。
    真的分得出來,就代表模型可以靠「這是哪個機位、這個人多遠」猜答案
    ——那時候主表上的任何分數都不能當成「學會了動作」。

    2026-08-18 首跑:探針 0.940。6 段抽菸正樣本全部來自同一場錄影,
    整份資料在場景維度上是可分的。這是資料的問題,補跨場景正樣本才會好,
    調模型沒有用。這支探針就是為了讓這件事每次訓練都被看見。
    """
    ds = FrameDataset(items, arch="mlp", use_global=True)
    n = len(ds.clips)
    sl = slice(FRAME_ARM_DIM, FRAME_ARM_DIM + FRAME_GLOBAL_DIM)
    counts = np.array([max(int((ds.clip_labels == c).sum()), 0)
                       for c in range(len(DEEP_CLASSES))], float)
    present = [c for c in range(len(DEEP_CLASSES)) if counts[c] > 0]
    w = torch.tensor(
        [(counts.sum() / (len(present) * counts[c])) if counts[c] > 0 else 0.0
         for c in range(len(DEEP_CLASSES))],
        dtype=torch.float32, device=device)
    score = np.zeros(n)
    for k in range(len(folds)):
        val = folds[k]
        tr = [i for i in range(n) if i not in set(val)]
        idx = ds.indices_of_clips(tr)
        if not idx or not val:
            continue
        X = torch.from_numpy(np.stack(
            [ds.clips[ci]["feat"][j][sl] for ci, j in
             (ds.samples[i] for i in idx)])).float().to(device)
        Y = torch.tensor([ds.clips[ds.samples[i][0]]["label_index"]
                          for i in idx], device=device)
        m = nn.Sequential(
            nn.Linear(FRAME_GLOBAL_DIM, 64), nn.LayerNorm(64),
            nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(64, len(DEEP_CLASSES))).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=args.lr,
                                weight_decay=1e-2)
        crit = nn.CrossEntropyLoss(weight=w)
        m.train()
        for _ in range(args.epochs):
            perm = torch.randperm(len(X), device=device)
            for b in range(0, len(perm), args.batch):
                j = perm[b:b + args.batch]
                loss = crit(m(X[j]), Y[j])
                opt.zero_grad()
                loss.backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            for ci in val:
                f = ds.clips[ci]["feat"]
                if len(f) == 0:
                    continue
                p = m(torch.from_numpy(f[:, sl]).float().to(device))
                score[ci] = float(aggregate(
                    p.softmax(-1).cpu().numpy())[SMOKE_I])
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if y.sum() and (1 - y).sum()         else float("nan")


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int = 42):
    """與 stage2/train_l2.py 完全相同的分層 k-fold(逐行照抄,刻意的)。

    照抄而不是抽成共用函式:兩支腳本的 fold 必須位元級一致,共用函式
    哪天被改一個預設值,兩邊的數字就悄悄失去可比性,而且不會有任何
    錯誤訊息。這裡的重複是保險。
    """
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_folds)]
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % n_folds].append(int(i))
    return folds


def session_of(stem: str) -> str:
    """檔名 → 錄影場次(日期)。alarm_track1_20260708_152117 → 20260708。"""
    m = re.search(r"_(\d{8})_", stem)
    return m.group(1) if m else "unknown"


def session_folds(clips):
    """一個錄影場次 = 一折。**訓練與驗證永遠不共用場次。**

    為什麼需要這個切法
    ──────────────────
    捷徑探針量到「只看軀幹傾斜/遠近/朝向就有 AUC 0.94」,代表這份資料
    在場景維度上是可分的。分層隨機 k-fold 會把同一場錄影的段同時放進
    訓練與驗證兩邊,模型記住機位就能得分——那個分數量的是記憶力,不是
    辨識力。

    依場次切之後,驗證段的機位在訓練時完全沒見過。掉多少,就是原本有
    多少分是靠場景拿到的。這是現有資料能做到的最誠實的估計。

    代價要講清楚:抽菸正樣本只有兩個場次(20260708 五段、20260709 一段),
    所以驗證 0708 那一折,訓練集只剩 1 段正樣本。那一折的模型必然很差,
    **這正是這個切法要揭露的事**——不是實作有問題。
    """
    groups = {}
    for i, c in enumerate(clips):
        groups.setdefault(session_of(c["stem"]), []).append(i)
    return [groups[k] for k in sorted(groups)], sorted(groups)


def main():
    ap = argparse.ArgumentParser(description="單幀對照組訓練與比較")
    ap.add_argument("--arch", default="mlp", choices=ARCHES,
                    help="mlp = 單幀運動學;gcn = 單幀骨架圖(L1 拿掉時間卷積)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", default="stratified",
                    choices=("stratified", "session"),
                    help="stratified = 與 train_l2 相同的分層隨機 k-fold;"
                         "session = 依錄影場次切,訓練與驗證不共用機位"
                         "(誠實但嚴苛,見 session_folds 的說明)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--augment", action="store_true", default=True)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--global-features", action="store_true",
                    help="把軀幹傾斜/遠近/朝向也餵進去。預設關閉:"
                         "現有資料下它是捷徑,見 frame_baseline 的說明")
    ap.add_argument("--topk-ratio", type=float, default=TOPK_RATIO)
    ap.add_argument("--min-topk", type=int, default=MIN_TOPK)
    ap.add_argument("--l1", default=None,
                    help="L1 權重;給了就一併算 L1+文法 這條主線路徑")
    ap.add_argument("--pose-dir", default="annotations/pose")
    ap.add_argument("--labels", default="annotations/clip_labels.json")
    ap.add_argument("--save", default=None,
                    help="預設 checkpoints/frame_{arch}.pt")
    args = ap.parse_args()
    save = args.save or f"checkpoints/frame_{args.arch}.pt"
    set_seed(42)
    device = resolve_device("auto")

    # ---- 資料 ----
    items = load_pose_items(args.pose_dir, args.labels)
    ds = FrameDataset(items, arch=args.arch, augment=args.augment, seed=42,
                      use_global=args.global_features)
    y_clip = ds.clip_labels
    n_clip = len(ds.clips)
    if n_clip == 0:
        raise SystemExit("沒有可用的段:先跑 stage2.extract_pose 與標記工具")

    empty = [i for i, c in enumerate(ds.clips) if not c["picks"]]
    dist = Counter(DEEP_CLASSES[c] for c in y_clip)
    coarse = sum(1 for c in ds.clips
                 if c["label"] in COARSE_NEGATIVE_CODES)
    print(f"單幀對照組:{args.arch.upper()}   裝置 {device}")
    print(f"可用段 {n_clip},候選幀樣本 {len(ds)} 筆"
          f"({'幀×側' if args.arch == 'mlp' else '幀'})")
    print("段級類別分佈:"
          + "  ".join(f"{DEEP_NAMES[k]} {v}" for k, v in dist.items()))
    if empty:
        print(f"⚠ 其中 {len(empty)} 段一幀手都沒舉起來 → 單幀模型棄權"
              f"(抽菸分數記 0),但仍計入評估")
    if coarse:
        print(f"⚠ {coarse} 段是 2026-07 的粗類負樣本(other_neg/desk_work),"
              f"六分類前應複標;現在實際上在做二分類")

    # 類別權重:負樣本壓倒性多數,不加權模型會學成「全部猜 other」
    counts = np.array([max(int((y_clip == c).sum()), 0)
                       for c in range(len(DEEP_CLASSES))], float)
    present = [c for c in range(len(DEEP_CLASSES)) if counts[c] > 0]
    w = torch.tensor(
        [(counts.sum() / (len(present) * counts[c])) if counts[c] > 0 else 0.0
         for c in range(len(DEEP_CLASSES))],
        dtype=torch.float32, device=device)

    # ---- k-fold(依**段**切,絕不依幀切)----
    if args.split == "session":
        folds, names = session_folds(ds.clips)
        print(f"\n切法:依錄影場次({len(folds)} 個場次,"
              f"訓練與驗證不共用機位)")
    else:
        folds = stratified_folds(y_clip, args.folds, seed=42)
        names = [f"fold {k}" for k in range(len(folds))]
        print(f"\n切法:分層隨機 {args.folds}-fold"
              f"(與 train_l2 相同的種子與邏輯)")

    frame_score = np.full(n_clip, np.nan)
    frame_prob = np.zeros((n_clip, len(DEEP_CLASSES)))
    for k, val_c in enumerate(folds):
        tr_c = [i for i in range(n_clip) if i not in set(val_c)]
        if not val_c or not tr_c:
            continue
        model = train_one(ds, ds.indices_of_clips(tr_c), args, w, device)
        pr = clip_probs(model, ds, val_c, device,
                        args.topk_ratio, args.min_topk)
        frame_prob[val_c] = pr
        frame_score[val_c] = pr[:, SMOKE_I]
        n_pos = int((y_clip[val_c] == SMOKE_I).sum())
        tr_pos = int((y_clip[tr_c] == SMOKE_I).sum())
        note = "  ⚠ 訓練集正樣本不足" if (n_pos and tr_pos <= 1) else ""
        print(f"  {names[k]}:驗證 {len(val_c)} 段(抽菸 {n_pos}),"
              f"訓練抽菸 {tr_pos}{note}")
    assert not np.isnan(frame_score).any(), "有段沒有被任何 fold 驗證到"

    # ---- 對照:主線的兩條路徑(同一批段、同一組 fold)----
    y_all = (y_clip == SMOKE_I).astype(int)
    if not args.l1:
        print("\n(未指定 --l1,略過「L1 + 文法」那一列;"
              "加上 --l1 checkpoints/hier_l1.pt 可以一起比)")

    # 主線分數整批算一次:文法沒有可學參數,不隨 fold 改變
    grammar = temporal_scores(ds.clips, None, device)
    l1_gram = temporal_scores(ds.clips, args.l1, device) if args.l1 else None

    def rows_for(idx):
        """同一批段索引 → 各條路徑的指標,保證比的是同一群段。"""
        out = [report(f"單幀對照組 {args.arch.upper()}(本次)",
                      frame_score[idx], y_all[idx]),
               report("片段文法(規則基元,無學習權重)",
                      grammar[idx], y_all[idx])]
        if l1_gram is not None:
            out.append(report("L1 網路基元 + 片段文法",
                              l1_gram[idx], y_all[idx]))
        return out

    all_idx = np.arange(n_clip)
    rows = rows_for(all_idx)
    y = y_all
    tag = ("依錄影場次" if args.split == "session"
           else f"分層隨機 {args.folds}-fold")
    print(f"\n=== {tag} 匯總:全部 {n_clip} 段"
          f"(抽菸 {int(y.sum())})===")
    print_table(rows, "主指標(段級,抽菸維度)")

    if empty:
        keep = np.array([i for i in all_idx if i not in set(empty)])
        print_table(rows_for(keep),
                    f"參考:只算單幀模型有輸入的 {len(keep)} 段")

    # ---- 混淆矩陣 ----
    print_all_confusions(rows, y, f"混淆矩陣({tag}、全部 {n_clip} 段)")
    print_class_breakdown(f"單幀對照組 {args.arch.upper()}",
                          frame_prob, y, DEEP_CLASSES)

    # ---- 捷徑探針:上面那些數字有多少可以相信 ----
    probe = shortcut_probe(items, folds, y, args, device)
    print(f"""
捷徑探針(只用軀幹傾斜/遠近/朝向,完全不看手臂):抽菸 AUC {probe:.3f}""")
    if probe > 0.65:
        print("⚠ 探針分很高 —— 這份資料光看「人在畫面哪裡、多遠」"
              "就猜得出抽菸。")
        print("  代表正樣本的場景太集中(目前 6 段抽菸全來自 2026-07-08"
              " 同一場錄影),上表所有數字都要當成上界看,換場景會掉。")
        print("  治本只有一條:補**跨場景**的抽菸正樣本。")
        if args.global_features:
            print(f"  而且你開了 --global-features,對照組正在直接吃這條"
                  f"捷徑,分數不能用。")
    else:
        print("  探針接近隨機 → 上表的分數是真的從手臂幾何學到的。")

    best = max(rows, key=lambda r: (r["auc"] if not np.isnan(r["auc"])
                                    else -1))
    gap = rows[0]["auc"] - max(r["auc"] for r in rows[1:]) if len(rows) > 1 \
        else float("nan")
    print()
    leaky = probe > 0.65 and args.split != "session"
    if len(rows) > 1 and not np.isnan(gap):
        if leaky:
            print("結論:先不要下。分層隨機切法會把同一場錄影的段同時放進"
                  "訓練與驗證,")
            print(f"  而捷徑探針說這份資料靠場景就能猜到 {probe:.3f}。"
                  "上表量到的是記憶力不是辨識力。")
            print("  請改跑 --split session(訓練與驗證不共用機位),"
                  "那個數字才對得起結論。")
        elif gap >= 0:
            print(f"⚠ 單幀對照組沒有輸給時序方法(差距 {gap:+.3f})。")
            print(f"  在下結論說「時序模型比較好」之前先看這裡:正樣本只有"
                  f" {int(y.sum())} 段,這個比較的雜訊遠大於差距,")
            print("  而且時序方法目前走的是無參數文法。"
                  "補正樣本之前,兩邊的排序都不穩定。")
        else:
            print(f"✓ 時序方法勝出 {-gap:.3f} AUC(最佳:{best['name']})。")
            print("  這就是「多幀」買到的東西——單幀已經把骨架的靜態"
                  "資訊用光了,剩下的差距只能來自時間軸。")
            if args.split == "session":
                print("  這一輪是跨場次驗證(訓練沒看過驗證段的機位),"
                      "所以這個差距不是靠記憶場景拿到的。")

    # ---- 存檔:用全部段重訓一次(k-fold 只是拿來估效果的)----
    final = train_one(ds, list(range(len(ds))), args, w, device)
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": final.state_dict(), "arch": args.arch,
                "split": args.split,
                "classes": DEEP_CLASSES, "select": {},
                "use_global": args.global_features,
                "topk_ratio": args.topk_ratio, "min_topk": args.min_topk,
                "folds": args.folds, "metrics": rows}, save)
    print(f"\n已存 {save}(用全部 {n_clip} 段重訓)。"
          f"\n掛上 GUI:inference/methods.py 的 "
          f"「純規則 + 單幀對照組({args.arch.upper()})」那一列。")


if __name__ == "__main__":
    main()
