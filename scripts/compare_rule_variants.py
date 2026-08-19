"""在已錄的警報片段上重播規則路徑,比較 rule 與 rule+order。

**這支腳本能回答什麼、不能回答什麼,先講清楚。**

能回答:**事件層級**的差異 —— 加上「必須看得到放下(S3)」之後,
哪些手到嘴事件被擋掉、它們來自哪一類片段。

不能回答:**警報層級**的差異。警報要求 90 秒視窗內 ≥ 3 次事件,而
`alarms/clips` 的片段中位數只有 13.6 秒 —— 片段長度根本放不下這個視窗,
重播出來的事件數最多只有 2,任何變體都不會觸發警報。想比警報層級的
精確率/召回率,只能重錄連續影像,不能靠現有片段。

也要記得這批片段本身是**規則已經開過火**的片段(見 gui.py 的 _write_clip),
所以它是條件樣本,不是隨機樣本。

用法:
    python -m scripts.compare_rule_variants
    python -m scripts.compare_rule_variants --release-window 1.5
"""
import argparse
from collections import Counter, defaultdict

import numpy as np

from inference import methods as reg
from inference.skeleton import SkeletonStageEstimator
from inference.state_machine import HandToMouthCounter
from stage2.hier_dataset import load_pose_items
from utils import load_config

VARIANTS = ("rule", "rule+order")
LABEL_COL = "片段標籤"


def replay(item: dict, method, cfg: dict, release_window: float):
    """把一段節點序列餵過規則路徑,回傳 (計入的事件數, [被擋掉的原因])。

    estimator 與 counter 的參數全部從 `Method.apply()` 之後的設定讀,
    確保重播用的門檻與 GUI 實際跑的是同一組。
    """
    conf = method.apply(cfg)
    sk = conf.get("skeleton", {})
    esc = conf.get("escalation", {})
    fps = item["fps"] or 10.0

    est = SkeletonStageEstimator(
        near_ratio=sk.get("near_ratio", 0.9),
        move_ratio=sk.get("move_ratio", 0.35),
        kpt_conf=sk.get("kpt_conf", 0.3),
        nose_conf=sk.get("nose_conf", 0.5),
        min_scale_px=sk.get("min_scale_px", 24.0),
        kpt_err_px=sk.get("kpt_err_px", 4.0),
        rise_margin=sk.get("rise_margin", 0.5),
        fps=fps)
    cnt = HandToMouthCounter(
        window_sec=esc.get("window_sec", 90.0),
        min_dwell=esc.get("min_dwell", 2.0),
        max_dwell=esc.get("max_dwell", 5.0),
        min_gap=esc.get("min_gap", 2.0),
        gap_tolerance=esc.get("gap_tolerance", 0.5),
        require_release=esc.get("require_release", False),
        release_window=release_window)

    counted, rejected = 0, []
    for i, k in enumerate(item["kpts"]):
        # 全零的幀代表該幀沒有關聯到對象(見 stage2/extract_pose.py)
        stage, _, _ = est.update(k if float(k[:, 2].max()) > 0 else None)
        r = cnt.update(stage, i / fps)
        if r is None:
            continue
        _, ok, why = r
        if ok:
            counted += 1
        else:
            rejected.append(why)
    return counted, rejected


def main():
    ap = argparse.ArgumentParser(description="rule vs rule+order 事件層級比較")
    ap.add_argument("--pose-dir", default="annotations/pose")
    ap.add_argument("--labels", default="annotations/clip_labels.json")
    ap.add_argument("--config", default="configs/inference.yaml")
    ap.add_argument("--release-window", type=float, default=None,
                    help="覆寫設定檔的 escalation.release_window(秒)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    window = (args.release_window if args.release_window is not None
              else cfg.get("escalation", {}).get("release_window", 2.0))
    items = [i for i in load_pose_items(args.pose_dir, args.labels)
             if i.get("label")]
    dur = [len(i["kpts"]) / (i["fps"] or 10.0) for i in items]
    print(f"片段 {len(items)} 段,長度中位數 {np.median(dur):.1f}s "
          f"(最長 {max(dur):.1f}s)")
    print(f"⚠ 事件計數視窗是 {cfg['escalation']['window_sec']:.0f} 秒、"
          f"min_events={cfg['alarm']['min_events']} —— 片段放不下,"
          f"本表只能比事件層級")
    print()

    per: dict = defaultdict(dict)
    rej: dict = defaultdict(Counter)
    for key in VARIANTS:
        m = reg.get(key)
        for it in items:
            n, why = replay(it, m, cfg, window)
            per[it["stem"]][key] = n
            per[it["stem"]]["label"] = it["label"]
            rej[key].update(why)

    print(f"  {'方法':<14}{'事件總數':>9}{'有事件的片段':>13}")
    print("  " + "─" * 40)
    for key in VARIANTS:
        tot = sum(p[key] for p in per.values())
        nz = sum(1 for p in per.values() if p[key] > 0)
        print(f"  {key:<14}{tot:>9}{nz:>13}")

    # 依片段標籤拆開:擋掉的是誤報還是真樣本,才是這個變體的價值所在
    print()
    print(f"  {LABEL_COL:<12}", end="")
    for key in VARIANTS:
        print(f"{key:>12}", end="")
    print(f"{'擋掉':>8}{'擋掉率':>9}")
    print("  " + "─" * 54)
    labels = sorted({p["label"] for p in per.values()})
    for lab in labels:
        a = sum(p[VARIANTS[0]] for p in per.values() if p["label"] == lab)
        b = sum(p[VARIANTS[1]] for p in per.values() if p["label"] == lab)
        if a == 0:
            continue
        print(f"  {lab:<12}{a:>12}{b:>12}{a - b:>8}{(a - b) / a:>9.0%}")

    diff = [(s, p) for s, p in per.items() if p[VARIANTS[0]] != p[VARIANTS[1]]]
    print(f"\n事件數有變的片段:{len(diff)} 段")
    for s, p in sorted(diff, key=lambda x: x[1]["label"]):
        print(f"  {p['label']:<11} {p[VARIANTS[0]]} → {p[VARIANTS[1]]}   {s}")

    changed = Counter(p["label"] for _, p in diff)
    pos = changed.get("smoking", 0)
    print(f"\n被擋掉的事件來自:{dict(changed) or '(無)'}")
    if diff:
        print(f"  其中抽菸片段 {pos} 段 —— 擋掉真樣本的代價")

    print("\n各方法的否決原因分佈:")
    for key in VARIANTS:
        line = "  ".join(f"{k} {v}" for k, v in rej[key].most_common())
        print(f"  {key:<14}{line or '(無)'}")


if __name__ == "__main__":
    main()
