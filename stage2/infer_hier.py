"""兩層模型的推論介面。

一次輸入一段節點序列(T,17,3),輸出:
  - 淺層基元時間軸(可以直接畫在 GUI 上)
  - 深層各類別分數
  - 一段人看得懂的判定理由

權重的有無決定走哪條路,但**介面不變**:
  L1 有權重 → 逐幀基元由網路給;沒有 → 由規則給
  L2 有權重 → 類別分數由網路給;沒有 → 由片段文法給
所以還沒標資料的現在就能上線,標完之後換權重即可,呼叫端不用改。

用法:
    rec = HierarchicalRecognizer(l1_ckpt="checkpoints/hier_l1.pt")
    out = rec.predict(kpts, fps=10.0)
    print(out["top"], out["scores"])
    print(rec.explain(kpts, fps=10.0))
"""
from typing import Optional

import numpy as np
import torch

from stage2.composition import (STAT_DIM, Analysis, analyze, explain,
                                grammar_scores, normalize_stats)
from stage2.hier_model import TOKEN_DIM, CompositionNet, PrimitiveNet
from stage2.kinematics import graph_features, kinematic_features
from stage2.taxonomy import DEEP_CLASSES, DEEP_NAMES
from utils import resolve_device


class HierarchicalRecognizer:
    """骨架序列 → 淺層基元 → 深層動作。"""

    def __init__(self, l1_ckpt: Optional[str] = None,
                 l2_ckpt: Optional[str] = None, device="auto"):
        self.device = resolve_device(device) if isinstance(device, str) \
            else device
        self.l1 = None
        self.l2 = None
        if l1_ckpt:
            ck = torch.load(l1_ckpt, map_location=self.device,
                            weights_only=False)
            self.l1 = PrimitiveNet().to(self.device)
            self.l1.load_state_dict(ck["model"])
            self.l1.eval()
        if l2_ckpt:
            ck = torch.load(l2_ckpt, map_location=self.device,
                            weights_only=False)
            # token_dim 與 encoder 一定要從權重檔讀回來,不能吃預設值:
            # 沒有 L1 時 token 少了 16 維嵌入(44 → 28),用預設值建出來的
            # 網路形狀對不上;encoder 更是直接決定有哪些權重。
            self.l2 = CompositionNet(
                num_classes=len(ck.get("classes", DEEP_CLASSES)),
                token_dim=int(ck.get("token_dim", TOKEN_DIM)),
                stat_dim=int(ck.get("stat_dim", STAT_DIM)),
                encoder=ck.get("encoder", "transformer")).to(self.device)
            self.l2.load_state_dict(ck["model"])
            self.l2.eval()
            self.l2_classes = ck.get("classes", DEEP_CLASSES)
            self.l2_encoder = ck.get("encoder", "transformer")

    # ---- 淺層 ----

    @torch.no_grad()
    def _run_l1(self, kpts: np.ndarray, fps: float):
        """→ (逐幀基元 (T,2), 逐幀嵌入 (T,2,E));無權重時回傳 (None, None)。"""
        if self.l1 is None:
            return None, None
        g = torch.from_numpy(graph_features(kpts)).permute(2, 0, 1)
        kin = torch.from_numpy(kinematic_features(kpts, fps))
        logits, emb = self.l1(g.unsqueeze(0).to(self.device),
                              kin.unsqueeze(0).to(self.device))
        return (logits[0].argmax(-1).cpu().numpy().astype(np.int8),
                emb[0].cpu().numpy())

    def analyze(self, kpts: np.ndarray, fps: float = 10.0) -> Analysis:
        """跑完淺層並切出片段序列。"""
        kin = kinematic_features(kpts, fps)
        prim, emb = self._run_l1(kpts, fps)
        dim = self.l1.embed_dim if self.l1 is not None else 0
        return analyze(kin, fps, prim=prim, frame_embed=emb, embed_dim=dim)

    # ---- 深層 ----

    @torch.no_grad()
    def predict(self, kpts: np.ndarray, fps: float = 10.0) -> dict:
        """→ {scores, top, source, analysis}。scores 為類別 → 機率。"""
        a = self.analyze(kpts, fps)
        if self.l2 is not None:
            tokens = torch.from_numpy(a.tokens).unsqueeze(0).to(self.device)
            times = torch.from_numpy(a.times).unsqueeze(0).to(self.device)
            mask = torch.ones(1, a.tokens.shape[0], dtype=torch.bool,
                              device=self.device)
            stats = torch.from_numpy(
                normalize_stats(a.stats)).unsqueeze(0).to(self.device)
            p = self.l2(tokens, times, mask, stats).softmax(-1)[0]
            scores = {c: float(v) for c, v in zip(self.l2_classes, p)}
            source = "learned"
        else:
            scores = grammar_scores(a.segments, a.stats, a.cycles)
            source = "grammar"
        top = max(scores, key=scores.get)
        return {"scores": scores, "top": top, "source": source,
                "analysis": a}

    def explain(self, kpts: np.ndarray, fps: float = 10.0) -> str:
        """人看的完整說明。"""
        out = self.predict(kpts, fps)
        a = out["analysis"]
        head = (f"淺層來源:{'L1 網路' if self.l1 else '規則'}   "
                f"深層來源:{'L2 網路' if self.l2 else '片段文法'}")
        body = explain(a.segments, a.stats, out["scores"])
        tail = (f"→ 判定:{DEEP_NAMES.get(out['top'], out['top'])} "
                f"({out['scores'][out['top']]:.2f})")
        return "\n".join([head, body, tail])


def main():
    import argparse
    import glob

    ap = argparse.ArgumentParser(description="兩層模型推論(單段/整批)")
    ap.add_argument("--pose", default="annotations/pose/*.npz")
    ap.add_argument("--l1", default=None)
    ap.add_argument("--l2", default=None)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    rec = HierarchicalRecognizer(l1_ckpt=args.l1, l2_ckpt=args.l2)
    for path in sorted(glob.glob(args.pose))[:args.limit]:
        d = np.load(path, allow_pickle=True)
        print(f"\n──── {path} ────")
        print(rec.explain(d["kpts"], float(d["fps"]) or 10.0))


if __name__ == "__main__":
    main()
