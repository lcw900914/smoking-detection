"""第二階段 Phase 0 測試:正規化不變性、模型形狀、翻轉增強一致性。"""
import numpy as np
import pytest
import torch

from stage2.normalize import normalize_sequence, FEATURE_DIM
from stage2.model import PoseTCN, CLASSES


def make_seq(T=20, cx=100.0, cy=100.0, scale=1.0, conf=0.9):
    """合成節點序列:雙肩/雙髖固定,右腕做週期運動。"""
    rng = np.random.RandomState(0)
    seq = np.zeros((T, 17, 3), np.float32)
    seq[:, :, 2] = conf
    base = {5: (-20, 0), 6: (20, 0), 11: (-15, 60), 12: (15, 60), 0: (0, -30)}
    for j in range(17):
        dx, dy = base.get(j, (rng.uniform(-25, 25), rng.uniform(-10, 70)))
        seq[:, j, 0] = cx + dx * scale
        seq[:, j, 1] = cy + dy * scale
    t = np.arange(T)
    seq[:, 10, 0] = cx + (25 + 10 * np.sin(t / 3)) * scale   # 右腕運動
    seq[:, 10, 1] = cy + (10 - 30 * np.sin(t / 3)) * scale
    return seq


class TestNormalize:
    def test_output_shape(self):
        out = normalize_sequence(make_seq(T=20))
        assert out.shape == (20, FEATURE_DIM)

    def test_translation_invariance(self):
        """整體平移(人在畫面不同位置)不改變特徵。"""
        a = normalize_sequence(make_seq(cx=100, cy=100))
        b = normalize_sequence(make_seq(cx=800, cy=400))
        assert np.allclose(a, b, atol=1e-5)

    def test_scale_invariance(self):
        """整體縮放(離鏡頭遠近)不改變特徵。"""
        a = normalize_sequence(make_seq(scale=1.0))
        b = normalize_sequence(make_seq(scale=3.0))
        assert np.allclose(a, b, atol=1e-5)

    def test_low_conf_zeroed(self):
        seq = make_seq()
        seq[:, 9, 2] = 0.1                     # 左腕不可見
        out = normalize_sequence(seq)
        coords = out[:, :34].reshape(-1, 17, 2)
        assert np.all(coords[:, 9] == 0)

    def test_velocity_first_frame_zero(self):
        out = normalize_sequence(make_seq())
        vel = out[:, 34:68]
        assert np.all(vel[0] == 0)
        assert np.any(vel[1:] != 0)            # 腕在動,速度非零


class TestModel:
    def test_forward_shapes(self):
        model = PoseTCN()
        x = torch.randn(4, 128, FEATURE_DIM)
        logits, emb = model(x)
        assert logits.shape == (4, len(CLASSES))
        assert emb.shape == (4, 128)

    def test_variable_length(self):
        """TCN 對任意時長皆可前向(節律版會用到變長)。"""
        model = PoseTCN()
        for T in (30, 128, 300):
            logits, _ = model(torch.randn(2, T, FEATURE_DIM))
            assert logits.shape == (2, len(CLASSES))

    def test_backward(self):
        model = PoseTCN()
        logits, _ = model(torch.randn(4, 64, FEATURE_DIM))
        logits.sum().backward()

    def test_param_budget(self):
        """小資料配方:參數量須壓在 ~15 萬以內。"""
        n = sum(p.numel() for p in PoseTCN().parameters())
        assert n < 150_000, f"參數量 {n:,} 超出小資料預算"
