"""Dataset 測試:視窗索引、jpg clip dataset、離線特徵 dataset。"""
import json

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from data.dataset import window_indices, SmokingClipDataset
from data.feature_dataset import FeatureClipDataset
from models.ring_buffer import stack_time_to_channels


class TestWindowIndices:
    def test_basic(self):
        assert window_indices(anchor=10, T=4, stride=1) == [7, 8, 9, 10]

    def test_stride(self):
        assert window_indices(anchor=24, T=4, stride=8) == [0, 8, 16, 24]

    def test_clamp_padding(self):
        """越界 clamp 到 0(等價最舊幀重複填充,與 RingBuffer 一致)。"""
        assert window_indices(anchor=1, T=4, stride=1) == [0, 0, 0, 1]


def _make_clip_dir(root, clip_id, label, n_frames=20, size=32):
    """建立合成 clip:jpg 序列 + label.json。"""
    d = root / clip_id
    d.mkdir(parents=True)
    frames = []
    for i in range(n_frames):
        img = np.full((size, size, 3), i * 10 % 255, dtype=np.uint8)
        fname = f"img_{i:06d}.jpg"
        cv2.imwrite(str(d / fname), img)
        stage = "S2" if 5 <= i < 12 else "none"
        frames.append({"file": fname, "src_idx": i, "stage": stage,
                       "stage_id": 1 if 5 <= i < 12 else 3, "track_id": 1})
    meta = {"clip_id": clip_id, "label": label, "fps": 10.0,
            "num_frames": n_frames, "frames": frames}
    with open(d / "label.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)


class TestSmokingClipDataset:
    def test_shapes_and_labels(self, tmp_path):
        _make_clip_dir(tmp_path, "clip_a", "smoking")
        _make_clip_dir(tmp_path, "clip_b", "drinking")
        ds = SmokingClipDataset(str(tmp_path), short_T=4, long_T=4,
                                long_stride=2, augment=None, image_size=32)
        assert len(ds) == 2
        s = ds[0]
        assert s["short_images"].shape == (4, 3, 32, 32)
        assert s["long_images"].shape == (4, 3, 32, 32)
        assert s["stage_seq"].shape == (4,)
        assert s["clip_label"].item() == 1.0     # clip_a = smoking
        assert ds[1]["clip_label"].item() == 0.0  # clip_b = drinking

    def test_augment_runs(self, tmp_path):
        """增強模式可運作且輸出形狀不變。"""
        _make_clip_dir(tmp_path, "clip_a", "smoking")
        ds = SmokingClipDataset(
            str(tmp_path), short_T=4, long_T=4, long_stride=2,
            augment={"bbox_jitter": 0.08, "hflip": 0.5, "color_jitter": 0.3},
            image_size=32)
        s = ds[0]
        assert s["short_images"].shape == (4, 3, 32, 32)
        assert torch.isfinite(s["short_images"]).all()


class TestFeatureClipDataset:
    def test_layout_matches_ring_buffer(self, tmp_path):
        """離線特徵疊合 layout 必須與 stack_time_to_channels 一致。"""
        n, C, H, W = 30, 8, 4, 4
        feats = np.random.randn(n, C, H, W).astype(np.float16)
        np.save(tmp_path / "clip_x.npy", feats)
        with open(tmp_path / "clip_x.json", "w", encoding="utf-8") as f:
            json.dump({"clip_id": "clip_x", "label": "smoking",
                       "stage_ids": [3] * n}, f)

        ds = FeatureClipDataset(str(tmp_path), short_T=4, long_T=4,
                                long_stride=8, train=False)
        s = ds[0]
        assert s["short_feats"].shape == (4 * C, H, W)

        # 驗證模式 anchor = n-1 = 29;短視窗 = [26, 27, 28, 29]
        expected = stack_time_to_channels(
            torch.from_numpy(feats[[26, 27, 28, 29]].astype(np.float32)))
        assert torch.allclose(s["short_feats"], expected)

        # 長視窗 stride 8 = [5, 13, 21, 29]
        expected_long = stack_time_to_channels(
            torch.from_numpy(feats[[5, 13, 21, 29]].astype(np.float32)))
        assert torch.allclose(s["long_feats"], expected_long)
