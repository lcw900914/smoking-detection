"""ROI 平滑與裁切測試。"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from tracking.roi import ROISmoother, upper_body_box, crop_roi, crop_upper_body


class TestROISmoother:
    def test_first_frame_passthrough(self):
        sm = ROISmoother(beta=0.8)
        box = np.array([10, 10, 50, 90], dtype=np.float32)
        out = sm.update(1, box)
        assert np.allclose(out, box)

    def test_ema(self):
        sm = ROISmoother(beta=0.8, jump_threshold=10.0)  # 關掉跳動檢查
        sm.update(1, np.array([0, 0, 10, 10], dtype=np.float32))
        out = sm.update(1, np.array([10, 10, 20, 20], dtype=np.float32))
        # 0.8*prev + 0.2*new
        assert np.allclose(out, [2, 2, 12, 12])

    def test_jump_rejected(self):
        """框中心大幅跳動時沿用前一幀平滑框。"""
        sm = ROISmoother(beta=0.8, jump_threshold=0.5)
        first = np.array([0, 0, 10, 10], dtype=np.float32)
        sm.update(1, first)
        out = sm.update(1, np.array([100, 100, 110, 110], dtype=np.float32))
        assert np.allclose(out, first)

    def test_per_track_independent(self):
        sm = ROISmoother(beta=0.8)
        a = sm.update(1, np.array([0, 0, 10, 10], dtype=np.float32))
        b = sm.update(2, np.array([50, 50, 60, 60], dtype=np.float32))
        assert not np.allclose(a, b)

    def test_remove(self):
        sm = ROISmoother()
        sm.update(1, np.array([0, 0, 10, 10], dtype=np.float32))
        sm.remove(1)
        # 移除後視為新 track,直接採用新框
        out = sm.update(1, np.array([100, 100, 110, 110], dtype=np.float32))
        assert np.allclose(out, [100, 100, 110, 110])


class TestUpperBodyBox:
    def test_geometry(self):
        """上緣不變、高度 60%、寬高比 3:4、中心 x 錨定。"""
        box = upper_body_box(np.array([100, 200, 200, 400]),
                             aspect_ratio=0.75, upper_body_ratio=0.6)
        x1, y1, x2, y2 = box
        assert y1 == 200                      # 上緣
        assert y2 - y1 == pytest.approx(120)  # 200 高 × 0.6
        assert x2 - x1 == pytest.approx(90)   # 120 × 0.75
        assert (x1 + x2) / 2 == pytest.approx(150)  # 人物中心 x


class TestCrop:
    def test_output_size(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        roi = crop_upper_body(frame, np.array([100, 100, 300, 460]),
                              out_size=224)
        assert roi.shape == (224, 224, 3)

    def test_out_of_bounds_padded(self):
        """ROI 超出邊界時黑邊補齊,不變形。"""
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        roi = crop_roi(frame, np.array([-50, -50, 50, 50]), out_size=100)
        assert roi.shape == (100, 100, 3)
        assert roi[:49, :49].max() == 0      # 左上是補的黑邊
        assert roi[51:, 51:].min() == 255    # 右下是原圖

    def test_degenerate_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = crop_roi(frame, np.array([50, 50, 50, 40]), out_size=64)
        assert roi.shape == (64, 64, 3)
