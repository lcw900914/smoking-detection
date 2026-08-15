"""累積路徑:抖動不該被當成移動,取樣率不該改變判定。

這兩件事先前都不成立,而且是靜默的——一個站著不動的人被判成「徘徊」,
配上「只有等待才做抽菸分析」的閘門,就是完全不會被偵測,畫面上也不會
有任何異常。
"""
import numpy as np
import pytest

from inference.state_machine import PresenceClassifier, accumulated_path
from tracking.roi import ROISmoother

H = 200.0          # 框高(像素)


def simulate(sigma=0.0, fps=10.0, seconds=60.0, speed_h=0.0, pace=None,
             seed=0):
    """回傳 (累積路徑, 位移範圍, 最終判定)。"""
    rng = np.random.default_rng(seed)
    sm = ROISmoother(beta=0.8)
    pc = PresenceClassifier()
    state = PresenceClassifier.UNKNOWN
    for i in range(int(seconds * fps)):
        t = i / fps
        cx = 320.0 + speed_h * H * t
        if pace:
            cx += pace[0] * H * np.sin(2 * np.pi * t / pace[1])
        box = np.array([cx - 50 + rng.normal(0, sigma),
                        240 - H / 2 + rng.normal(0, sigma),
                        cx + 50 + rng.normal(0, sigma),
                        240 + H / 2 + rng.normal(0, sigma)], np.float32)
        state = pc.update(t, sm.update(1, box))
    _stay, path, span, _speed = pc.stats()
    return path, span, state


class TestJitterIsNotMovement:
    @pytest.mark.parametrize("sigma", [1.0, 3.0, 5.0, 8.0])
    def test_stationary_person_stays_waiting(self, sigma):
        """先前 σ=5px 就能累到 3.85 身高,超過徘徊門檻 3.0。"""
        path, _span, state = simulate(sigma=sigma)
        assert path < 1.0, f"抖動 σ={sigma}px 累出 {path:.2f} 身高"
        assert state == PresenceClassifier.WAITING


class TestSamplingRateDoesNotChangeTheVerdict:
    @pytest.mark.parametrize("fps", [5, 10, 20, 30])
    def test_stationary(self, fps):
        """target_fps 是效能旋鈕,不該決定一個人算不算在徘徊。

        先前 5→30 fps 會讓累積路徑從 1.18 變 7.12,判定跟著翻面。
        """
        _path, _span, state = simulate(sigma=3.0, fps=fps)
        assert state == PresenceClassifier.WAITING

    def test_walking_is_consistent_across_rates(self):
        paths = [simulate(sigma=3.0, fps=f, speed_h=0.3)[0]
                 for f in (5, 10, 20, 30)]
        assert min(paths) > 3.0                      # 每個取樣率都判得出走動
        assert max(paths) / min(paths) < 1.6         # 而且量值相近


class TestRealMotionStillDetected:
    def test_walking(self):
        _p, _s, state = simulate(sigma=3.0, speed_h=0.5)
        assert state == PresenceClassifier.WANDERING

    def test_pacing_in_place(self):
        """原地來回踱步:位移範圍很小、累積路徑很大——這正是徘徊。

        區分這兩者是這個分類器存在的理由,不能被去雜訊順手做掉。
        """
        path, span, state = simulate(sigma=3.0, pace=(0.5, 8.0))
        assert span < 1.5 and path > 5.0
        assert state == PresenceClassifier.WANDERING


class TestAccumulatedPath:
    def test_deadband_drops_small_steps(self):
        pts = [(i * 0.5, 100.0 + i * 0.5, 100.0, H) for i in range(20)]
        assert accumulated_path(pts, H, dt=0.5, deadband=0.05) == 0.0

    def test_real_steps_are_kept(self):
        pts = [(i * 0.5, 100.0 + i * 40.0, 100.0, H) for i in range(5)]
        assert accumulated_path(pts, H, dt=0.5, deadband=0.05) == \
            pytest.approx(0.8, abs=0.01)

    def test_tail_shorter_than_dt_is_included(self):
        """不補尾段的話 4 秒的路只算得到 3.5 秒,短視窗誤差可達 12%。"""
        pts = [(i * 0.1, 100.0 + i * 10.0, 100.0, H) for i in range(41)]
        assert accumulated_path(pts, H, dt=0.5, deadband=0.01) == \
            pytest.approx(2.0, abs=0.01)

    def test_too_few_points(self):
        assert accumulated_path([(0.0, 1.0, 1.0, H)], H) == 0.0
