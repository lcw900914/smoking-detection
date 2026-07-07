"""次數警戒等級與逗留偵測測試。"""
import numpy as np
import pytest

from inference.state_machine import (HandToMouthCounter, LoiterDetector,
                                     MovementGate, S2, BG)


def feed_s2_episode(counter, t0, dwell=1.0, dt=0.1):
    """餵入一段 S2 停留(t0 起持續 dwell 秒)並離開。回傳結束時間。

    離開後再推一次超過 gap_tolerance 的 BG,讓事件確定結算。
    """
    t = t0
    while t < t0 + dwell:
        counter.update(S2, t)
        t += dt
    counter.update(BG, t)
    t += counter.gap_tolerance + 0.2
    counter.update(BG, t)  # 中斷超過容忍值 → 結算事件
    return t


class TestHandToMouthCounter:
    def test_single_event_low(self):
        c = HandToMouthCounter(min_dwell=0.5, min_gap=2.0)
        feed_s2_episode(c, 0.0, dwell=1.0)
        assert c.count() == 1
        assert c.score() == pytest.approx(0.2)   # 1 次 → 低

    def test_escalation_levels(self):
        """1 次低 0.2、2 次中 0.5、3 次高 0.8。"""
        c = HandToMouthCounter(min_dwell=0.5, min_gap=2.0)
        t = feed_s2_episode(c, 0.0)
        assert c.score() == pytest.approx(0.2)
        t = feed_s2_episode(c, t + 5.0)
        assert c.score() == pytest.approx(0.5)
        t = feed_s2_episode(c, t + 5.0)
        assert c.score() == pytest.approx(0.8)
        feed_s2_episode(c, t + 5.0)              # 第 4 次仍為高
        assert c.score() == pytest.approx(0.8)

    def test_short_dwell_not_counted(self):
        c = HandToMouthCounter(min_dwell=0.5)
        feed_s2_episode(c, 0.0, dwell=0.2)       # 停留不足
        assert c.count() == 0
        assert c.score() == 0.0

    def test_min_gap_dedup(self):
        """與上次事件間隔小於 min_gap 的停留不重複計數。"""
        c = HandToMouthCounter(min_dwell=0.5, min_gap=2.0)
        t = feed_s2_episode(c, 0.0, dwell=1.0)   # 事件時間 ≈ 0.9
        # 第二段結束於 ≈2.4,距上次事件 1.5 秒 < min_gap 2 秒 → 不計
        feed_s2_episode(c, t + 0.2, dwell=0.6)
        assert c.count() == 1

    def test_gap_tolerance_merges_flicker(self):
        """短暫中斷(≤ gap_tolerance)視為同一次停留,不切碎事件。"""
        c = HandToMouthCounter(min_dwell=0.5, min_gap=2.0,
                               gap_tolerance=0.5)
        # 0.3 秒 S2 → 0.3 秒中斷 → 0.3 秒 S2:合併後 dwell 足夠
        t = 0.0
        for _ in range(3):
            c.update(S2, t); t += 0.15
        for _ in range(2):
            c.update(BG, t); t += 0.15
        for _ in range(3):
            c.update(S2, t); t += 0.15
        c.update(BG, t)
        c.update(BG, t + 0.8)                    # 超過容忍值 → 結算
        assert c.count() == 1

    def test_window_expiry(self):
        """視窗外的舊事件被移出,警戒等級回落。"""
        c = HandToMouthCounter(window_sec=30.0, min_dwell=0.5, min_gap=2.0)
        t = feed_s2_episode(c, 0.0)
        t = feed_s2_episode(c, t + 5.0)
        assert c.count() == 2
        c.update(BG, t + 60.0)                   # 60 秒後全部過期
        assert c.count() == 0

    def test_reset(self):
        c = HandToMouthCounter()
        feed_s2_episode(c, 0.0)
        c.reset()
        assert c.count() == 0


class TestLoiterDetector:
    BOX = np.array([100, 100, 200, 300], dtype=np.float32)

    def _feed(self, det, duration, wrist_visible, drift=0.0, dt=0.5):
        """餵入 duration 秒的觀測;drift 為每步的中心位移(像素)。"""
        t, box, out = 0.0, self.BOX.copy(), False
        while t <= duration:
            out = det.update(t, box, wrist_visible)
            box = box + np.array([drift, 0, drift, 0], dtype=np.float32)
            t += dt
        return out

    def test_loiter_when_static_and_no_wrist(self):
        det = LoiterDetector(min_duration=20.0, move_ratio=0.6,
                             wrist_vis_max=0.1)
        assert self._feed(det, 25.0, wrist_visible=False) is True

    def test_no_loiter_when_wrist_visible(self):
        det = LoiterDetector(min_duration=20.0)
        assert self._feed(det, 25.0, wrist_visible=True) is False

    def test_no_loiter_when_moving(self):
        det = LoiterDetector(min_duration=20.0, move_ratio=0.6)
        # 每 0.5 秒移動 5px,25 秒位移 250px > 0.6×對角線(約 224)
        assert self._feed(det, 25.0, wrist_visible=False, drift=5.0) is False

    def test_no_loiter_before_min_duration(self):
        det = LoiterDetector(min_duration=20.0)
        assert self._feed(det, 10.0, wrist_visible=False) is False

    def test_reset(self):
        det = LoiterDetector(min_duration=20.0)
        self._feed(det, 25.0, wrist_visible=False)
        det.reset()
        assert det.update(100.0, self.BOX, False) is False


class TestMovementGate:
    """移動排除:累積移動 ≥ N 倍身高 → 不視為抽菸(以人物尺寸為單位)。"""

    def _walk(self, gate, duration, step_px, height=200.0, dt=0.5):
        """模擬移動:每步中心平移 step_px 像素,框高 height。"""
        x, out = 0.0, False
        t = 0.0
        while t <= duration:
            out = gate.update(t, [x, 100, x + 80, 100 + height])
            x += step_px
            t += dt
        return out

    def test_static_person_not_excluded(self):
        gate = MovementGate(max_heights=3.0, window_sec=10.0)
        assert self._walk(gate, 10.0, step_px=0.0) is False

    def test_small_drift_not_excluded(self):
        """輕微晃動(視窗內 < 3 倍身高)不排除。"""
        gate = MovementGate(max_heights=3.0, window_sec=10.0)
        # 20 步 × 10px = 200px = 1 倍身高
        assert self._walk(gate, 10.0, step_px=10.0) is False

    def test_walking_excluded(self):
        """走動(視窗內累積 > 3 倍身高)排除。"""
        gate = MovementGate(max_heights=3.0, window_sec=10.0)
        # 20 步 × 40px = 800px = 4 倍身高(200px)
        assert self._walk(gate, 10.0, step_px=40.0) is True

    def test_scale_invariant(self):
        """同樣的「相對移動」,遠近人物(框高不同)判定一致。"""
        near = MovementGate(max_heights=3.0, window_sec=10.0)
        far = MovementGate(max_heights=3.0, window_sec=10.0)
        # 近:身高 400px、每步 80px;遠:身高 100px、每步 20px → 同為 4 倍
        assert self._walk(near, 10.0, step_px=80.0, height=400.0) is True
        assert self._walk(far, 10.0, step_px=20.0, height=100.0) is True

    def test_stops_after_standing_still(self):
        """走動後停下,視窗滑出舊軌跡 → 解除排除(可再偵測抽菸)。"""
        gate = MovementGate(max_heights=3.0, window_sec=10.0)
        assert self._walk(gate, 10.0, step_px=40.0) is True
        # 原地站 12 秒(超過視窗),舊移動全部過期
        out = True
        for i in range(24):
            out = gate.update(11.0 + i * 0.5, [800, 100, 880, 300])
        assert out is False
