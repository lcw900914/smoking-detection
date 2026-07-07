"""階段狀態機測試:S1→S2→S3 順序、S2 dwell、視窗滑動。"""
import pytest

from inference.state_machine import StageStateMachine, cycle_score, S1, S2, S3, BG


def feed(sm, seq, dt=0.2, t0=0.0):
    """依序餵入階段序列,每步間隔 dt 秒。"""
    for i, s in enumerate(seq):
        sm.push(s, t0 + i * dt)


class TestStateMachine:
    def test_full_sequence(self):
        """完整 S1→S2(dwell 足)→S3:1.0。"""
        sm = StageStateMachine(window_sec=8.0, s2_min_dwell=0.8)
        feed(sm, [S1] * 3 + [S2] * 6 + [S3] * 3)  # S2 持續 1.0 秒
        assert sm.score() == 1.0

    def test_s2_dwell_too_short(self):
        """S2 出現但 dwell 不足:0.3。"""
        sm = StageStateMachine(window_sec=8.0, s2_min_dwell=0.8)
        feed(sm, [S1] * 3 + [S2] * 2 + [S3] * 3)  # S2 僅 0.2 秒
        assert sm.score() == pytest.approx(0.3)

    def test_missing_s1(self):
        """缺 S1(順序不完整)但 S2 dwell 足:打折為 0.6。"""
        sm = StageStateMachine(window_sec=8.0, s2_min_dwell=0.8)
        feed(sm, [BG] * 3 + [S2] * 6 + [S3] * 3)
        assert sm.score() == pytest.approx(0.6)

    def test_wrong_order(self):
        """亂序(S3→S2→S1)不得滿分。"""
        sm = StageStateMachine(window_sec=8.0, s2_min_dwell=0.8)
        feed(sm, [S3] * 3 + [S2] * 6 + [S1] * 3)
        assert sm.score() < 1.0

    def test_only_background(self):
        sm = StageStateMachine()
        feed(sm, [BG] * 10)
        assert sm.score() == 0.0

    def test_only_s1(self):
        sm = StageStateMachine()
        feed(sm, [S1] * 5)
        assert sm.score() == pytest.approx(0.15)

    def test_window_expiry(self):
        """視窗外的舊紀錄要被移出(S1 過期後不再滿分)。"""
        sm = StageStateMachine(window_sec=2.0, s2_min_dwell=0.4)
        feed(sm, [S1] * 3, dt=0.2, t0=0.0)
        feed(sm, [S2] * 4, dt=0.2, t0=10.0)   # 10 秒後,S1 已過期
        feed(sm, [S3] * 2, dt=0.2, t0=10.8)
        assert sm.score() < 1.0

    def test_reset(self):
        sm = StageStateMachine()
        feed(sm, [S1, S2, S3])
        sm.reset()
        assert sm.score() == 0.0


def test_cycle_score_weighting():
    assert cycle_score(1.0, 0.0, 0.5, 0.5) == 0.5
    assert cycle_score(0.8, 0.6, 0.5, 0.5) == pytest.approx(0.7)
