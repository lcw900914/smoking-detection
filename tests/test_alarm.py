"""警報邏輯測試:EMA 累積、雙門檻 hysteresis、持續時間、per-track 獨立。"""
import pytest

from inference.alarm import AlarmManager


def make_manager(**kw):
    calls = []
    mgr = AlarmManager(
        ema_alpha=kw.pop("ema_alpha", 0.5),  # 測試用較快的 EMA
        trigger_threshold=kw.pop("trigger", 0.6),
        release_threshold=kw.pop("release", 0.3),
        sustain_sec=kw.pop("sustain", 1.0),
        callback=lambda tid, P, t, fr: calls.append((tid, P, t)), **kw)
    return mgr, calls


class TestEMA:
    def test_accumulation(self):
        mgr, _ = make_manager()
        P1, _ = mgr.update(1, 1.0, 0.0)
        assert P1 == pytest.approx(0.5)      # 0.5*0 + 0.5*1
        P2, _ = mgr.update(1, 1.0, 0.1)
        assert P2 == pytest.approx(0.75)

    def test_per_track_independent(self):
        mgr, _ = make_manager()
        mgr.update(1, 1.0, 0.0)
        P2, _ = mgr.update(2, 0.0, 0.0)
        assert P2 == 0.0


class TestHysteresis:
    def test_trigger_requires_sustain(self):
        """P 超過門檻但未持續 sustain_sec 不觸發。"""
        mgr, calls = make_manager()
        _, active = mgr.update(1, 1.0, 0.0)   # P=0.5,未過 0.6
        assert not active
        _, active = mgr.update(1, 1.0, 0.5)   # P=0.75 > 0.6,開始計時
        assert not active
        _, active = mgr.update(1, 1.0, 1.0)   # 持續 0.5 秒,不足 1 秒
        assert not active
        _, active = mgr.update(1, 1.0, 1.6)   # 持續 1.1 秒 → 觸發
        assert active
        assert len(calls) == 1 and calls[0][0] == 1

    def test_release_below_low_threshold(self):
        """觸發後 P 介於兩門檻間維持警報,低於 release 才解除。"""
        mgr, _ = make_manager()
        for t in (0.0, 0.5, 1.0, 1.6):
            _, active = mgr.update(1, 1.0, t)
        assert active
        # P 掉到兩門檻之間:0.5*0.9375 + 0.5*0 ≈ 0.47 > 0.3 → 仍觸發
        _, active = mgr.update(1, 0.0, 2.0)
        assert active
        # 繼續下降至 < 0.3 → 解除
        _, active = mgr.update(1, 0.0, 2.5)
        _, active = mgr.update(1, 0.0, 3.0)
        assert not active

    def test_dip_resets_sustain_timer(self):
        """未觸發前 P 掉回門檻下,持續計時要重來。"""
        mgr, calls = make_manager()
        mgr.update(1, 1.0, 0.0)
        mgr.update(1, 1.0, 0.5)               # P=0.75,開始計時
        mgr.update(1, 0.0, 0.9)               # P=0.375 < 0.6,計時重置
        mgr.update(1, 1.0, 1.0)
        _, active = mgr.update(1, 1.0, 1.5)   # P>0.6 但持續僅 0.5 秒
        assert not active
        assert len(calls) == 0


def test_remove_resets_state():
    mgr, _ = make_manager()
    mgr.update(1, 1.0, 0.0)
    mgr.remove(1)
    P, _ = mgr.update(1, 0.0, 1.0)
    assert P == 0.0
