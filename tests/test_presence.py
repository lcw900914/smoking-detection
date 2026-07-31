"""在場型態分類測試:經過 / 徘徊 / 等待,以及距離不變性。"""
import pytest

from inference.state_machine import PresenceClassifier as PC


def walk(pc, seconds, speed_h, fps=10.0, box_h=200.0, t0=0.0, x0=100.0,
         back_and_forth=False):
    """餵入一段軌跡。

    Args:
        speed_h: 移動速度(身高/秒);0 = 原地不動
        back_and_forth: True = 原地來回踱步(位移小但累積路徑大)
    """
    n = int(seconds * fps)
    step = speed_h * box_h / fps
    state = PC.UNKNOWN
    for i in range(n):
        if back_and_forth:
            leg = (i // int(2 * fps)) % 2      # 每 2 秒換方向
            x = x0 + (step * (i % int(2 * fps)) * (1 if leg == 0 else -1))
        else:
            x = x0 + step * i
        t = t0 + i / fps
        state = pc.update(t, [x, 0.0, x + box_h * 0.4, box_h])
    return state


class TestPresence:
    def test_passing_walk(self):
        """短暫在場 + 明顯移動 → 經過。"""
        pc = PC()
        assert walk(pc, seconds=5.0, speed_h=0.8) == PC.PASSING

    def test_passing_run(self):
        """同樣短暫在場,但速度快 → 記為跑步。"""
        pc = PC()
        assert walk(pc, seconds=4.0, speed_h=2.5) == PC.RUNNING

    def test_waiting(self):
        """久留 + 幾乎不動 → 等待。"""
        pc = PC()
        assert walk(pc, seconds=30.0, speed_h=0.0) == PC.WAITING

    def test_wandering(self):
        """久留 + 一直在動 → 徘徊。"""
        pc = PC()
        assert walk(pc, seconds=30.0, speed_h=0.5) == PC.WANDERING

    def test_wandering_in_place(self):
        """原地來回踱步:位移範圍小但累積路徑大,必須判徘徊而非等待。

        這是只看位移會出錯、非看累積路徑不可的情境。
        """
        pc = PC()
        state = walk(pc, seconds=30.0, speed_h=0.6, back_and_forth=True)
        _stay, path, span, _speed = pc.stats()
        assert span < path          # 來回抵銷:位移遠小於路徑
        assert state == PC.WANDERING

    def test_unknown_when_just_appeared(self):
        """剛出現、還沒累積足夠軌跡 → 不表態。"""
        pc = PC()
        assert walk(pc, seconds=0.5, speed_h=0.0) == PC.UNKNOWN

    def test_stationary_short_stay_is_unknown(self):
        """短暫在場但沒移動(例如偵測閃一下)→ 不當成經過。"""
        pc = PC()
        assert walk(pc, seconds=5.0, speed_h=0.0) == PC.UNKNOWN

    @pytest.mark.parametrize("box_h", [80.0, 200.0, 500.0])
    def test_scale_invariance(self, box_h):
        """判定與人物離鏡頭遠近無關(量測以自身身高為單位)。"""
        pc = PC()
        assert walk(pc, seconds=30.0, speed_h=0.5, box_h=box_h) == PC.WANDERING
        pc2 = PC()
        assert walk(pc2, seconds=30.0, speed_h=0.0, box_h=box_h) == PC.WAITING

    def test_stats_units(self):
        """累積路徑以身高為單位:走 2 個身高的距離就該回報約 2.0。"""
        pc = PC()
        walk(pc, seconds=4.0, speed_h=0.5)      # 4 秒 × 0.5 身高/秒 = 2 身高
        _stay, path, _span, speed = pc.stats()
        assert path == pytest.approx(2.0, abs=0.15)
        assert speed == pytest.approx(0.5, abs=0.1)

    def test_reset(self):
        pc = PC()
        walk(pc, seconds=30.0, speed_h=0.0)
        pc.reset()
        assert walk(pc, seconds=0.5, speed_h=0.0) == PC.UNKNOWN
