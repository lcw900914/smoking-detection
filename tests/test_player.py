"""播放器的純邏輯:時間格式、標記跳轉、播放速度、進度條座標換算。

視窗本身沒辦法在測試裡跑,但「按下一個標記該跳到哪」這種事跟畫面無關,
拆出來就測得到——而那正是最容易寫錯、也最容易讓人以為播放器壞掉的部分。
"""
import pytest

from ui.player import SPEEDS, format_time, frame_delay_ms, next_marker


class TestFormatTime:
    @pytest.mark.parametrize("sec,expect", [
        (0, "0:00"), (5, "0:05"), (65, "1:05"), (599, "9:59"),
        (3600, "1:00:00"), (3725, "1:02:05"),
    ])
    def test_basic(self, sec, expect):
        assert format_time(sec) == expect

    def test_rounds_down(self):
        assert format_time(9.9) == "0:09"

    @pytest.mark.parametrize("bad", [None, -5, float("nan")])
    def test_bad_values_show_zero_not_crash(self, bad):
        """影片讀不到長度時會拿到 0/負值/NaN。時間顯示壞掉沒關係,
        但不能讓整個播放器炸掉。"""
        assert format_time(bad) == "0:00"


class TestNextMarker:
    MARKS = [5.0, 20.0, 26.5]

    def test_forward(self):
        assert next_marker(self.MARKS, 0.0) == 5.0
        assert next_marker(self.MARKS, 6.0) == 20.0

    def test_backward(self):
        assert next_marker(self.MARKS, 25.0, forward=False) == 20.0
        assert next_marker(self.MARKS, 21.0, forward=False) == 20.0

    def test_none_past_the_ends(self):
        assert next_marker(self.MARKS, 30.0) is None
        assert next_marker(self.MARKS, 1.0, forward=False) is None

    def test_empty(self):
        assert next_marker([], 5.0) is None

    def test_standing_on_a_marker_moves_on(self):
        """跳過去之後就停在標記上。沒有容差的話再按一次會原地不動,
        看起來像按鈕壞了。"""
        assert next_marker(self.MARKS, 20.0) == 26.5
        assert next_marker(self.MARKS, 20.0, forward=False) == 5.0

    def test_unsorted_input(self):
        assert next_marker([26.5, 5.0, 20.0], 6.0) == 20.0

    def test_mixed_manual_and_detected(self):
        """手動與偵測的標記合在一起排序,跳轉不分來源。"""
        assert next_marker([26.5] + [5.0, 20.0], 0.0) == 5.0


class TestFrameDelay:
    def test_normal_speed(self):
        assert frame_delay_ms(30.0, 1.0) == 33
        assert frame_delay_ms(10.0, 1.0) == 100

    def test_slow_motion_waits_longer(self):
        """0.25x 是看「舉手→停留→放下」轉折時最常用的一檔。"""
        assert frame_delay_ms(30.0, 0.25) == 133

    def test_fast_waits_less(self):
        assert frame_delay_ms(30.0, 2.0) == 16

    def test_floor_at_10ms(self):
        """再細的計時器 Tk 也跑不動,而且只會把 CPU 吃滿。"""
        assert frame_delay_ms(240.0, 2.0) == 10

    @pytest.mark.parametrize("fps", [0, None, -1])
    def test_bad_fps_falls_back(self, fps):
        """有些容器回報 fps=0。退回 25 總比除以零好。"""
        assert frame_delay_ms(fps, 1.0) == 40

    def test_every_offered_speed_is_usable(self):
        for s in SPEEDS:
            assert frame_delay_ms(30.0, s) >= 10


class TestSeekBarMapping:
    """進度條的時間↔像素換算(不需要真的畫出來)。"""

    class _Bar:
        """只借用換算邏輯,避免在測試裡開 tkinter 視窗。"""

        def __init__(self, duration, width):
            self.duration, self._w = duration, width

        def winfo_width(self):
            return self._w

        t_to_x = property(lambda self: None)

    def _mk(self, duration=100.0, width=500):
        from ui.player import SeekBar
        bar = self._Bar(duration, width)
        return (SeekBar.t_to_x.__get__(bar, SeekBar),
                SeekBar.x_to_t.__get__(bar, SeekBar))

    def test_round_trip(self):
        t_to_x, x_to_t = self._mk()
        for t in (0.0, 25.0, 50.0, 99.0):
            assert x_to_t(t_to_x(t)) == pytest.approx(t, abs=0.3)

    def test_clamped_to_the_clip(self):
        """拖到條子外面不該跳到負秒數或超過片長。"""
        _t_to_x, x_to_t = self._mk(duration=100.0, width=500)
        assert x_to_t(-50) == 0.0
        assert x_to_t(9999) == 100.0

    def test_zero_duration_does_not_divide_by_zero(self):
        t_to_x, x_to_t = self._mk(duration=0.0)
        assert t_to_x(10.0) == 0
        assert x_to_t(250) == 0.0
