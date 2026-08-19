"""rule+order 的「必須看得到放下」規則。

守三件事:
1. 預設關閉 —— 這是待驗證的變體,不是新的預設行為。
2. 事件的時間戳用「停留結束」而不是「確認放下」——不然 min_gap 與
   90 秒滾動視窗會被 release_window 拖著跑,兩個門檻的語意就變了。
3. release_window 必須大於 gap_tolerance。estimator 判 S3 要跟 0.6 秒前
   比較,而結算只等 gap_tolerance(0.5 秒);窗口開太小的話 S3 還沒被
   判出來事件就被丟掉,等於把**所有**事件都擋掉——而且不會有錯誤訊息。
"""
import pytest

from inference import methods as reg
from inference.state_machine import S2, S3, BG, HandToMouthCounter

FPS = 10.0


def feed(counter, stages, t0=0.0, fps=FPS):
    """把 [(階段, 幀數), ...] 餵進計數器,回傳所有結算結果。"""
    out, t = [], t0
    for stage, n in stages:
        for _ in range(int(n)):
            r = counter.update(stage, t)
            if r is not None:
                out.append(r)
            t += 1.0 / fps
    return out


class TestDefaultOff:
    def test_off_by_default(self):
        assert HandToMouthCounter().require_release is False

    def test_only_rule_order_turns_it_on(self):
        on = [m.key for m in reg.METHODS if m.require_release]
        assert on == ["rule+order"]

    def test_apply_injects_the_flag(self):
        """GUI 與 CLI 都靠 Method.apply() 把旗標塞進設定,漏了就是靜默失效。"""
        cfg = {"escalation": {"min_dwell": 2.0}}
        assert reg.get("rule+order").apply(cfg)["escalation"]["require_release"]
        assert not reg.get("rule").apply(cfg)["escalation"]["require_release"]
        assert cfg["escalation"] == {"min_dwell": 2.0}, "不可改動傳入的 cfg"


class TestBehaviour:
    def _counter(self, **kw):
        return HandToMouthCounter(min_dwell=2.0, max_dwell=5.0, min_gap=2.0,
                                  gap_tolerance=0.5, **kw)

    def test_stay_then_release_counts(self):
        c = self._counter(require_release=True)
        got = feed(c, [(S2, 30), (BG, 8), (S3, 3), (BG, 10)])
        assert [g[1] for g in got] == [True]
        assert c.count() == 1

    def test_stay_then_vanish_does_not_count(self):
        """手到嘴之後手直接不見了 —— 這正是要擋掉的情況。"""
        c = self._counter(require_release=True)
        got = feed(c, [(S2, 30), (BG, 40)])
        assert [g[1] for g in got] == [False]
        assert "沒看到放下" in got[0][2]
        assert c.count() == 0

    def test_same_sequence_counts_without_the_flag(self):
        """同一段輸入在預設設定下會計入 —— 差別確實只有這一條規則。"""
        c = self._counter(require_release=False)
        got = feed(c, [(S2, 30), (BG, 40)])
        assert [g[1] for g in got] == [True]

    def test_event_timestamp_is_the_stay_end(self):
        """兩次停留間隔 2.1 秒應該都計入;若時間戳改用確認時刻,
        第二次會被 min_gap 誤擋。"""
        c = self._counter(require_release=True)
        got = feed(c, [(S2, 30), (BG, 6), (S3, 3), (BG, 12),
                       (S2, 30), (BG, 6), (S3, 3), (BG, 10)])
        assert [g[1] for g in got] == [True, True], got

    def test_release_window_must_exceed_gap_tolerance(self):
        """窗口比容忍值還小 → 連正常的放下都來不及被看到。

        這個測試是留給未來調參數的人的警告,不是在測一個好設定。
        """
        c = self._counter(require_release=True, release_window=0.2)
        got = feed(c, [(S2, 30), (BG, 8), (S3, 3), (BG, 10)])
        assert [g[1] for g in got] == [False]

    def test_short_and_long_stays_still_rejected_first(self):
        """既有的三個否決條件優先,不會變成「等放下」再說。"""
        c = self._counter(require_release=True)
        short = feed(c, [(S2, 5), (BG, 10), (S3, 3), (BG, 10)])
        assert short and short[0][1] is False and "太短" in short[0][2]
        c.reset()
        long_ = feed(c, [(S2, 70), (BG, 10), (S3, 3), (BG, 10)])
        assert long_ and long_[0][1] is False and "太長" in long_[0][2]

    def test_reset_clears_pending(self):
        c = self._counter(require_release=True)
        feed(c, [(S2, 30), (BG, 8)])
        assert c._pending is not None
        c.reset()
        assert c._pending is None
