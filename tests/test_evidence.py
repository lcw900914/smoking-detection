"""標記軸指向哪裡:第一次疑似抬手,而不是警報成立的時刻。

警報是「累積夠 min_events 次、且 P 持續超過門檻」的結論,成立時動作往往
已經結束十幾秒。標記若指在那裡,使用者點過去只會看到人站著——看不到自己
想複查的那口菸,會以為是誤報。

指的是**最早**的那次抬手,不是促成計入的那一次:太短沒被計入的抬手照樣
是疑似時刻,確定是抽菸之後回頭看,那才是這輪行為的開頭。
"""
from collections import deque

from inference.pipeline import TrackState, mark_evidence


def _st(raises=(), evidence=None):
    # buffer / state_machine / counter 是必填但這裡用不到:mark_evidence 只
    # 讀寫時間欄位,不碰它們。
    st = TrackState(buffer=None, state_machine=None, counter=None)
    st.raises = deque(raises)
    st.evidence_start_t = evidence
    return st


class TestMarkEvidence:
    def test_points_at_the_first_raise_not_the_counted_one(self):
        """5.0 碰一下臉(太短沒計入)、10.0 才是計入的那次——標記要指 5.0。

        這是使用者反覆指正的那一點:要標的是「疑似」的起點,不是「成立」
        的起點。
        """
        st = _st(raises=[5.0, 10.0])
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 5.0

    def test_single_raise(self):
        st = _st(raises=[10.0])
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 10.0

    def test_later_events_do_not_move_it(self):
        """第二、三口是同一輪證據,起點不該一路往後跑到警報邊上。"""
        st = _st(raises=[5.0, 10.0])
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        st.raises.append(40.0)
        mark_evidence(st, had=1, counted=True, timestamp=44.0)
        assert st.evidence_start_t == 5.0

    def test_uncounted_episodes_do_not_arm_it(self):
        """全程都沒有計入的事件 → 這輪根本不算證據,不標。

        注意這與上面不衝突:沒被計入的抬手「可以當起點」,但不能「自己
        開啟一輪」——否則每個扶眼鏡的人都會被標。
        """
        st = _st(raises=[5.0, 10.0])
        mark_evidence(st, had=0, counted=False, timestamp=13.5)
        assert st.evidence_start_t is None

    def test_falls_back_to_now_without_a_raise(self):
        """手已經舉著才進畫面:沒看到前導 S1,標在當下也好過沒有標記。"""
        st = _st(raises=[])
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 13.5

    def test_does_not_overwrite_an_existing_start(self):
        st = _st(raises=[10.0], evidence=4.0)
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 4.0

    def test_zero_timestamp_is_a_real_raise_not_missing(self):
        """影片第 0 秒就抬手是合法的,不能被當成「沒有抬手」。"""
        st = _st(raises=[0.0])
        mark_evidence(st, had=0, counted=True, timestamp=3.5)
        assert st.evidence_start_t == 0.0

    def test_takes_the_earliest_not_the_first_appended(self):
        """順序不保證:手到臉的起點是事件結算時回填的(結束 − 停留),
        可能早於已經存進去的 S1。取隊首會標到比較晚的那個。"""
        st = _st(raises=[12.0, 4.4])
        mark_evidence(st, had=0, counted=True, timestamp=15.0)
        assert st.evidence_start_t == 4.4


class TestTrackStateDefaults:
    def test_starts_clean(self):
        st = TrackState(buffer=None, state_machine=None, counter=None)
        assert list(st.raises) == []
        assert st.evidence_start_t is None

    def test_raises_are_not_shared_between_tracks(self):
        """default_factory 寫成可變預設值的話,兩個人的抬手會混在一起。"""
        a = TrackState(buffer=None, state_machine=None, counter=None)
        b = TrackState(buffer=None, state_machine=None, counter=None)
        a.raises.append(1.0)
        assert list(b.raises) == []
