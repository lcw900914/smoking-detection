"""標記軸指向哪裡:證據起點,而不是警報成立的時刻。

警報是「累積夠 min_events 次、且 P 持續超過門檻」的結論,成立時動作往往
已經結束十幾秒。標記若指在那裡,使用者點過去只會看到人站著——看不到自己
想複查的那口菸,會以為是誤報。
"""
from inference.pipeline import TrackState, mark_evidence


def _st(last_raise=None, evidence=None):
    # buffer / state_machine / counter 是必填但這裡用不到:mark_evidence 只
    # 讀寫兩個時間欄位,不碰它們。
    st = TrackState(buffer=None, state_machine=None, counter=None)
    st.last_raise_t = last_raise
    st.evidence_start_t = evidence
    return st


class TestMarkEvidence:
    def test_points_at_the_raise_not_the_count(self):
        """抬手在 10.0,手放下計入在 13.5——標記要指 10.0。"""
        st = _st(last_raise=10.0)
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 10.0

    def test_later_events_do_not_move_it(self):
        """第二、三口是同一輪證據,起點不該一路往後跑到警報邊上。"""
        st = _st(last_raise=10.0)
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        st.last_raise_t = 40.0
        mark_evidence(st, had=1, counted=True, timestamp=44.0)
        assert st.evidence_start_t == 10.0

    def test_uncounted_episodes_do_not_arm_it(self):
        """扶眼鏡、講電話都會結算成事件但不計入,不算證據。"""
        st = _st(last_raise=10.0)
        mark_evidence(st, had=0, counted=False, timestamp=13.5)
        assert st.evidence_start_t is None

    def test_falls_back_to_now_without_a_raise(self):
        """手已經舉著才進畫面:沒有前導 S1,標在當下也好過沒有標記。"""
        st = _st(last_raise=None)
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 13.5

    def test_does_not_overwrite_an_existing_start(self):
        st = _st(last_raise=10.0, evidence=4.0)
        mark_evidence(st, had=0, counted=True, timestamp=13.5)
        assert st.evidence_start_t == 4.0

    def test_zero_timestamp_is_a_real_raise_not_missing(self):
        """影片第 0 秒就抬手是合法的,不能被當成「沒有抬手」。"""
        st = _st(last_raise=0.0)
        mark_evidence(st, had=0, counted=True, timestamp=3.5)
        assert st.evidence_start_t == 0.0


class TestTrackStateDefaults:
    def test_starts_clean(self):
        st = TrackState(buffer=None, state_machine=None, counter=None)
        assert st.last_raise_t is None
        assert st.evidence_start_t is None
