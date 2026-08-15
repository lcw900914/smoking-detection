"""稽核後的修正:每一項都對應更新日誌裡的一條編號。

放在一起是因為它們的共通點是「壞掉時不會出聲」——沒有例外、沒有錯誤
訊息,只是判定悄悄變成另一回事。
"""
import numpy as np
import pytest

from eval.event_eval import match_events
from inference.skeleton import SkeletonStageEstimator
from inference.state_machine import HandToMouthCounter

L_WRI, R_WRI, NOSE = 9, 10, 0


def kpts(wrist_side: int, d_px: float) -> np.ndarray:
    """造一組正面關鍵點,指定哪隻手離鼻子多遠(像素)。"""
    k = np.zeros((17, 3), np.float32)
    k[NOSE] = (100.0, 50.0, 0.9)
    # 正對鏡頭:人的右肩(kpt 6)在畫面 x 較小側,擺反會被判成背向而棄權
    k[5] = (140.0, 100.0, 0.9)     # 左肩
    k[6] = (60.0, 100.0, 0.9)      # 右肩
    k[11] = (130.0, 220.0, 0.9)    # 左髖
    k[12] = (70.0, 220.0, 0.9)     # 右髖
    far = R_WRI if wrist_side == L_WRI else L_WRI
    k[wrist_side] = (100.0, 50.0 + d_px, 0.9)
    k[far] = (100.0, 50.0 + 400.0, 0.9)
    return k


class TestA4EventGap:
    """min_gap 用停留的**起點**比,不是結束點。

    用結束點的話,差值含本次停留自身的長度(2-5 秒),實際門檻變成
    min_gap − dwell,對典型停留是負的,等於沒有門檻。
    """

    def _drag(self, c, t, dwell, down):
        for _ in range(int(dwell * 10)):
            c.update(1, t)
            t += 0.1
        out = None
        for _ in range(int(down * 10)):
            r = c.update(3, t)
            t += 0.1
            if r:
                out = r
        return t, out

    def test_rapid_reentry_is_rejected(self):
        c = HandToMouthCounter(min_dwell=2.0, max_dwell=5.0, min_gap=2.0)
        t = 0.0
        t, first = self._drag(c, t, 3.0, 0.7)
        t, second = self._drag(c, t, 3.0, 0.7)
        assert first[1] is True
        assert second[1] is False and "間隔" in second[2]

    def test_normal_rhythm_still_counts(self):
        """真正的抽菸節律(停留 3 秒、放下 5 秒)不能被誤擋。"""
        c = HandToMouthCounter(min_dwell=2.0, max_dwell=5.0, min_gap=2.0)
        t = 0.0
        for _ in range(3):
            t, r = self._drag(c, t, 3.0, 5.0)
            assert r[1] is True
        assert c.count() == 3


class TestA5WristIdentity:
    """d 取左右腕較近者,換手時 delta 會比較到兩隻不同的手。"""

    def test_no_stage_when_the_near_wrist_swaps(self):
        est = SkeletonStageEstimator(fps=10.0)
        for i in range(10):                       # 左手一直在遠處
            est.update(kpts(L_WRI, 200.0), i * 0.1)
        # 右手忽然出現在臉旁:距離「暴跌」,但那是換手不是舉手
        stage, _d, _o = est.update(kpts(R_WRI, 30.0), 1.0)
        assert stage != 0

    def test_same_wrist_approaching_still_reports_s1(self):
        est = SkeletonStageEstimator(fps=10.0)
        for i in range(10):
            est.update(kpts(L_WRI, 200.0), i * 0.1)
        est.update(kpts(L_WRI, 120.0), 1.0)
        stage, _d, _o = est.update(kpts(L_WRI, 60.0), 1.1)
        assert stage in (0, 1)


class TestA6TimeBasedLookback:
    """回看要以時間為基準——等待閘門會整段跳過 update()。"""

    def test_stale_history_is_not_compared(self):
        est = SkeletonStageEstimator(fps=10.0)
        for i in range(10):
            est.update(kpts(L_WRI, 200.0), i * 0.1)
        # 閘門擋住這個人 30 秒後才放行:不該拿 30 秒前的距離算「快速靠近」
        stage, _d, _o = est.update(kpts(L_WRI, 60.0), 31.0)
        assert stage != 0

    def test_without_timestamps_falls_back_to_index(self):
        """舊呼叫方式(不傳時間)仍須可用。"""
        est = SkeletonStageEstimator(fps=10.0)
        for _ in range(10):
            est.update(kpts(L_WRI, 200.0))
        assert est.update(kpts(L_WRI, 200.0)) is not None


class TestA7RearmOnLeavingTheFace:
    """手不必退到 near+rise_margin 那麼遠才能再武裝。

    手肘撐著、手在胸口與嘴之間小幅來回的抽菸姿勢永遠退不到那麼遠,
    原本第一口之後就再也武裝不了。
    """

    def test_moderate_retreat_rearms(self):
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5,
                                     fps=10.0)
        scale = 80.0                       # 肩寬 80px
        near_px = 0.9 * scale
        for i in range(15):                # 手退到剛好超過 near,但不到 1.4
            est.update(kpts(L_WRI, near_px * 1.15), i * 0.1)
        stage, _d, _o = est.update(kpts(L_WRI, near_px * 0.5), 1.5)
        assert stage == 1                  # 採信為 S2

    def test_hallucinated_wrist_still_blocked(self):
        """幻覺腕點恆定落在臉部區域內,走不到這條放寬路徑。"""
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5,
                                     fps=10.0)
        scale = 80.0
        for i in range(30):                # 距離一直在 near 以內
            stage, _d, _o = est.update(kpts(L_WRI, 0.9 * scale * 0.7),
                                       i * 0.1)
            assert stage != 1


class TestA8EvaluationMatching:
    """長真值 + 短警報:IoU 會把正確偵測算成漏測。"""

    GT = [{"start": 0.0, "end": 300.0}]
    PRED = [{"start": 40.0, "end": 70.0}]

    def test_overlap_counts_it_as_a_hit(self):
        m = match_events(self.PRED, self.GT)
        assert m["tp"] == 1 and m["recall"] == pytest.approx(1.0)

    def test_iou_mode_misses_it(self):
        m = match_events(self.PRED, self.GT, mode="iou")
        assert m["tp"] == 0

    def test_alarms_outside_the_truth_are_still_false_positives(self):
        m = match_events([{"start": 400.0, "end": 430.0}], self.GT)
        assert m["fp"] == 1 and m["tp"] == 0

    def test_two_alarms_in_one_truth_count_once(self):
        m = match_events([{"start": 40.0, "end": 70.0},
                          {"start": 120.0, "end": 150.0}], self.GT)
        assert m["tp"] == 1 and m["fp"] == 1
