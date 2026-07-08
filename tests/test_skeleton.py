"""骨架階段推斷測試:S1/S2/S3 規則、朝向判斷、背向棄權。"""
import numpy as np
import pytest

from inference.skeleton import (SkeletonStageEstimator, estimate_orientation,
                                NOSE, L_SHO, R_SHO, L_WRI, R_WRI,
                                L_HIP, R_HIP)

S1, S2, S3, BG = 0, 1, 2, 3


def make_kpts(wrist_dist: float, shoulder_w: float = 100.0,
              facing: str = "front") -> np.ndarray:
    """合成關鍵點:鼻在 (0,0),右腕距鼻 wrist_dist。

    COCO 左右語意:正對鏡頭時,人的左肩(kpt 5)在畫面 x 較大側;
    背對時左右顛倒,且鼻點低置信。
    """
    k = np.zeros((17, 3), dtype=np.float32)
    k[:, 2] = 0.0
    flip = -1.0 if facing == "back" else 1.0
    k[NOSE] = [0, 0, 0.15 if facing == "back" else 0.9]
    k[L_SHO] = [flip * shoulder_w / 2, 60, 0.9]
    k[R_SHO] = [-flip * shoulder_w / 2, 60, 0.9]
    k[R_WRI] = [0, wrist_dist, 0.9]
    return k


class TestOrientation:
    def test_front(self):
        assert estimate_orientation(make_kpts(200)) == "front"

    def test_back_by_shoulder_order(self):
        """肩序顛倒 → 背向(即使鼻點幻覺高置信也優先信肩序)。"""
        k = make_kpts(200, facing="back")
        assert estimate_orientation(k) == "back"
        k[NOSE, 2] = 0.9  # 幻覺鼻點
        assert estimate_orientation(k) == "back"

    def test_unknown_when_nose_weak_and_no_flip(self):
        k = make_kpts(200)
        k[NOSE, 2] = 0.3   # 低於 nose_conf 0.5
        assert estimate_orientation(k) == "unknown"

    def test_side_view_narrow_shoulders_uses_nose(self):
        """側面肩距壓縮(符號不穩)→ 不判背向,回退鼻點規則。"""
        k = make_kpts(200)
        # 肩距壓到很小且順序顛倒,但軀幹高可見 → 分離度不足,不算背向
        k[L_SHO, 0], k[R_SHO, 0] = -5, 5
        k[L_HIP] = [-5, 260, 0.9]
        k[R_HIP] = [5, 260, 0.9]
        assert estimate_orientation(k) == "front"


class TestStages:
    def test_s2_when_hand_at_mouth(self):
        est = SkeletonStageEstimator(near_ratio=0.6, fps=10)
        est.update(make_kpts(200))                  # 手在遠處 → 武裝
        stage, d, ori = est.update(make_kpts(30))   # d_norm = 0.3 < 0.6
        assert stage == S2 and ori == "front"
        assert d == pytest.approx(0.3)

    def test_background_when_hand_down(self):
        est = SkeletonStageEstimator(fps=10)
        stage, d, _ = est.update(make_kpts(200))    # d_norm = 2.0
        assert stage == BG

    def test_s1_on_approach(self):
        """手由遠到近(未達嘴部)→ S1 舉手。"""
        est = SkeletonStageEstimator(near_ratio=0.6, move_ratio=0.35, fps=10)
        for dist in np.linspace(200, 80, 10):
            stage, _, _ = est.update(make_kpts(dist))
        assert stage == S1

    def test_s3_on_leave(self):
        """手自嘴部離開下降 → S3 放下。"""
        est = SkeletonStageEstimator(near_ratio=0.6, move_ratio=0.35, fps=10)
        est.update(make_kpts(200))                  # 武裝
        for _ in range(8):
            est.update(make_kpts(40))
        for dist in np.linspace(40, 150, 8):
            stage, _, _ = est.update(make_kpts(dist))
        assert stage == S3

    def test_full_cycle_feeds_state_machine(self):
        """完整 舉手→停留→放下 序列經狀態機得滿分。"""
        from inference.state_machine import StageStateMachine
        est = SkeletonStageEstimator(near_ratio=0.6, move_ratio=0.35, fps=10)
        sm = StageStateMachine(window_sec=10.0, s2_min_dwell=0.8)
        t = 0.0
        seq = (list(np.linspace(220, 70, 8))
               + [40] * 12
               + list(np.linspace(70, 220, 8)))
        for dist in seq:
            stage, _, _ = est.update(make_kpts(dist))
            sm.push(stage, t)
            t += 0.1
        assert sm.score() == 1.0

    def test_invisible_nose_gives_background(self):
        est = SkeletonStageEstimator(fps=10)
        k = make_kpts(30)
        k[NOSE, 2] = 0.4                    # 低於 nose_conf 0.5
        stage, d, _ = est.update(k)
        assert stage == BG and d is None

    def test_none_kpts(self):
        est = SkeletonStageEstimator(fps=10)
        stage, d, ori = est.update(None)
        assert stage == BG and d is None and ori == "unknown"

    def test_reset(self):
        est = SkeletonStageEstimator(fps=10)
        for dist in np.linspace(200, 80, 10):
            est.update(make_kpts(dist))
        est.reset()
        stage, _, _ = est.update(make_kpts(100))
        assert stage == BG


class TestBackAbstain:
    """背向棄權:背對時手在頭附近也不得產生 S2(誤報主因)。"""

    def test_back_hand_near_head_no_s2(self):
        est = SkeletonStageEstimator(near_ratio=0.9, fps=10)
        k = make_kpts(30, facing="back")   # 腕在「頭」附近
        k[NOSE, 2] = 0.9                   # 即使鼻點幻覺高置信
        stage, d, ori = est.update(k)
        assert ori == "back"
        assert stage == BG and d is None

    def test_back_produces_no_counter_events(self):
        """背向連續「手到嘴樣態」不得累積次數警戒。"""
        from inference.state_machine import HandToMouthCounter
        est = SkeletonStageEstimator(near_ratio=0.9, fps=10)
        c = HandToMouthCounter(min_dwell=0.3)
        t = 0.0
        for _ in range(30):                # 3 秒背向、腕貼頭
            stage, _, _ = est.update(make_kpts(25, facing="back"))
            c.update(stage, t)
            t += 0.1
        c.update(BG, t + 1.0)
        assert c.count() == 0


class TestDistanceAdaptive:
    """距離自適應:太遠棄權、門檻隨距離放寬。"""

    def test_far_person_abstains(self):
        """身體尺度 < min_scale_px → 棄權(關鍵點雜訊佔比過大)。"""
        est = SkeletonStageEstimator(near_ratio=0.9, min_scale_px=24, fps=10)
        # 肩寬僅 16px 的遠處人物,腕貼「嘴」也不判 S2
        stage, d, _ = est.update(make_kpts(4, shoulder_w=16))
        assert stage == BG and d is None

    def test_near_threshold_relaxes_with_distance(self):
        """相同相對距離:近距離不過門檻、遠距離因誤差餘裕通過。

        d_norm = 0.95,基礎門檻 0.9:
        - 近(肩寬 200px):0.9 + 4/200 = 0.92 → 不算 S2
        - 遠(肩寬 40px): 0.9 + 4/40  = 1.00 → 算 S2
        """
        near_est = SkeletonStageEstimator(near_ratio=0.9, kpt_err_px=4.0,
                                          min_scale_px=24, fps=10)
        far_est = SkeletonStageEstimator(near_ratio=0.9, kpt_err_px=4.0,
                                         min_scale_px=24, fps=10)
        near_est.update(make_kpts(400, shoulder_w=200))   # 武裝
        far_est.update(make_kpts(80, shoulder_w=40))      # 武裝
        stage_near, _, _ = near_est.update(make_kpts(190, shoulder_w=200))
        stage_far, _, _ = far_est.update(make_kpts(38, shoulder_w=40))
        assert stage_near == BG
        assert stage_far == S2


class TestRiseArming:
    """S2「由遠而近」武裝機制:防手被遮擋時的高置信腕點幻覺。"""

    def test_hallucinated_hover_never_s2(self):
        """腕點幻覺在衣領附近(d≈0.74 恆定,從未遠離)→ 永不判 S2。

        重現實測誤報:雙手背在身後,姿態模型以 conf 0.9+ 把腕點
        放在身體輪廓上,距離恰好落在門檻內。
        """
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        for _ in range(50):                     # 5 秒恆定懸停
            stage, d, _ = est.update(make_kpts(74))   # d = 0.74
            assert stage != S2

    def test_armed_puff_detected(self):
        """真的舉手(由遠而近)→ S2 正常判定。"""
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        est.update(make_kpts(200))              # d=2.0 遠 → 武裝
        stage, _, _ = est.update(make_kpts(50))  # d=0.5
        assert stage == S2

    def test_rearm_required_between_puffs(self):
        """兩口之間手須放回遠處;只退到中距離再靠近 → 不採信。"""
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        est.update(make_kpts(200))              # 武裝
        assert est.update(make_kpts(50))[0] == S2    # 第一口
        est.update(make_kpts(110))              # 只退到 d=1.1(< 1.4 未武裝)
        assert est.update(make_kpts(50))[0] != S2    # 不採信
        est.update(make_kpts(200))              # 放回遠處 → 再武裝
        assert est.update(make_kpts(50))[0] == S2    # 第二口

    def test_ongoing_s2_survives_within_episode(self):
        """進行中的 S2 持續有效(不因 armed 消耗而中斷)。"""
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        est.update(make_kpts(200))
        for _ in range(20):                     # 停留 2 秒
            stage, _, _ = est.update(make_kpts(50))
            assert stage == S2

    def test_reset_clears_arming(self):
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        est.update(make_kpts(200))
        est.reset()
        assert est.update(make_kpts(50))[0] != S2

    def test_invisible_wrist_arms(self):
        """腕點持續不可見(手出畫面/垂下)≥0.5 秒 → 武裝。

        特寫鏡頭手放下即出畫面,量不到「遠」,以不可見代替。
        """
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        k_no_wrist = make_kpts(50)
        k_no_wrist[R_WRI, 2] = 0.1              # 腕點不可見
        for _ in range(6):                      # 0.6 秒
            est.update(k_no_wrist)
        assert est.update(make_kpts(50))[0] == S2   # 已武裝 → 可信 S2

    def test_brief_invisible_wrist_does_not_arm(self):
        """腕點只消失 1-2 幀(偵測閃爍)不武裝,幻覺懸停仍擋得住。"""
        est = SkeletonStageEstimator(near_ratio=0.9, rise_margin=0.5, fps=10)
        k_no_wrist = make_kpts(74)
        k_no_wrist[R_WRI, 2] = 0.1
        for _ in range(30):                     # 幻覺懸停,偶爾閃 1 幀
            assert est.update(make_kpts(74))[0] != S2
            est.update(k_no_wrist)              # 單幀不可見,不足以武裝


def test_alarm_allow_trigger_gate():
    """allow_trigger=False 時 EMA 照常但不觸發;恢復後可觸發。"""
    from inference.alarm import AlarmManager
    calls = []
    mgr = AlarmManager(ema_alpha=0.5, trigger_threshold=0.6,
                       release_threshold=0.3, sustain_sec=1.0,
                       callback=lambda *a: calls.append(a))
    # 背向期間:P 衝高也不觸發
    for i in range(10):
        P, active = mgr.update(1, 1.0, i * 0.5, allow_trigger=False)
    assert P > 0.6 and not active and not calls
    # 轉正面:需重新累積 sustain 才觸發
    _, active = mgr.update(1, 1.0, 5.0, allow_trigger=True)
    assert not active
    _, active = mgr.update(1, 1.0, 6.2, allow_trigger=True)
    assert active and len(calls) == 1
