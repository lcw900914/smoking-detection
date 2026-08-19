"""方法登錄表與第二階段複核的測試。

這裡守兩件事:

1. **登錄表的完整性。** 方法清單是 GUI 選單與 CLI `--method` 的唯一來源,
   一個欄位填錯(stage2 打錯字、key 重複)在介面上看起來只是「少了一個
   選項」,很難察覺。
2. **降級不否決這條紅線。** `decide()` 是整個第二階段唯一能把警報變橘色
   的地方。它必須:抽菸分數還在合理範圍就不降級、骨架看不到手時棄權
   而不是降級。這是召回不會下降的結構性保證,不能只靠註解。
"""
import numpy as np
import pytest

from inference import methods as reg
from inference import verifier as V
from tests.test_hier import FPS, make_reach_sequence


class TestRegistry:
    def test_keys_unique(self):
        assert len(reg.keys()) == len(set(reg.keys()))

    def test_names_unique(self):
        """顯示名重複的話,GUI 選了也分不出是哪一個。"""
        assert len(reg.names()) == len(set(reg.names()))

    def test_fields_are_valid(self):
        for m in reg.METHODS:
            assert m.stage1 in reg.STAGE1_MODES, m.key
            assert m.stage2 in reg.STAGE2_MODES, m.key
            assert m.name and m.desc, m.key

    def test_default_exists(self):
        assert reg.DEFAULT_KEY in reg.keys()
        assert reg.default().available, "預設方法不可以需要還沒有的權重"

    def test_get_unknown_key_lists_options(self):
        with pytest.raises(KeyError) as e:
            reg.get("no_such_method")
        assert reg.DEFAULT_KEY in str(e.value)

    def test_by_name_roundtrip(self):
        for m in reg.METHODS:
            assert reg.by_name(m.name) is m

    def test_frame_baseline_is_in_the_list(self):
        """單幀對照組必須掛在同一份清單上。

        論文的橫向比較要求同一支 GUI、同一份輸入、只換選單那一格。
        對照組另開一支腳本跑就沒有這個保證了。
        """
        assert "rule+frame_gcn" in reg.keys()

    def test_only_the_clean_ablation_is_exposed(self):
        """選單上只掛 gcn 一格。

        frame_baseline 另外實作了 mlp,離線比較還在用,但不進選單:
        它換掉的是一整組手工特徵,與時序比會多一個變因。掛上去之後
        「單幀 vs 時序」的差距就講不清楚是時間軸還是特徵造成的。
        """
        assert set(reg.FRAME_CKPT) == {"frame-gcn"}
        assert sum(1 for m in reg.METHODS if m.is_frame_baseline) == 1

    def test_frame_methods_declare_their_ckpt(self):
        for m in reg.METHODS:
            if m.is_frame_baseline:
                assert m.frame_ckpt, m.key
                assert m.frame_ckpt in reg.FRAME_CKPT.values()
            else:
                assert m.frame_ckpt is None, m.key

    def test_frame_methods_need_skeleton_and_are_not_ai_free(self):
        """對照組吃骨架、而且含學習權重——這兩個旗標錯了,GUI 會關掉
        pose 分支或把它標成「零學習權重」,兩種都是假的。"""
        for m in reg.METHODS:
            if m.is_frame_baseline:
                assert m.needs_skeleton, m.key
                assert not m.ai_free_decision, m.key
                assert not m.needs_appearance, m.key

    def test_default_is_rule_only(self):
        """預設走純規則:它不需要任何權重,換機器就能跑。"""
        assert reg.default().ai_free_decision

    def test_ai_free_decision_excludes_learned_stage2(self):
        assert reg.get("rule").ai_free_decision
        assert reg.get("rule+grammar").ai_free_decision
        assert not reg.get("rule+l1grammar").ai_free_decision
        assert not reg.get("hybrid").ai_free_decision
        assert not reg.get("net").ai_free_decision

    def test_needs_appearance(self):
        assert not reg.get("rule").needs_appearance
        assert not reg.get("rule+l1l2").needs_appearance
        assert reg.get("net").needs_appearance
        assert reg.get("hybrid").needs_appearance

    def test_ckpts_follow_stage2(self):
        assert reg.get("rule").ckpts == (None, None)
        assert reg.get("rule+grammar").ckpts == (None, None)
        assert reg.get("rule+l1grammar").ckpts == (reg.L1_CKPT, None)
        for key, mode in (("rule+l1l2", "l1+l2"),
                          ("rule+l1l2gru", "l1+l2gru")):
            assert reg.get(key).ckpts == (reg.L1_CKPT, reg.L2_CKPT[mode])

    def test_每個_l2_編碼器各有自己的權重檔(self):
        """兩格共用一個權重檔的話,訓練完 GRU 會覆蓋掉 Transformer,
        選單上還是兩格但其實是同一個模型 —— 而且看不出來。"""
        paths = list(reg.L2_CKPT.values())
        assert len(paths) == len(set(paths))
        for m in reg.METHODS:
            if m.stage2 in reg.L2_CKPT:
                assert m.ckpts[1] == reg.L2_CKPT[m.stage2], m.key
                assert m.ckpts[0] == reg.L1_CKPT, m.key

    def test_grammar_methods_never_need_weights(self):
        """無學習權重的方法必須永遠可用,不然「換機器就能跑」是假的。"""
        for m in reg.METHODS:
            if m.ai_free_decision:
                assert m.available and not m.missing(), m.key


class TestApplyConfig:
    def test_rule_forces_skeleton_on(self):
        """選純規則卻載到 skeleton.enabled=false 的設定,
        會變成沒有任何階段來源、P_t 永遠 0 的靜默失敗。"""
        cfg = {"skeleton": {"enabled": False, "model": "yolov8s-pose.pt"}}
        out = reg.get("rule").apply(cfg)
        assert out["skeleton"]["enabled"] is True
        assert out["skeleton"]["model"] == "yolov8s-pose.pt", "其餘設定要保留"

    def test_network_only_turns_skeleton_off(self):
        out = reg.get("net").apply({"skeleton": {"enabled": True}})
        assert out["skeleton"]["enabled"] is False

    def test_stage2_requires_skeleton(self):
        """stage2 吃關鍵點,所以只要有複核就一定要開骨架。"""
        for m in reg.METHODS:
            if m.stage2 is not None:
                assert m.needs_skeleton, m.key

    def test_apply_does_not_mutate_input(self):
        cfg = {"skeleton": {"enabled": False}}
        reg.get("rule").apply(cfg)
        assert cfg["skeleton"]["enabled"] is False


class TestDecide:
    """降級不否決:這一組是紅線,改動前請先確認理由已不成立。"""

    OK = {"smoking": 0.6, "other": 0.4}
    NOT_SMOKING = {"smoking": 0.05, "other": 0.9, "phone_call": 0.05}
    MARGINAL = {"smoking": 0.3, "other": 0.4, "drinking": 0.3}

    def test_top_smoking_confirms(self):
        status, _ = V.decide(self.OK, valid_ratio=0.8, span_sec=30.0)
        assert status == V.CONFIRMED

    def test_clearly_other_downgrades(self):
        status, reason = V.decide(self.NOT_SMOKING, 0.8, 30.0)
        assert status == V.REVIEW
        assert "other" in reason

    def test_marginal_smoking_is_not_downgraded(self):
        """抽菸沒拿第一但分數還在合理範圍 → 維持紅色。
        誤降一次真警報的代價遠高於多留一個橘色待複查。"""
        status, _ = V.decide(self.MARGINAL, 0.8, 30.0)
        assert status == V.CONFIRMED

    def test_invisible_skeleton_abstains_not_downgrades(self):
        """47% 的實地片段骨架不可用。若這種情況也降級,等於因為
        攝影機角度不好就把真警報壓掉。"""
        status, reason = V.decide(
            self.NOT_SMOKING, valid_ratio=0.05, span_sec=30.0)
        assert status == V.ABSTAIN
        assert "骨架" in reason

    def test_short_sequence_abstains(self):
        status, _ = V.decide(self.NOT_SMOKING, 0.9, span_sec=1.0)
        assert status == V.ABSTAIN

    def test_only_review_downgrades(self):
        for scores, valid, span in ((self.OK, 0.8, 30.0),
                                    (self.NOT_SMOKING, 0.02, 30.0),
                                    (self.NOT_SMOKING, 0.9, 0.5)):
            status, _ = V.decide(scores, valid, span)
            assert V.VerifyResult(status=status).downgraded is False

    def test_empty_scores_downgrade_is_safe(self):
        status, _ = V.decide({}, 0.9, 30.0)
        assert status == V.REVIEW      # 沒有任何抽菸證據


class TestPoseWindow:
    def test_resamples_onto_uniform_grid(self):
        """漏幀要留零,不能把序列壓縮 —— 否則停留 3 秒看起來像 1 秒。"""
        k = np.ones((17, 3), np.float32)
        hist = [(0.0, k), (0.1, k), (0.5, k)]     # 中間漏了 3 幀
        out, span = V.pose_window(hist, now=0.5, window_sec=90.0, fps=10.0)
        assert span == pytest.approx(0.5)
        assert len(out) == 6              # 0.0 ~ 0.5 秒共 6 格(含兩端)
        assert out[0].any() and out[1].any() and out[5].any()
        assert not out[2:5].any()         # 漏掉的 3 幀留零(= 棄權)

    def test_span_clamps_to_actual_history(self):
        """track 才進場 2 秒就配 90 秒的窗,會有 88 秒的零幀
        把有效比例洗掉,然後被誤判成骨架不可用。"""
        k = np.ones((17, 3), np.float32)
        hist = [(10.0 + 0.1 * i, k) for i in range(20)]
        _out, span = V.pose_window(hist, now=12.0, window_sec=90.0, fps=10.0)
        assert span == pytest.approx(2.0)

    def test_window_trims_old_frames(self):
        k = np.ones((17, 3), np.float32)
        hist = [(0.0, k), (50.0, k)]
        out, span = V.pose_window(hist, now=50.0, window_sec=10.0, fps=10.0)
        assert span == pytest.approx(10.0)
        assert len(out) == 101
        assert out[:-1].sum() == 0 and out[-1].any()  # 40 秒前那幀被切掉

    def test_empty_history(self):
        out, span = V.pose_window([], now=1.0, window_sec=90.0, fps=10.0)
        assert len(out) == 0 and span == 0.0

    def test_none_frames_are_skipped(self):
        hist = [(0.0, None), (0.1, np.ones((17, 3), np.float32))]
        out, _ = V.pose_window(hist, 0.1, 90.0, 10.0)
        assert not out[0].any() and out[1].any()


class TestSecondStageVerifier:
    def test_build_returns_none_without_stage2(self):
        assert V.build(reg.get("rule")) is None
        assert V.build(reg.get("hybrid")) is None

    def test_build_reads_config(self):
        v = V.build(reg.get("rule+grammar"),
                    {"verify": {"min_smoking": 0.4, "min_span_sec": 5.0}})
        assert v.mode == "grammar"
        assert v.min_smoking == 0.4 and v.min_span_sec == 5.0
        assert v.l1_ckpt is None and v.l2_ckpt is None

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            V.SecondStageVerifier("magic")

    def test_grammar_mode_needs_no_weights(self):
        """『純規則 + 片段文法』必須在完全沒有權重檔的機器上跑得完。"""
        v = V.SecondStageVerifier("grammar")
        kpts = make_reach_sequence(T=120, hold_frames=30)
        res = v.verify(kpts, FPS)
        assert res.source == "grammar"
        assert res.status in (V.CONFIRMED, V.REVIEW, V.ABSTAIN)
        assert res.detail and "淺層基元時間軸" in res.detail
        assert 0.0 <= res.smoking <= 1.0

    def test_hand_to_mouth_is_not_downgraded(self):
        """合成的「舉手 → 手停在嘴邊 → 放下」不該被降級 ——
        那正是抽菸一口的樣子。target 要給嘴的位置:預設落點在耳朵旁,
        文法會正確地讀成講電話。"""
        v = V.SecondStageVerifier("grammar")
        kpts = make_reach_sequence(T=160, rest_frames=20, hold_frames=20,
                                   rise_frames=8, fall_frames=8,
                                   target=(0.0, -20.0))
        res = v.verify(kpts, FPS)
        assert not res.downgraded, res.detail
        assert res.smoking > 0.0, res.detail

    def test_hand_to_ear_is_downgraded(self):
        """手貼耳久留 = 講電話,是第二階段該過濾掉的典型誤報。"""
        v = V.SecondStageVerifier("grammar")
        kpts = make_reach_sequence(T=200, rest_frames=20, hold_frames=100,
                                   target=(12.0, -26.0))
        res = v.verify(kpts, FPS)
        assert res.top == "phone_call", res.detail
        assert res.status == V.REVIEW, res.detail

    def test_empty_sequence_abstains(self):
        v = V.SecondStageVerifier("grammar")
        res = v.verify(np.zeros((0, 17, 3), np.float32), FPS)
        assert res.status == V.ABSTAIN

    def test_zero_confidence_sequence_abstains(self):
        """整段偵測不到人:必須棄權,不能拿憑空生成的判定去降級。"""
        v = V.SecondStageVerifier("grammar")
        kpts = np.zeros((120, 17, 3), np.float32)
        res = v.verify(kpts, FPS)
        assert res.status == V.ABSTAIN

    def test_status_names_cover_every_status(self):
        for s in (V.CONFIRMED, V.REVIEW, V.ABSTAIN, "pending"):
            assert s in V.STATUS_NAMES
