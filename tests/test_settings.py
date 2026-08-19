"""方法參數的覆寫:套用、夾範圍、各方法互不干擾、壞檔處理。

這些值會被寫進使用者的檔案再讀回來餵給管線,所以「手改壞了會怎樣」跟
「正常情況對不對」一樣重要——帶著負的停留秒數跑起來,判定會安靜地變成
另一回事。
"""
import json

from inference import methods as reg
from ui.settings import (all_params, apply_overrides, defaults_for,
                         diff_from, get_in, load_overrides, save_overrides,
                         set_in)

CFG = {
    "presence": {"smoking_requires_waiting": True,
                 "long_stay": 20.0, "wander_path": 3.0, "short_stay": 8.0,
                 "pass_path": 1.0, "run_speed": 1.5, "window_sec": 60.0},
    "escalation": {"min_dwell": 2.0, "max_dwell": 5.0, "min_gap": 2.0,
                   "window_sec": 90.0},
    "alarm": {"min_events": 2, "trigger_threshold": 0.6,
              "release_threshold": 0.3, "sustain_sec": 2.0,
              "clip_overlay": False, "clip_pre_sec": 10.0},
    "skeleton": {"near_ratio": 0.9, "move_ratio": 0.35, "min_scale_px": 24,
                 "rise_margin": 0.5},
    "move_gate": {"enabled": True, "max_heights": 3.0,
                  "window_sec": 10.0},
    "fusion": {"count": 0.4, "network": 0.6},
    "verify": {"min_smoking": 0.25, "min_valid_ratio": 0.15,
               "min_span_sec": 3.0, "window_sec": 90.0},
}


class TestParamTable:
    def test_keys_are_unique(self):
        keys = [p.key for p in all_params()]
        assert len(keys) == len(set(keys))

    def test_every_param_exists_in_the_config(self):
        """參數表指到設定檔裡沒有的路徑,對話框就會少一項而且沒人發現。"""
        for p in all_params():
            assert get_in(CFG, p.path) is not None, p.key

    def test_ranges_are_sane(self):
        for p in all_params():
            assert p.lo < p.hi, p.key
            assert p.step > 0, p.key

    def test_defaults_sit_inside_the_range(self):
        """預設值落在可調範圍外的話,一開對話框就等於被偷改了。"""
        for p in all_params():
            v = float(get_in(CFG, p.path))
            assert p.lo <= v <= p.hi, f"{p.key}={v} 不在 [{p.lo},{p.hi}]"


class TestFilteringByMethod:
    def test_pure_rule_hides_fusion_weights(self):
        """純規則沒有網路可融合,列出來只會讓人以為調了有用。"""
        keys = defaults_for(CFG, reg.get("rule"))
        assert "fusion.network" not in keys
        assert "fusion.count" not in keys

    def test_pure_rule_hides_verify(self):
        assert "verify.min_smoking" not in defaults_for(CFG, reg.get("rule"))

    def test_stage2_method_shows_verify(self):
        assert "verify.min_smoking" in defaults_for(CFG,
                                                    reg.get("rule+grammar"))

    def test_appearance_method_shows_weights(self):
        keys = defaults_for(CFG, reg.get("hybrid"))
        assert "fusion.network" in keys
        assert "fusion.count" in keys

    def test_fusion_weights_are_not_under_state_machine(self):
        """融合權重與 StageStateMachine 無關 —— 它的 score() 根本沒被讀。

        舊版把它們放在 state_machine.weights 底下、第一項還叫
        「規則權重」,讓人以為調的是順序檢查的權重;實際上乘的一直是
        HandToMouthCounter 的次數分數。放回原處就是把這個誤導種回去。
        """
        assert not any(p.key.startswith("state_machine.weights")
                       for p in all_params())

    def test_network_only_hides_skeleton_rules(self):
        """純外觀網路不看骨架,near_ratio 之類調了不會有作用。"""
        assert "skeleton.near_ratio" not in defaults_for(CFG, reg.get("net"))

    def test_presence_shows_for_every_method(self):
        for m in reg.METHODS:
            assert "presence.long_stay" in defaults_for(CFG, m), m.key


class TestApplyOverrides:
    def test_applies_value(self):
        out = apply_overrides(CFG, {"presence.long_stay": 35})
        assert out["presence"]["long_stay"] == 35

    def test_does_not_mutate_the_source(self):
        apply_overrides(CFG, {"presence.long_stay": 35})
        assert CFG["presence"]["long_stay"] == 20.0

    def test_nested_path(self):
        out = apply_overrides(CFG, {"fusion.network": 0.9})
        assert out["fusion"]["network"] == 0.9
        assert out["fusion"]["count"] == 0.4

    def test_clamps_out_of_range(self):
        """覆寫檔是純文字,手改壞了不該讓管線帶著荒謬的值跑起來。"""
        assert apply_overrides(CFG, {"presence.long_stay": 9999}
                               )["presence"]["long_stay"] == 120
        assert apply_overrides(CFG, {"presence.long_stay": -5}
                               )["presence"]["long_stay"] == 5

    def test_unknown_keys_ignored(self):
        out = apply_overrides(CFG, {"沒有這個參數": 1, "presence.long_stay": 30})
        assert out["presence"]["long_stay"] == 30

    def test_non_numeric_ignored(self):
        out = apply_overrides(CFG, {"presence.long_stay": "abc"})
        assert out["presence"]["long_stay"] == 20.0

    def test_integer_params_stay_integers(self):
        v = apply_overrides(CFG, {"alarm.min_events": 3.7})["alarm"]["min_events"]
        assert v == 4 and isinstance(v, int)

    def test_empty_overrides_is_a_plain_copy(self):
        assert apply_overrides(CFG, {}) == CFG
        assert apply_overrides(CFG, None) == CFG


class TestPersistence:
    def test_round_trip_per_method(self, tmp_path):
        """各方法各記各的:調鬆 rule 不該連帶改到 hybrid。"""
        f = tmp_path / "ov.json"
        save_overrides({"rule": {"presence.long_stay": 35},
                        "hybrid": {"alarm.min_events": 4}}, f)
        got = load_overrides(f)
        assert got["rule"]["presence.long_stay"] == 35
        assert got["hybrid"]["alarm.min_events"] == 4
        assert "presence.long_stay" not in got["hybrid"]

    def test_missing_file_is_empty_not_an_error(self, tmp_path):
        assert load_overrides(tmp_path / "還沒存過.json") == {}

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path):
        f = tmp_path / "ov.json"
        f.write_text("{壞掉的 json", encoding="utf-8")
        assert load_overrides(f) == {}

    def test_non_dict_json_rejected(self, tmp_path):
        f = tmp_path / "ov.json"
        f.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert load_overrides(f) == {}

    def test_save_creates_the_folder(self, tmp_path):
        f = tmp_path / "深" / "一層" / "ov.json"
        assert save_overrides({"rule": {}}, f) is True
        assert f.exists()


class TestDiff:
    def test_lists_only_changed(self):
        d = diff_from(CFG, {"presence.long_stay": 35,
                            "presence.wander_path": 3.0})
        assert d == ["presence.long_stay"]

    def test_empty_when_same(self):
        assert diff_from(CFG, {"alarm.min_events": 2}) == []

    def test_ignores_unknown_keys(self):
        assert diff_from(CFG, {"亂寫": 999}) == []


class TestPathHelpers:
    def test_get_missing_returns_default(self):
        assert get_in(CFG, ("沒有", "這個")) is None
        assert get_in(CFG, ("沒有",), "預設") == "預設"

    def test_set_creates_missing_levels(self):
        d = {}
        set_in(d, ("a", "b", "c"), 1)
        assert d == {"a": {"b": {"c": 1}}}


class TestBooleanParam:
    """錄影疊加是布林值:關 = 乾淨影像(訓練外觀模型的前提)。"""

    def test_stays_boolean_after_apply(self):
        out = apply_overrides(CFG, {"alarm.clip_overlay": 1})
        assert out["alarm"]["clip_overlay"] is True

    def test_zero_means_off(self):
        out = apply_overrides(CFG, {"alarm.clip_overlay": 0})
        assert out["alarm"]["clip_overlay"] is False

    def test_shows_for_every_method(self):
        for m in reg.METHODS:
            assert "alarm.clip_overlay" in defaults_for(CFG, m), m.key


class TestMoveGateSwitch:
    """移動排除決定「經過/徘徊的人會不會被判抽菸」,一定要關得掉。

    先前主畫面有這個勾選框,改版時被移除卻沒補進參數表——行為沒變,
    但使用者失去了唯一的開關,看起來就像規則壞掉。
    """

    def test_is_adjustable(self):
        assert "move_gate.enabled" in defaults_for(CFG, reg.get("rule"))

    def test_can_be_turned_off(self):
        out = apply_overrides(CFG, {"move_gate.enabled": 0})
        assert out["move_gate"]["enabled"] is False

    def test_shows_for_every_method(self):
        for m in reg.METHODS:
            assert "move_gate.enabled" in defaults_for(CFG, m), m.key


class TestWaitingGate:
    """只有「等待」才判抽菸——使用者反覆強調的規則,必須看得到也關得掉。"""

    def test_is_adjustable(self):
        assert "presence.smoking_requires_waiting" in defaults_for(
            CFG, reg.get("rule"))

    def test_default_is_on(self):
        assert CFG["presence"]["smoking_requires_waiting"] is True

    def test_can_be_turned_off(self):
        out = apply_overrides(CFG, {"presence.smoking_requires_waiting": 0})
        assert out["presence"]["smoking_requires_waiting"] is False
