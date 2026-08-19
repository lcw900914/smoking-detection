"""單幀對照組的測試。

對照組唯一的價值是「比較是公平的」,所以這裡守的全是**公平性**,
不是準確率:

1. 輸入真的只有一幀 —— 特徵版面裡不能留下任何差分量。這條壞掉的話
   對照組會悄悄變成兩幀模型,跑得出漂亮數字,而且沒有任何錯誤訊息。
2. 候選幀真的是「手舉起來」的那些 —— 手垂著的幀不該進來。
3. 段級聚合對單幀幻覺是穩的 —— 一幀爆表不該讓整段變成警報
   (腕點幻覺的教訓,見 docs/專案現況與後續計畫.md)。
4. 沒有輸入就不生成輸出 —— 一幀手都沒舉起來的段要棄權,不是猜 0.5。
5. 介面與 HierarchicalRecognizer 一致 —— verifier 換一個建構子就能用,
   降級不否決的判定邏輯兩邊完全共用。
"""
import inspect

import numpy as np
import pytest
import torch

from stage2 import frame_baseline as FB
from stage2.infer_hier import HierarchicalRecognizer
from stage2.kinematics import K_SPEED, K_V_DNOSE, K_V_H, kinematic_features
from stage2.taxonomy import DEEP_CLASSES
from tests.test_hier import FPS, make_reach_sequence


class TestFeatureLayout:
    """單幀就是單幀:任何一個差分特徵混進來,比較就作廢。"""

    def test_velocity_dims_are_excluded(self):
        for d in (K_V_DNOSE, K_V_H, K_SPEED):
            assert d not in FB.FRAME_ARM_KEEP

    def test_arm_block_has_no_global_leak(self):
        assert FB.FRAME_ARM_DIM == 16
        assert FB.frame_side_dim(use_global=False) == FB.FRAME_ARM_DIM
        assert (FB.frame_side_dim(use_global=True)
                == FB.FRAME_ARM_DIM + FB.FRAME_GLOBAL_DIM)

    def test_global_features_off_by_default(self):
        """全域特徵是捷徑(只用它們就有 AUC 0.94),預設必須關著。"""
        assert FB.USE_GLOBAL_DEFAULT is False

    def test_graph_channels_drop_velocity(self):
        """graph_features 的 (vx, vy) 是第 2、3 通道,不能留。"""
        assert FB.FRAME_GRAPH_KEEP == (0, 1, 4)
        assert FB.FRAME_GRAPH_CHANNELS == 3

    def test_frame_features_depend_only_on_that_frame(self):
        """把某一幀之外的內容全部換掉,該幀的特徵不能變。

        這是「只看一幀」最直接的檢查:真有差分量殘留,改動鄰幀就會
        讓輸出跟著動。
        """
        seq = make_reach_sequence(T=80)
        t = 45
        other = seq.copy()
        other[:t] = make_reach_sequence(T=80)[:t][::-1]
        other[t + 1:] = 0.0
        a = FB.frame_side_features(kinematic_features(seq, FPS), "R")[t]
        b = FB.frame_side_features(kinematic_features(other, FPS), "R")[t]
        assert np.allclose(a, b, atol=1e-5)

        ga = FB.frame_graph_features(seq)[t]
        gb = FB.frame_graph_features(other)[t]
        assert np.allclose(ga, gb, atol=1e-5)


class TestFrameSelection:
    def test_raised_frames_are_selected_and_resting_are_not(self):
        seq = make_reach_sequence(T=120, rest_frames=30, rise_frames=8,
                                  hold_frames=30, fall_frames=8,
                                  target=(0.0, -20.0))
        kin = kinematic_features(seq, FPS)
        mask = FB.hand_raised_mask(kin, "R")
        assert mask[40:65].mean() > 0.8, "停在臉上的幀應該全被選到"
        assert not mask[:25].any(), "手垂在身側的幀不該被選到"

    def test_invisible_wrist_is_never_selected(self):
        """沒量到手腕就沒有輸入,不該產生候選幀。"""
        seq = make_reach_sequence(T=60, target=(0.0, -20.0))
        seq[:, 10, 2] = 0.0                      # 右腕不可信
        kin = kinematic_features(seq, FPS)
        assert not FB.hand_raised_mask(kin, "R").any()

    def test_candidate_frames_covers_both_sides(self):
        kin = kinematic_features(make_reach_sequence(T=60), FPS)
        picks = FB.candidate_frames(kin)
        assert picks == sorted(picks), "候選幀必須依時間排序"
        assert all(si in (0, 1) for _, si in picks)


class TestAggregate:
    def test_single_spike_cannot_carry_a_clip(self):
        """一幀爆表 != 整段是抽菸。腕點幻覺就長這樣。"""
        n, c = 100, len(DEEP_CLASSES)
        p = np.full((n, c), 1.0 / c)
        p[7] = 0.0
        p[7, 0] = 1.0                            # 單一幀 100% 抽菸
        assert FB.aggregate(p)[0] < 0.3

    def test_sustained_evidence_wins(self):
        """三十幀持續的證據要壓倒性勝出。

        注意上限不是 1.0:每一類各取自己的 top-k 之後才正規化,其餘
        五類的 top-k 也不是 0(它們在中性幀上有 1/6 的機率),所以
        分子 1.0 會被分母 1.83 稀釋成 0.55。看的是**相對差距**。
        """
        n, c = 100, len(DEEP_CLASSES)
        p = np.full((n, c), 1.0 / c)
        p[10:40] = 0.0
        p[10:40, 0] = 1.0                        # 三十幀持續
        agg = FB.aggregate(p)
        assert agg[0] > 0.5
        assert agg[0] > 5 * agg[1:].max()

    def test_scores_sum_to_one(self):
        rng = np.random.RandomState(0)
        p = rng.dirichlet(np.ones(len(DEEP_CLASSES)), size=50)
        assert FB.aggregate(p).sum() == pytest.approx(1.0)

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            FB.aggregate(np.zeros((0, len(DEEP_CLASSES))))

    def test_abstain_gives_zero_smoking(self):
        """沒有候選幀時不可以憑空生一個中間分數出來。"""
        s = FB.abstain_scores()
        assert s["smoking"] == 0.0
        assert sum(s.values()) == pytest.approx(1.0)


class TestModels:
    @pytest.mark.parametrize("arch", FB.ARCHES)
    def test_forward_shape(self, arch):
        m = FB.build_model(arch)
        x = (torch.zeros(4, FB.mlp_input_dim()) if arch == "mlp"
             else torch.zeros(4, FB.FRAME_GRAPH_CHANNELS, 13))
        assert m(x).shape == (4, len(DEEP_CLASSES))

    def test_gcn_reuses_l1_graph(self):
        """對照組必須與 L1 共用同一份拓樸,否則變因不只一個。"""
        from stage2.graph import build_adjacency
        m = FB.build_model("gcn")
        assert np.allclose(m.gcns[0].A.numpy(), build_adjacency(True))

    def test_unknown_arch_lists_options(self):
        with pytest.raises(ValueError) as e:
            FB.build_model("transformer")
        assert "mlp" in str(e.value)


class TestDataset:
    def _items(self, n=4):
        out = []
        for i in range(n):
            seq = make_reach_sequence(T=80, target=(0.0, -20.0))
            out.append({"kpts": seq, "fps": FPS, "clip": f"c{i}",
                        "stem": f"c{i}",
                        "label": "smoking" if i % 2 else "no_contact"})
        return out

    @pytest.mark.parametrize("arch", FB.ARCHES)
    def test_samples_carry_clip_index(self, arch):
        """依段切 fold 是唯一防洩題的手段,樣本必須認得自己的段。"""
        ds = FB.FrameDataset(self._items(), arch=arch)
        assert len(ds) > 0
        picked = ds.indices_of_clips([0, 1])
        assert picked
        assert all(ds.samples[i][0] in (0, 1) for i in picked)
        a = set(ds.indices_of_clips([0]))
        b = set(ds.indices_of_clips([1]))
        assert not (a & b)

    def test_clip_without_raised_hand_is_kept_for_eval(self):
        """一幀手都沒舉起來的段:不產生訓練樣本,但仍留在 clips 裡。

        從評估裡拿掉它等於幫對照組挑掉難題(主線模型照樣會給分數)。
        """
        items = self._items(2)
        items[0]["kpts"] = items[0]["kpts"].copy()
        items[0]["kpts"][:, 9:11, 2] = 0.0       # 兩腕都不可信
        ds = FB.FrameDataset(items, arch="mlp")
        assert len(ds.clips) == 2
        assert ds.clips[0]["picks"] == []
        assert len(ds.clip_features(0)) == 0
        assert all(ds.samples[i][0] == 1 for i in range(len(ds)))

    def test_augment_is_off_by_default_and_deterministic(self):
        ds = FB.FrameDataset(self._items(2), arch="mlp")
        assert np.allclose(ds[0][0].numpy(), ds[0][0].numpy())
        aug = FB.FrameDataset(self._items(2), arch="mlp", augment=True,
                              seed=1)
        aug.resample(0)
        first = aug[0][0].numpy().copy()
        aug.resample(0)
        assert np.allclose(first, aug[0][0].numpy()), "同一 epoch 要可重現"
        aug.resample(1)
        assert not np.allclose(first, aug[0][0].numpy()), "換 epoch 要重抽"


class TestRecognizerInterface:
    """與 HierarchicalRecognizer 同介面 —— verifier 才能只換建構子。"""

    def _ckpt(self, tmp_path, arch="mlp"):
        path = tmp_path / f"frame_{arch}.pt"
        torch.save({"model": FB.build_model(arch).state_dict(),
                    "arch": arch, "classes": DEEP_CLASSES,
                    "select": {}, "use_global": False}, path)
        return str(path)

    def test_predict_signatures_match(self):
        a = inspect.signature(HierarchicalRecognizer.predict).parameters
        b = inspect.signature(FB.FrameRecognizer.predict).parameters
        assert list(a) == list(b)
        a = inspect.signature(HierarchicalRecognizer.explain).parameters
        b = inspect.signature(FB.FrameRecognizer.explain).parameters
        assert list(a) == list(b)

    @pytest.mark.parametrize("arch", FB.ARCHES)
    def test_predict_returns_same_keys(self, tmp_path, arch):
        rec = FB.FrameRecognizer(self._ckpt(tmp_path, arch), device="cpu")
        seq = make_reach_sequence(T=80, target=(0.0, -20.0))
        out = rec.predict(seq, FPS)
        for k in ("scores", "top", "source", "analysis"):
            assert k in out
        assert set(out["scores"]) == set(DEEP_CLASSES)
        assert sum(out["scores"].values()) == pytest.approx(1.0)
        assert "單幀" in rec.explain(seq, FPS)

    def test_abstains_when_no_hand_is_raised(self, tmp_path):
        rec = FB.FrameRecognizer(self._ckpt(tmp_path), device="cpu")
        seq = make_reach_sequence(T=60)
        seq[:, 9:11, 2] = 0.0
        out = rec.predict(seq, FPS)
        assert out["n_frames"] == 0
        assert out["scores"]["smoking"] == 0.0
