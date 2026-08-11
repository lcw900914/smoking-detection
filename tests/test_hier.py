"""兩層骨架時序模型的測試:拓樸、不變性、基元語意、模型形狀。

合成序列刻意做成「看得懂的動作」——手從腿邊舉到臉、停一下、放下——
這樣測試失敗時能直接讀出是哪一段語意壞了,而不是只知道某個張量形狀
對不上。
"""
import numpy as np
import torch

from stage2.composition import (STAT_DIM, S_D_MIN, S_HOLD_MEAN, S_N_CYCLE,
                                analyze, grammar_scores, normalize_stats)
from stage2.graph import (ANATOMICAL_EDGES, FUNCTIONAL_EDGES, L_WRI,
                          NOSE, NUM_NODES, build_adjacency, hop_distance)
from stage2.hier_model import (CompositionNet, PrimitiveNet, TOKEN_DIM,
                               count_params)
from stage2.kinematics import (GRAPH_CHANNELS, KIN_DIM, K_COS_ELBOW,
                               K_D_NOSE, K_FACE_OK, K_H_WRI, K_VALID,
                               SIDE_SLICE, graph_features,
                               kinematic_features, side_view)
from stage2.primitives import (P_HOLD, P_RAISE, P_REST, PRIMITIVES,
                               SEG_ATTR_DIM, find_cycles, rule_primitives,
                               segments_from_kinematics)
from stage2.taxonomy import DEEP_CLASSES, deep_index

FPS = 10.0


def make_reach_sequence(T=120, cx=300.0, cy=200.0, scale=1.0, conf=0.9,
                        rest_frames=30, rise_frames=8, hold_frames=30,
                        fall_frames=8, rot_deg=0.0):
    """合成「手垂著 → 舉到臉 → 停留 → 放下 → 垂著」的序列。

    版面(未旋轉、未縮放時,單位為像素):
        鼻在肩線上方 30,雙肩各距中心 20,髖在下方 60 → 身體尺度 40。
        右腕由 (60, 60)(垂在身側,腕-鼻距離約 2.7)移到鼻旁(約 0.3)。

    舉手預設只花 8 幀(0.8 秒)——這是真人的速度,也必須是:規則把
    「腕-鼻距離每秒縮小 0.85 以上」才算舉手,慢動作的合成序列會被
    正確地判成 free,測不到 raise。
    """
    seq = np.zeros((T, 17, 3), np.float32)
    seq[:, :, 2] = conf
    base = {0: (0, -30), 1: (-6, -34), 2: (6, -34), 3: (-14, -30),
            4: (14, -30), 5: (-20, 0), 6: (20, 0), 7: (-34, 30),
            8: (34, 30), 9: (-60, 60), 10: (60, 60),
            11: (-15, 60), 12: (15, 60), 13: (-15, 110), 14: (15, 110),
            15: (-15, 160), 16: (15, 160)}
    for j, (dx, dy) in base.items():
        seq[:, j, 0] = dx
        seq[:, j, 1] = dy

    start = np.array([60.0, 60.0])
    target = np.array([12.0, -26.0])           # 鼻子右下方一點點
    t_rise, t_hold, t_fall = (rest_frames, rest_frames + rise_frames,
                              rest_frames + rise_frames + hold_frames)
    for t in range(T):
        if t < t_rise:
            a = 0.0
        elif t < t_hold:
            a = (t - t_rise + 1) / max(rise_frames, 1)
        elif t < t_fall:
            a = 1.0
        elif t < t_fall + fall_frames:
            a = 1.0 - (t - t_fall + 1) / max(fall_frames, 1)
        else:
            a = 0.0
        a = float(np.clip(a, 0.0, 1.0))
        seq[t, 10, :2] = start + a * (target - start)
        seq[t, 8, :2] = 0.55 * seq[t, 10, :2]   # 肘跟著走,維持手臂鏈

    r = np.deg2rad(rot_deg)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    seq[:, :, :2] = seq[:, :, :2] @ rot.T * scale + np.array([cx, cy])
    return seq


class TestGraph:
    def test_adjacency_shape_and_normalisation(self):
        A = build_adjacency()
        assert A.shape == (3, NUM_NODES, NUM_NODES)
        rows = A.sum(axis=2)
        nonzero = rows > 0
        assert np.allclose(rows[nonzero], 1.0, atol=1e-5), "分區未做度正規化"

    def test_partitions_are_disjoint(self):
        """同一條邊只能落在一個分區,否則等於重複計算。"""
        A = build_adjacency()
        occupied = (A > 0).sum(axis=0)
        assert occupied.max() <= 1

    def test_hop_distance_from_shoulders(self):
        hop = hop_distance()
        assert hop[5] == 0 and hop[6] == 0        # 雙肩是中心
        assert hop[L_WRI] == 2                    # 肩→肘→腕
        assert np.isfinite(hop).all(), "有節點與軀幹不連通"

    def test_functional_edges_shorten_wrist_to_nose(self):
        """功能邊的重點:腕到鼻從繞路變成一步。"""
        with_fn = build_adjacency(include_functional=True)
        without = build_adjacency(include_functional=False)
        assert with_fn[:, L_WRI, NOSE].sum() > 0
        assert without[:, L_WRI, NOSE].sum() == 0

    def test_no_duplicate_edges(self):
        both = ANATOMICAL_EDGES + FUNCTIONAL_EDGES
        norm = {tuple(sorted(e)) for e in both}
        assert len(norm) == len(both)


class TestKinematics:
    def test_shapes(self):
        seq = make_reach_sequence()
        assert graph_features(seq).shape == (len(seq), NUM_NODES,
                                             GRAPH_CHANNELS)
        assert kinematic_features(seq, FPS).shape == (len(seq), KIN_DIM)

    def test_translation_invariance(self):
        a = kinematic_features(make_reach_sequence(cx=200, cy=150), FPS)
        b = kinematic_features(make_reach_sequence(cx=900, cy=500), FPS)
        assert np.allclose(a, b, atol=1e-4)

    def test_scale_invariance(self):
        """人走遠一半,所有角度與正規化距離都不該變。"""
        a = kinematic_features(make_reach_sequence(scale=1.0), FPS)
        b = kinematic_features(make_reach_sequence(scale=2.5), FPS)
        cols = [K_D_NOSE, K_H_WRI, K_COS_ELBOW]
        for c in cols:
            assert np.allclose(a[:, SIDE_SLICE["R"]][:, c],
                               b[:, SIDE_SLICE["R"]][:, c], atol=1e-3)

    def test_rotation_invariance_of_body_frame(self):
        """相機滾轉 / 人側傾:身體座標系裡的量不變。"""
        a = kinematic_features(make_reach_sequence(rot_deg=0), FPS)
        b = kinematic_features(make_reach_sequence(rot_deg=25), FPS)
        for c in (K_D_NOSE, K_H_WRI, K_COS_ELBOW):
            assert np.allclose(a[:, SIDE_SLICE["R"]][:, c],
                               b[:, SIDE_SLICE["R"]][:, c], atol=2e-2)

    def test_wrist_approaches_nose(self):
        """語意檢查:序列中段的腕-鼻距離必須明顯小於開頭。"""
        kin = kinematic_features(make_reach_sequence(), FPS)
        d = kin[:, SIDE_SLICE["R"]][:, K_D_NOSE]
        assert d[len(d) // 2] < d[0] * 0.4

    def test_face_validity_separate_from_geometry(self):
        """鼻點不可信時,幾何仍有效——這是覆蓋率的關鍵。"""
        seq = make_reach_sequence()
        seq[:, 0, 2] = 0.1                       # 鼻子不可信
        kin = kinematic_features(seq, FPS)
        blk = kin[:, SIDE_SLICE["R"]]
        assert (blk[:, K_VALID] > 0.5).all()
        assert (blk[:, K_FACE_OK] < 0.5).all()
        assert np.all(blk[:, K_D_NOSE] == 0)     # 臉部量清零,不留幻覺

    def test_mirror_canonicalisation(self):
        """左右手經鏡像正規化後,同一個動作應得到同一組特徵。"""
        seq = make_reach_sequence()
        flip = seq.copy()
        idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        cx = flip[:, :, 0].mean()
        flip = flip[:, idx]
        flip[:, :, 0] = 2 * cx - flip[:, :, 0]
        r = side_view(kinematic_features(seq, FPS), "R")
        left = side_view(kinematic_features(flip, FPS), "L")
        assert np.allclose(r, left, atol=5e-2)


class TestPrimitives:
    def test_reach_produces_raise_hold(self):
        kin = kinematic_features(make_reach_sequence(), FPS)
        p = rule_primitives(kin, "R", FPS)
        assert (p == P_HOLD).sum() >= 5, "停留在臉部沒有被標出來"
        assert (p == P_RAISE).sum() >= 2, "舉手沒有被標出來"

    def test_still_hand_is_rest(self):
        """手垂在身側不動 → 全程靜置,不該冒出舉手。"""
        seq = make_reach_sequence(T=60, hold_frames=0)
        seq[:, 10, :2] = seq[0, 10, :2]          # 右腕釘住
        seq[:, 8, :2] = seq[0, 8, :2]
        kin = kinematic_features(seq, FPS)
        p = rule_primitives(kin, "R", FPS)
        assert (p == P_REST).sum() > (p == P_RAISE).sum()

    def test_ambiguous_frames_are_ignored(self):
        """關鍵點全部不可信 → 全部棄權,不能硬給標籤。"""
        seq = make_reach_sequence()
        seq[:, :, 2] = 0.05
        kin = kinematic_features(seq, FPS)
        assert (rule_primitives(kin, "R", FPS) == -1).all()

    def test_segments_have_sane_attributes(self):
        """停留片段的最小腕-鼻距離必須是真的量到的,不能是缺值的 0。"""
        from stage2.primitives import A_D_MIN, A_FACE_RATIO
        kin = kinematic_features(make_reach_sequence(), FPS)
        segs = segments_from_kinematics(kin, FPS)
        holds = [s for s in segs if s.prim == P_HOLD]
        assert holds
        for h in holds:
            assert 0.0 < h.attrs[A_D_MIN] < 1.5
            assert h.attrs[A_FACE_RATIO] > 0.5

    def test_hold_without_approach_is_not_armed(self):
        """幻覺腕點的情境:手一直在臉旁邊、從未離開 → 不算完整循環。

        這條是第一階段最貴的教訓(手背在身後時腕點被幻覺在衣領上),
        直接寫成測試,避免重構時被改掉。
        """
        seq = make_reach_sequence(T=60, hold_frames=60)
        seq[:, 10, :2] = np.array([12.0, -26.0])   # 腕點恆定停在鼻旁
        seq[:, 8, :2] = np.array([6.0, -14.0])
        seq[:, :, :2] += np.array([300.0, 200.0])
        kin = kinematic_features(seq, FPS)
        segs = segments_from_kinematics(kin, FPS)
        cycles = find_cycles(segs, kin)
        assert not any(c.armed for c in cycles)

    def test_reach_is_armed(self):
        kin = kinematic_features(make_reach_sequence(), FPS)
        segs = segments_from_kinematics(kin, FPS)
        assert any(c.armed for c in find_cycles(segs, kin))


class TestComposition:
    def test_analysis_shapes(self):
        kin = kinematic_features(make_reach_sequence(), FPS)
        a = analyze(kin, FPS)
        assert a.stats.shape == (STAT_DIM,)
        assert a.tokens.shape[0] == len(a.times)
        # 沒有 L1 嵌入時,token 只有基元 one-hot + 側別 + 片段屬性
        assert a.tokens.shape[1] == len(PRIMITIVES) + 2 + SEG_ATTR_DIM
        assert np.isfinite(a.norm_stats).all()

    def test_empty_segments_still_yield_one_token(self):
        seq = make_reach_sequence()
        seq[:, :, 2] = 0.05
        a = analyze(kinematic_features(seq, FPS), FPS)
        assert a.tokens.shape[0] == 1 and a.times.shape[0] == 1

    def test_stats_reflect_the_action(self):
        kin = kinematic_features(make_reach_sequence(hold_frames=25), FPS)
        a = analyze(kin, FPS)
        assert a.stats[S_D_MIN] < 0.9              # 手真的到臉了
        assert a.stats[S_HOLD_MEAN] > 0.5
        assert np.expm1(a.stats[S_N_CYCLE]) >= 1   # 有完整循環

    def test_grammar_scores_form_a_distribution(self):
        kin = kinematic_features(make_reach_sequence(), FPS)
        a = analyze(kin, FPS)
        sc = grammar_scores(a.segments, a.stats, a.cycles)
        assert set(sc) == set(DEEP_CLASSES)
        assert abs(sum(sc.values()) - 1.0) < 1e-5
        assert all(v >= 0 for v in sc.values())

    def test_no_contact_scores_other(self):
        """手全程垂著 → 深層應該說「其他」。"""
        seq = make_reach_sequence(T=60, hold_frames=0)
        seq[:, 10, :2] = seq[0, 10, :2]
        seq[:, 8, :2] = seq[0, 8, :2]
        kin = kinematic_features(seq, FPS)
        a = analyze(kin, FPS)
        sc = grammar_scores(a.segments, a.stats, a.cycles)
        assert max(sc, key=sc.get) == "other"

    def test_normalize_stats_is_finite_on_empty(self):
        assert np.isfinite(normalize_stats(np.zeros(STAT_DIM,
                                                    np.float32))).all()

    def test_l1_predictions_masked_where_there_is_no_input(self):
        """腕點偵測不到的幀,L1 的預測不得變成片段。

        L1 對每一幀都會給答案,包括根本沒有輸入的幀。實測有片段 96% 的
        幀量不到手腕,卻照樣被切出六段「停留臉部」——那是憑空生成的證據。
        """
        seq = make_reach_sequence()
        seq[:, 10, 2] = 0.0                       # 右腕整段偵測不到
        seq[:, 8, 2] = 0.0
        kin = kinematic_features(seq, FPS)
        # 模擬 L1「每幀都說手停在臉部」
        prim = np.full((len(seq), 2), P_HOLD, np.int8)
        a = analyze(kin, FPS, prim=prim)
        assert not [s for s in a.segments if s.side == "R"], \
            "沒有輸入的側別不該產生片段"
        assert [s for s in a.segments if s.side == "L"], \
            "有輸入的側別仍應正常分段"


class TestModels:
    def test_l1_forward_shapes(self):
        net = PrimitiveNet()
        g = torch.randn(2, GRAPH_CHANNELS, 64, NUM_NODES)
        kin = torch.randn(2, 64, KIN_DIM)
        logits, emb = net(g, kin)
        assert logits.shape == (2, 64, 2, len(PRIMITIVES))
        assert emb.shape == (2, 64, 2, net.embed_dim)

    def test_l1_variable_length(self):
        net = PrimitiveNet().eval()
        for T in (32, 128, 300):
            logits, _ = net(torch.randn(1, GRAPH_CHANNELS, T, NUM_NODES),
                            torch.randn(1, T, KIN_DIM))
            assert logits.shape[1] == T

    def test_l1_backward(self):
        net = PrimitiveNet()
        logits, _ = net(torch.randn(2, GRAPH_CHANNELS, 48, NUM_NODES),
                        torch.randn(2, 48, KIN_DIM))
        logits.sum().backward()
        assert net.trunk.blocks[0].gcn.edge_mask.grad is not None

    def test_l2_forward_and_batch_one(self):
        """batch=1 也要能跑:推論時就是一段一段來的。"""
        net = CompositionNet(token_dim=TOKEN_DIM).eval()
        for B, N in ((1, 1), (4, 12)):
            out = net(torch.randn(B, N, TOKEN_DIM), torch.rand(B, N) * 30,
                      torch.ones(B, N, dtype=torch.bool),
                      torch.randn(B, STAT_DIM))
            assert out.shape == (B, len(DEEP_CLASSES))

    def test_l2_ignores_padding(self):
        """補齊的位置不得影響結果,否則 batch 大小會改變預測。"""
        net = CompositionNet(token_dim=TOKEN_DIM).eval()
        tok = torch.randn(1, 5, TOKEN_DIM)
        tm = torch.rand(1, 5) * 10
        mask = torch.ones(1, 5, dtype=torch.bool)
        with torch.no_grad():
            a = net(tok, tm, mask, torch.zeros(1, STAT_DIM))
            pad_tok = torch.cat([tok, torch.randn(1, 4, TOKEN_DIM)], 1)
            pad_tm = torch.cat([tm, torch.rand(1, 4) * 10], 1)
            pad_mask = torch.cat([mask, torch.zeros(1, 4, dtype=torch.bool)],
                                 1)
            b = net(pad_tok, pad_tm, pad_mask, torch.zeros(1, STAT_DIM))
        assert torch.allclose(a, b, atol=1e-5)

    def test_param_budget(self):
        """小資料配方:L1 < 15 萬、L2 < 5 萬。"""
        assert count_params(PrimitiveNet()) < 150_000
        assert count_params(CompositionNet(token_dim=TOKEN_DIM)) < 50_000


class TestTaxonomy:
    def test_deep_index_round_trip(self):
        assert DEEP_CLASSES[deep_index("smoking")] == "smoking"
        assert DEEP_CLASSES[deep_index("scratch_head")] == "hair"
        assert DEEP_CLASSES[deep_index("phone")] == "phone_call"

    def test_unusable_codes_excluded(self):
        """骨架看不清 / 深層詞彙沒有的動作 → 不進 L2 訓練。"""
        assert deep_index("bad_pose") is None
        assert deep_index("back_view") is None
        assert deep_index("eating") is None      # 五類詞彙裡沒有吃東西

    def test_face_touch_is_other_not_a_named_action(self):
        """托腮/摸鼻歸「其他」——但扶眼鏡與抓頭髮必須各自成類,
        不能被吸回去,否則兩層模型分它們的能力就白做了。"""
        assert DEEP_CLASSES[deep_index("face_touch")] == "other"
        assert DEEP_CLASSES[deep_index("glasses")] == "glasses"
        assert DEEP_CLASSES[deep_index("scratch_head")] == "hair"
