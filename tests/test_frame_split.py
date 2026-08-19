"""場次切分的測試。

`--split session` 是這個對照組唯一能給出可信數字的切法(分層隨機切法
在現有資料下會讓模型靠記憶機位拿分,實測差 0.28 AUC)。這裡守兩件事:

1. 場次真的解析得出來 —— 檔名格式一改,切法會靜默退化成「全部同一折」,
   然後所有數字又變回洩題版本,而且不會有錯誤訊息。
2. 同一場次的段永遠在同一折 —— 這是這個切法存在的全部意義。
"""
from stage2.train_frame import session_folds, session_of, stratified_folds

import numpy as np


class TestSessionOf:
    def test_parses_alarm_clip_stem(self):
        assert session_of("alarm_track1_20260708_152117") == "20260708"
        assert session_of("alarm_track283_20260709_192653") == "20260709"

    def test_unknown_format_is_explicit(self):
        """認不出來就回 unknown,而不是猜一個 —— 猜錯會靜默洩題。"""
        assert session_of("something_else") == "unknown"


class TestSessionFolds:
    def _clips(self):
        stems = ["alarm_track1_20260708_100000",
                 "alarm_track2_20260708_110000",
                 "alarm_track3_20260709_100000",
                 "alarm_track4_20260710_100000",
                 "alarm_track5_20260710_110000"]
        return [{"stem": s} for s in stems]

    def test_one_session_per_fold(self):
        folds, names = session_folds(self._clips())
        assert names == ["20260708", "20260709", "20260710"]
        assert folds == [[0, 1], [2], [3, 4]]

    def test_no_session_spans_two_folds(self):
        """同場次的段跨折 = 訓練看得到驗證的機位 = 這個切法白做了。"""
        clips = self._clips()
        folds, _ = session_folds(clips)
        for f in folds:
            assert len({session_of(clips[i]["stem"]) for i in f}) == 1
        assert sorted(i for f in folds for i in f) == list(range(len(clips)))

    def test_every_clip_is_validated_exactly_once(self):
        folds, _ = session_folds(self._clips())
        flat = [i for f in folds for i in f]
        assert len(flat) == len(set(flat)) == len(self._clips())


class TestStratifiedFoldsUnchanged:
    """分層切法必須與 train_l2 位元級一致,加了 --split 不能動到它。"""

    def test_deterministic_and_covers_everything(self):
        y = np.array([0] * 20 + [1] * 6)
        a = stratified_folds(y, 5, seed=42)
        b = stratified_folds(y, 5, seed=42)
        assert a == b
        flat = sorted(i for f in a for i in f)
        assert flat == list(range(len(y)))

    def test_each_fold_gets_positives(self):
        """類別再少也不能整折缺席,否則那一折的 AUC 沒有定義。"""
        y = np.array([0] * 58 + [1] * 6)
        folds = stratified_folds(y, 5, seed=42)
        got = [sum(1 for i in f if y[i] == 1) for f in folds]
        assert sum(got) == 6
        assert all(n >= 1 for n in got)
