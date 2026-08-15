"""分析結果的側車檔:存、讀、壞檔處理。

實際分析(要跑整條管線)不在單元測試裡。這裡測的是快取——它決定「同一支
影片第二次開要不要再等好幾分鐘」,而且會寫檔到使用者的資料夾,錯了很煩。
"""
import numpy as np
import pytest

from ui import analysis
from ui.analysis import Analysis, CACHE_DIR, cache_path


def make(n_poses=3, n_people=2):
    poses = {i * 3: [np.full((17, 3), i + j, np.float32)
                     for j in range(n_people)]
             for i in range(n_poses)}
    return Analysis(alarms=[1.5, 26.5], poses=poses, stride=3, fps=30.0)


class TestCachePath:
    def test_sits_in_a_subfolder(self, tmp_path):
        """側車檔放子資料夾,影片清單才不會被雜檔塞滿。"""
        p = cache_path(tmp_path / "clip.mp4")
        assert p.parent.name == CACHE_DIR
        assert p.name == "clip.npz"

    def test_does_not_touch_the_video_name(self, tmp_path):
        """原始影片不能被動到——分析只是旁邊多一個檔。"""
        video = tmp_path / "重要素材.mp4"
        assert cache_path(video) != video
        assert cache_path(video).suffix == ".npz"


class TestRoundTrip:
    def test_save_then_load(self, tmp_path):
        video = tmp_path / "clip.mp4"
        a = make()
        a.save(str(video))
        b = Analysis.load(str(video))
        assert b.alarms == pytest.approx(a.alarms)
        assert b.stride == a.stride and b.fps == pytest.approx(a.fps)
        assert sorted(b.poses) == sorted(a.poses)
        for k in a.poses:
            assert len(b.poses[k]) == len(a.poses[k])
            for x, y in zip(a.poses[k], b.poses[k]):
                assert np.allclose(x, y)

    def test_varying_people_per_frame(self, tmp_path):
        """每一幀的人數不一樣,重建時不能錯位。"""
        video = tmp_path / "c.mp4"
        poses = {0: [np.ones((17, 3), np.float32)],
                 3: [np.full((17, 3), 2, np.float32),
                     np.full((17, 3), 3, np.float32)],
                 6: []}
        Analysis(alarms=[], poses=poses, stride=3).save(str(video))
        b = Analysis.load(str(video))
        assert len(b.poses[0]) == 1 and len(b.poses[3]) == 2
        assert np.allclose(b.poses[3][1], 3)

    def test_empty_analysis(self, tmp_path):
        video = tmp_path / "c.mp4"
        Analysis().save(str(video))
        b = Analysis.load(str(video))
        assert b.alarms == [] and not b.poses

    def test_missing_cache_returns_none(self, tmp_path):
        assert Analysis.load(str(tmp_path / "沒分析過.mp4")) is None

    def test_corrupt_cache_returns_none_not_crash(self, tmp_path):
        """側車檔壞了就當沒有、重新分析,不該讓使用者連影片都打不開。"""
        video = tmp_path / "c.mp4"
        dst = cache_path(video)
        dst.parent.mkdir(parents=True)
        dst.write_bytes("這不是 npz".encode("utf-8"))
        assert Analysis.load(str(video)) is None


class TestCacheVersion:
    """側車檔是靜悄悄的:版本沒跟上,使用者重開影片看到的是上一版的結果,
    畫面上沒有任何跡象說它其實沒重跑。

    這正好發生過一次——標記從「警報成立時刻」改成「證據起點」,程式改對了
    但版本沒 +1,舊快取照載,看起來就像根本沒改。
    """

    def test_stale_version_is_discarded(self, tmp_path):
        video = tmp_path / "clip.mp4"
        make().save(str(video))
        import numpy as _np
        from ui.analysis import cache_path
        d = dict(_np.load(cache_path(str(video)), allow_pickle=False))
        d["version"] = _np.asarray(analysis.CACHE_VERSION - 1)
        _np.savez_compressed(cache_path(str(video)), **d)
        assert Analysis.load(str(video)) is None

    def test_current_version_still_loads(self, tmp_path):
        video = tmp_path / "clip.mp4"
        make().save(str(video))
        assert Analysis.load(str(video)) is not None

    def test_version_is_written_not_assumed(self, tmp_path):
        """讀的時候拿不到 version 會丟例外,那 load 就永遠回 None。"""
        import numpy as _np
        from ui.analysis import cache_path
        video = tmp_path / "clip.mp4"
        make().save(str(video))
        d = _np.load(cache_path(str(video)), allow_pickle=False)
        assert int(d["version"]) == analysis.CACHE_VERSION


class TestTruthiness:
    def test_empty_is_falsy(self):
        assert not Analysis()

    def test_alarms_only_is_truthy(self):
        assert Analysis(alarms=[1.0])

    def test_poses_only_is_truthy(self):
        """沒抽菸但有骨架也算分析過了 —— 不該因此又跑一次。"""
        assert Analysis(poses={0: [np.zeros((17, 3), np.float32)]})


class TestStride:
    def test_never_below_one(self):
        assert Analysis(stride=0).stride == 1


class TestBatcher:
    """批次偵測的代理:對應關係錯了,標記會整個偏掉而且畫面上看不出來。"""

    class _FakeDetector:
        def __init__(self):
            self.batch_calls = 0
            self.single_calls = 0

        def detect_batch(self, frames):
            self.batch_calls += 1
            return [("box", f) for f in frames]

        def detect(self, frame):
            self.single_calls += 1
            return ("single", frame)

    class _FakePipe:
        def __init__(self, det):
            self.detector = det

    def _mk(self):
        from ui.analysis import _Batcher
        det = self._FakeDetector()
        pipe = self._FakePipe(det)
        return _Batcher(pipe, 4), det, pipe

    def test_replaces_the_pipeline_detector(self):
        b, _det, pipe = self._mk()
        assert pipe.detector is b

    def test_serves_results_in_the_order_preloaded(self):
        """一對一、同順序。錯位一格就等於把某一幀的骨架安到別幀身上。"""
        b, _det, _pipe = self._mk()
        b.preload(["f0", "f1", "f2", "f3"])
        assert [b.detect(None) for _ in range(4)] == [
            ("box", "f0"), ("box", "f1"), ("box", "f2"), ("box", "f3")]

    def test_batches_once_per_preload(self):
        b, det, _pipe = self._mk()
        b.preload(["a", "b", "c", "d"])
        assert det.batch_calls == 1

    def test_falls_back_when_queue_is_empty(self):
        """預載用完還被呼叫,就照常單張算——寧可慢也不要拿到錯的那一幀。"""
        b, det, _pipe = self._mk()
        b.preload(["a"])
        b.detect(None)
        assert b.detect("later") == ("single", "later")
        assert det.single_calls == 1

    def test_empty_preload_does_not_break(self):
        b, det, _pipe = self._mk()
        b.preload([])
        assert b.detect("x") == ("single", "x")
