"""追蹤狀態面板的分欄規則(`scripts/gui.py:triage`)。

面板要顯示誰、放哪一欄,是整個介面最常被調整的地方。分類抽成純函式就
測得到——tkinter 沒辦法在測試裡跑,但「誰該進哪一欄」跟畫面無關。
"""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "_gui", Path(__file__).resolve().parents[1] / "scripts" / "gui.py")
gui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gui)


def track(tid=1, presence="passing", alarm=False, events=0, level=0.0,
          stage=3, P=0.0, **extra):
    r = {"presence": presence, "alarm": alarm, "events": events,
         "level": level, "stage": stage, "P": P, "bbox": (0, 0, 1, 1)}
    r.update(extra)
    return {tid: r}


def ids(column):
    return [tid for tid, _text in column]


class TestPresentColumn:
    def test_waiting_people_are_excluded(self):
        """等待的人歸到「等待·待確認」欄,不重複列進在場。"""
        present, _, _ = gui.triage(track(presence="waiting"))
        assert present == []

    @pytest.mark.parametrize("presence", ["passing", "running", "wandering"])
    def test_non_waiting_people_are_listed(self, presence):
        present, _, _ = gui.triage(track(tid=7, presence=presence))
        assert ids(present) == [7]

    def test_shows_the_presence_behaviour(self):
        """使用者要的就是這個:ID 旁邊看得到徘徊還是經過。"""
        present, _, _ = gui.triage(track(tid=3, presence="wandering"))
        assert "徘徊" in present[0][1]
        present, _, _ = gui.triage(track(tid=3, presence="passing"))
        assert "經過" in present[0][1]

    def test_undetermined_presence_is_labelled_not_blank(self):
        """軌跡還不夠長時型態是 unknown。標「判定中」而不是留白,
        免得看起來像壞掉。"""
        present, _, _ = gui.triage(track(presence="unknown"))
        assert "判定中" in present[0][1]

    def test_keeps_the_existing_markers(self):
        present, _, _ = gui.triage(track(
            tid=5, presence="passing", events=2, orientation="back",
            moving=True, phone=True))
        text = present[0][1]
        for expect in ("ID5", "2次", "背向", "移動中", "講電話"):
            assert expect in text

    def test_sorted_by_track_id(self):
        results = {}
        for tid in (9, 2, 5):
            results.update(track(tid=tid))
        present, _, _ = gui.triage(results)
        assert ids(present) == [2, 5, 9]


class TestWatchingColumn:
    """站著不動的人才是可疑對象 —— 抽菸是站定了才做的事,走過去的只是路過。"""

    def test_waiting_and_not_yet_smoking(self):
        _, watching, _ = gui.triage(track(tid=4, presence="waiting"))
        assert ids(watching) == [4]

    def test_alarmed_people_leave_this_column(self):
        """判定出來之後就不是『待確認』了,要移到抽菸欄,不能兩邊都在。"""
        _, watching, smoking = gui.triage(
            track(tid=4, presence="waiting", alarm=True))
        assert ids(watching) == [] and ids(smoking) == [4]

    @pytest.mark.parametrize("presence", ["passing", "running", "wandering",
                                          "unknown"])
    def test_people_still_moving_are_not_listed(self, presence):
        _, watching, _ = gui.triage(track(presence=presence))
        assert watching == [], presence

    def test_shows_alert_level(self):
        _, watching, _ = gui.triage(track(presence="waiting", level=0.8))
        assert "警戒高" in watching[0][1]
        _, watching, _ = gui.triage(track(presence="waiting", level=0.0))
        assert "觀察中" in watching[0][1]

    def test_facing_away_is_flagged(self):
        """背向時骨架棄權 —— 那個人是『看不到手』,不是『確認沒抽』。"""
        _, watching, _ = gui.triage(
            track(presence="waiting", orientation="back"))
        assert "背向" in watching[0][1]


class TestSmokingColumn:
    def test_alarm_lands_here(self):
        _, _, smoking = gui.triage(track(tid=8, alarm=True, P=0.82, events=4))
        assert ids(smoking) == [8]
        assert "P0.82" in smoking[0][1] and "4次" in smoking[0][1]

    def test_alarm_while_passing_still_shows(self):
        """路過的人如果真的判出抽菸,絕不能因為不在『等待』欄就被藏起來。"""
        present, watching, smoking = gui.triage(
            track(tid=8, presence="passing", alarm=True))
        assert ids(present) == [8] and watching == [] and ids(smoking) == [8]

    def test_downgraded_alarm_is_marked_but_still_listed(self):
        """降級不否決:第二階段判『待複查』時警報還在,只是要人看一眼。"""
        _, _, smoking = gui.triage(track(alarm=True, verify="review"))
        assert len(smoking) == 1
        assert "待複查" in smoking[0][1]

    def test_confirmed_alarm_is_marked(self):
        _, _, smoking = gui.triage(track(alarm=True, verify="confirmed"))
        assert "已確認" in smoking[0][1]

    def test_no_alarm_no_entry(self):
        _, _, smoking = gui.triage(track(alarm=False))
        assert smoking == []


class TestTriageOverall:
    def test_empty_input(self):
        assert gui.triage({}) == ([], [], [])

    def test_a_person_is_never_in_both_result_columns(self):
        """觀察中與已判定是互斥的兩種狀態。"""
        for presence in ("passing", "running", "wandering", "waiting",
                         "unknown"):
            for alarm in (False, True):
                _, watching, smoking = gui.triage(
                    track(presence=presence, alarm=alarm))
                assert not (ids(watching) and ids(smoking))

    def test_mixed_scene(self):
        results = {}
        results.update(track(tid=1, presence="waiting"))              # 待確認
        results.update(track(tid=2, presence="wandering"))            # 在場
        results.update(track(tid=3, presence="passing"))              # 在場
        results.update(track(tid=4, presence="passing", alarm=True))  # 在場+抽菸
        results.update(track(tid=5, presence="waiting", alarm=True))  # 抽菸
        present, watching, smoking = gui.triage(results)
        assert ids(present) == [2, 3, 4]
        assert ids(watching) == [1]
        assert ids(smoking) == [4, 5]

    def test_every_track_lands_somewhere(self):
        """三欄加起來要涵蓋所有人,沒有人會憑空消失。"""
        for presence in ("passing", "running", "wandering", "waiting",
                         "unknown"):
            for alarm in (False, True):
                cols = gui.triage(track(tid=1, presence=presence,
                                        alarm=alarm))
                assert any(ids(c) for c in cols), (presence, alarm)


class TestVideoListing:
    """影片下載分頁的縮圖清單:掃資料夾與讀縮圖。

    影片用臨時檔現做,不吃 demo_videos/ —— 那個目錄在 .gitignore 裡,
    測試不該依賴一份新 clone 拿不到的東西。
    """

    def _make(self, path, frames=20, fps=10.0, size=(64, 48)):
        import cv2
        import numpy as np
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, size)
        for i in range(frames):
            vw.write(np.full((size[1], size[0], 3), (i * 10) % 255, np.uint8))
        vw.release()
        return path

    def test_lists_only_video_files(self, tmp_path):
        self._make(tmp_path / "a.avi")
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "cover.jpg").write_bytes(b"x")
        (tmp_path / "sub").mkdir()
        got = gui.list_videos(tmp_path)
        assert [p.name for p in got] == ["a.avi"]

    def test_newest_first(self, tmp_path):
        import os
        import time
        for name in ("old.avi", "new.avi"):
            self._make(tmp_path / name)
        os.utime(tmp_path / "old.avi", (time.time() - 999, time.time() - 999))
        assert [p.name for p in gui.list_videos(tmp_path)] == ["new.avi",
                                                               "old.avi"]

    def test_missing_folder_is_empty_not_an_error(self, tmp_path):
        assert gui.list_videos(tmp_path / "還沒建立") == []

    def test_meta_has_thumbnail_duration_and_size(self, tmp_path):
        p = self._make(tmp_path / "clip.avi", frames=20, fps=10.0)
        meta = gui.video_meta(p)
        assert meta["thumb"] is not None
        assert meta["thumb"].shape[2] == 3          # RGB,給 PIL 用
        assert meta["thumb"].shape[0] <= gui.THUMB_H
        assert meta["thumb"].shape[1] <= gui.THUMB_W
        assert meta["seconds"] == pytest.approx(2.0, abs=0.3)
        assert meta["size"] > 0

    def test_broken_file_does_not_raise(self, tmp_path):
        """壞檔只是沒有縮圖,不該讓整份清單掛掉。"""
        bad = tmp_path / "broken.mp4"
        bad.write_bytes(b"not a video")
        meta = gui.video_meta(bad)
        assert meta["thumb"] is None
        assert meta["size"] > 0

    def test_thumbnail_is_not_the_first_frame(self, tmp_path):
        """取 10% 的位置:很多影片開頭是黑畫面或版權卡,
        取第 0 幀會整排縮圖全黑,等於沒有縮圖。"""
        import cv2
        import numpy as np
        p = tmp_path / "dark_start.avi"
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (64, 48))
        for i in range(40):
            val = 0 if i < 2 else 200                # 開頭兩幀全黑
            vw.write(np.full((48, 64, 3), val, np.uint8))
        vw.release()
        assert gui.video_meta(p)["thumb"].mean() > 50
