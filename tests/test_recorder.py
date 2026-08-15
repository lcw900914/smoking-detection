"""直播錄影:資料夾命名、跨日準備、保留天數刪除、ffmpeg 命令組裝。

`prune_days()` 會**遞迴刪除目錄**,所以這裡花最多篇幅在「不該刪什麼」上:
一個過度熱心的清理程式會把使用者手動整理的資料一起帶走,而那種損失是
救不回來的。實際連線錄影不在單元測試裡(會依賴外部服務與網路)。
"""
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from inference.recorder import (DAY_RE, build_command, day_name,
                                ensure_day_dirs, prune_days, site_slug)


class TestSiteSlug:
    @pytest.mark.parametrize("url,expect", [
        ("https://www.youtube.com/watch?v=3nyPER2kzqk", "youtube_3nyPER2kzqk"),
        ("https://youtu.be/3nyPER2kzqk", "youtube_3nyPER2kzqk"),
        ("https://www.youtube.com/live/3nyPER2kzqk", "youtube_3nyPER2kzqk"),
    ])
    def test_youtube_uses_video_id(self, url, expect):
        assert site_slug(url) == expect

    def test_same_stream_same_folder_despite_tracking_params(self):
        """同一個直播每次啟動都要接續寫進同一個資料夾,不是每次開新的。"""
        a = site_slug("https://www.youtube.com/watch?v=abc123")
        b = site_slug("https://www.youtube.com/watch?v=abc123&t=42&si=xyz")
        assert a == b == "youtube_abc123"

    def test_m3u8_uses_host_and_path(self):
        slug = site_slug("https://cam.example.gov.tw/live/ch1.m3u8")
        assert slug == "cam_example_gov_tw_live_ch1"

    def test_slug_is_filesystem_safe(self):
        slug = site_slug("https://a.b/c d?e=f&g=h#i")
        assert not set(slug) & set(' <>:"/\\|?*')

    def test_different_streams_get_different_folders(self):
        assert site_slug("https://youtu.be/aaa") != \
            site_slug("https://youtu.be/bbb")


class TestDayDirs:
    def test_day_name_format(self):
        assert day_name(datetime(2026, 8, 13)) == "20260813"
        assert DAY_RE.match(day_name())

    def test_creates_today_and_tomorrow(self, tmp_path):
        """明天的資料夾要**先**建好。

        輸出路徑交給 ffmpeg 的 strftime 展開,午夜一到它直接往新日期寫;
        那個資料夾不存在的話 ffmpeg 開檔失敗、錄影中斷 —— 而且正好發生在
        沒人看著的半夜。
        """
        made = ensure_day_dirs(tmp_path, datetime(2026, 8, 13, 23, 59))
        assert [d.name for d in made] == ["20260813", "20260814"]
        assert all(d.is_dir() for d in made)

    def test_idempotent(self, tmp_path):
        when = datetime(2026, 8, 13)
        ensure_day_dirs(tmp_path, when)
        ensure_day_dirs(tmp_path, when)          # 不該炸
        assert (tmp_path / "20260813").is_dir()

    def test_month_and_year_rollover(self, tmp_path):
        made = ensure_day_dirs(tmp_path, datetime(2026, 12, 31, 12, 0))
        assert [d.name for d in made] == ["20261231", "20270101"]


class TestPruneDays:
    def _days(self, root: Path, names):
        for n in names:
            d = root / n
            d.mkdir(parents=True)
            (d / "clip.ts").write_bytes(b"x")
        return root

    def test_keeps_newest_n(self, tmp_path):
        self._days(tmp_path, ["20260810", "20260811", "20260812",
                              "20260813", "20260814"])
        gone = prune_days(tmp_path, keep=3)
        assert sorted(d.name for d in gone) == ["20260810", "20260811"]
        assert sorted(d.name for d in tmp_path.iterdir()) == \
            ["20260812", "20260813", "20260814"]

    def test_noop_when_within_limit(self, tmp_path):
        self._days(tmp_path, ["20260813", "20260814"])
        assert prune_days(tmp_path, keep=3) == []
        assert len(list(tmp_path.iterdir())) == 2

    def test_dry_run_deletes_nothing(self, tmp_path):
        self._days(tmp_path, ["20260810", "20260813", "20260814",
                              "20260815"])
        gone = prune_days(tmp_path, keep=2, dry_run=True)
        assert len(gone) == 2
        assert len(list(tmp_path.iterdir())) == 4, "dry-run 竟然真的刪了"

    def test_ignores_anything_not_a_day_folder(self, tmp_path):
        """只認『剛好 8 位數字』。使用者手動整理的東西一律不碰。"""
        self._days(tmp_path, ["20260810", "20260811", "20260812",
                              "20260813"])
        keep_me = [
            tmp_path / "備份",
            tmp_path / "20260809_舊的",     # 有 8 位數字但不只 8 位
            tmp_path / "2026081",           # 7 位
            tmp_path / "202608100",         # 9 位
        ]
        for d in keep_me:
            d.mkdir()
            (d / "important.txt").write_text("keep")
        (tmp_path / "notes.md").write_text("keep")

        prune_days(tmp_path, keep=1)
        for d in keep_me:
            assert d.is_dir(), f"誤刪了 {d.name}"
            assert (d / "important.txt").exists()
        assert (tmp_path / "notes.md").exists()
        assert (tmp_path / "20260813").is_dir()   # 最新的日期資料夾留著

    def test_files_named_like_days_are_not_touched(self, tmp_path):
        self._days(tmp_path, ["20260813", "20260814"])
        stray = tmp_path / "20260101"
        stray.write_text("我是檔案不是資料夾")
        prune_days(tmp_path, keep=1)
        assert stray.is_file()

    def test_rejects_keep_below_one(self, tmp_path):
        """keep=0 等於要求清空,不會是有意的。"""
        self._days(tmp_path, ["20260813"])
        with pytest.raises(ValueError):
            prune_days(tmp_path, keep=0)
        assert (tmp_path / "20260813").is_dir()

    def test_missing_root_is_noop(self, tmp_path):
        assert prune_days(tmp_path / "還沒錄過", keep=3) == []

    def test_does_not_follow_symlinks(self, tmp_path):
        """符號連結不跟進去刪 —— 免得刪到連結指向的別處資料。"""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "precious.txt").write_text("keep")
        site = tmp_path / "site"
        self._days(site, ["20260813", "20260814"])
        try:
            (site / "20260101").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("此環境不允許建立符號連結")
        prune_days(site, keep=1)
        assert (outside / "precious.txt").exists()

    def test_three_day_retention_end_to_end(self, tmp_path):
        """使用者要的行為:每天一個資料夾,最多留三天。"""
        site = tmp_path / "youtube_abc"
        base = datetime(2026, 8, 10)
        for i in range(7):                        # 連錄七天
            day = base + timedelta(days=i)
            ensure_day_dirs(site, day)
            prune_days(site, keep=3, when=day)
        left = sorted(d.name for d in site.iterdir())
        # 三天**錄影**(0814-0816)+ 預先備好的明天(0817)。
        # 預建的明天不該佔掉保留額度——先前會,於是 keep=3 實際只留 2 天
        assert left == ["20260814", "20260815", "20260816", "20260817"]

    def test_prepared_tomorrow_does_not_eat_the_quota(self, tmp_path):
        """這正是先前的 off-by-one:空的明天佔掉一格,少留一天錄影。"""
        site = tmp_path / "s"
        for name in ("20260813", "20260814", "20260815", "20260816"):
            (site / name).mkdir(parents=True)
        ensure_day_dirs(site, datetime(2026, 8, 16))     # 建出 0817
        prune_days(site, keep=3, when=datetime(2026, 8, 16))
        left = sorted(d.name for d in site.iterdir())
        assert left == ["20260814", "20260815", "20260816", "20260817"]

    def test_future_folders_are_never_deleted(self, tmp_path):
        site = tmp_path / "s"
        for name in ("20260810", "20260816", "20260901"):
            (site / name).mkdir(parents=True)
        prune_days(site, keep=1, when=datetime(2026, 8, 16))
        assert sorted(d.name for d in site.iterdir()) == [
            "20260816", "20260901"]


class TestBuildCommand:
    def _cmd(self, **kw):
        return build_command("https://host/live.m3u8", "out/%Y%m%d/%H.ts",
                             "ffmpeg", **kw)

    def test_copies_without_re_encoding(self):
        """`-c copy` 是不掉幀的根本原因:沒有解碼就沒有『來不及』。"""
        cmd = self._cmd()
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
        for forbidden in ("-vcodec", "-crf", "-preset", "-vf", "-r"):
            assert forbidden not in cmd, f"{forbidden} 會觸發重編碼"

    def test_segments_into_playable_mpegts(self):
        """輸出 .ts:行程被砍在半途,檔案仍然播得動(mp4 會整個報廢)。"""
        cmd = self._cmd()
        assert cmd[cmd.index("-f") + 1] == "segment"
        assert cmd[cmd.index("-segment_format") + 1] == "mpegts"

    def test_strftime_enabled_for_day_rollover(self):
        assert cmd_value(self._cmd(), "-strftime") == "1"

    def test_segment_time_is_passed(self):
        assert cmd_value(self._cmd(segment_sec=300), "-segment_time") == "300"

    def test_audio_dropped_by_default(self):
        assert "-an" in self._cmd()
        assert "-an" not in self._cmd(audio=True)

    def test_reconnect_flags_precede_input(self):
        """-reconnect 只在 -i 之前才有效。"""
        cmd = self._cmd()
        assert cmd.index("-reconnect") < cmd.index("-i")

    def test_no_reconnect_flags_for_local_input(self):
        cmd = build_command("D:/video.ts", "out/%H.ts", "ffmpeg")
        assert "-reconnect" not in cmd


def cmd_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]
