"""影片下載:格式選擇、選項組裝、取消。

實際下載不在單元測試裡(會依賴網路與外部服務)。這裡測的是**下載之外**
的部分:選到什麼格式、選項有沒有組對、取消旗標會不會生效。
"""
from pathlib import Path

import pytest

from inference.downloader import (Cancelled, DEFAULT_MAX_HEIGHT,
                                  OUT_TEMPLATE, VideoDownloader,
                                  format_selector, human_duration,
                                  human_size)


class TestFormatSelector:
    def test_prefers_split_streams_for_higher_quality(self):
        """720p 以上只有 DASH 影音分離,要先試 bestvideo+bestaudio。

        只挑已合流的格式會被鎖在 360p(itag 18)—— 那是現代 YouTube
        唯一的漸進式格式,看手部動作會不夠清楚。
        """
        sel = format_selector(720)
        assert sel.startswith("bestvideo[height<=720]+bestaudio")

    def test_falls_back_to_muxed_then_anything(self):
        """挑不到就退,寧可拿 360p 也不要整個失敗。"""
        parts = format_selector(720).split("/")
        assert parts[-1] == "best"
        assert any(p.startswith("best[height<=720]") for p in parts)

    def test_height_is_honoured(self):
        assert "height<=480" in format_selector(480)
        assert "height<=1080" in format_selector(1080)

    def test_no_audio_selector_skips_bestaudio(self):
        sel = format_selector(720, audio=False)
        assert "bestaudio" not in sel
        assert "bestvideo[height<=720]" in sel


class TestOptions:
    def _dl(self, **kw):
        return VideoDownloader("https://example.com/watch?v=x", **kw)

    def test_probe_options_do_not_write_anything(self):
        """查詢只是看一眼,不該碰檔案系統或設定輸出路徑。"""
        opts = self._dl()._opts(for_probe=True)
        for key in ("paths", "outtmpl", "progress_hooks"):
            assert key not in opts

    def test_playlist_is_not_followed(self):
        """貼到播放清單網址時只下這一支 —— 不要默默開始下載 500 支。"""
        assert self._dl()._opts()["noplaylist"] is True
        assert self._dl()._opts(for_probe=True)["noplaylist"] is True

    def test_ffmpeg_location_is_pinned(self):
        """本機 PATH 上沒有 ffmpeg。不指定的話影音合流會失敗,
        而且錯誤訊息看起來像格式問題。"""
        loc = self._dl()._opts()["ffmpeg_location"]
        assert loc and Path(loc).exists()

    def test_output_template_includes_video_id(self):
        """同名影片(翻拍、轉載)很常見,加 id 才不會互相覆蓋。"""
        assert "%(id)s" in OUT_TEMPLATE
        assert self._dl()._opts()["outtmpl"] == OUT_TEMPLATE

    def test_windows_safe_filenames(self):
        assert self._dl()._opts()["windowsfilenames"] is True

    def test_resume_enabled(self):
        assert self._dl()._opts()["continuedl"] is True

    def test_out_dir_is_passed(self, tmp_path):
        opts = self._dl(out_dir=str(tmp_path))._opts()
        assert opts["paths"]["home"] == str(tmp_path)

    def test_default_height(self):
        assert f"height<={DEFAULT_MAX_HEIGHT}" in self._dl()._opts()["format"]


class TestCancel:
    def _dl(self):
        return VideoDownloader("https://example.com/x")

    def test_hook_raises_after_cancel(self):
        dl = self._dl()
        dl._hook({"status": "downloading"})       # 還沒取消 → 不該丟
        dl.cancel()
        with pytest.raises(Cancelled):
            dl._hook({"status": "downloading"})

    def test_postprocessor_hook_also_cancels(self):
        """按了取消不該還要等影音合流跑完。"""
        dl = self._dl()
        dl.cancel()
        with pytest.raises(Cancelled):
            dl._pp_hook({"status": "started"})

    def test_progress_callback_receives_fraction_and_text(self):
        seen = []
        dl = VideoDownloader("https://example.com/x",
                             on_progress=seen.append)
        dl._hook({"status": "downloading", "downloaded_bytes": 512,
                  "total_bytes": 1024, "speed": 2 * 2**20})
        assert seen[0]["frac"] == pytest.approx(0.5)
        assert "MB/s" in seen[0]["text"]

    def test_unknown_total_does_not_crash(self):
        """伺服器沒給總大小時進度算不出來,但不能因此爆掉。"""
        seen = []
        dl = VideoDownloader("https://example.com/x",
                             on_progress=seen.append)
        dl._hook({"status": "downloading", "downloaded_bytes": 512})
        assert seen[0]["frac"] == 0.0
        assert "未知" in seen[0]["text"]


class TestHumanFormat:
    @pytest.mark.parametrize("n,expect", [
        (None, "未知"), (0, "未知"), (512, "512.0 B"),
        (1536, "1.5 KB"), (5 * 2**20, "5.0 MB"), (3 * 2**30, "3.0 GB"),
    ])
    def test_size(self, n, expect):
        assert human_size(n) == expect

    @pytest.mark.parametrize("sec,expect", [
        (None, "未知"), (0, "未知"), (45, "0:45"),
        (95, "1:35"), (3661, "1:01:01"),
    ])
    def test_duration(self, sec, expect):
        assert human_duration(sec) == expect
