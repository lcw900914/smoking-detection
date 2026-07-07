"""影像來源抽象測試:RTSP URL 正規化與檔案來源讀取。"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from inference.stream import normalize_rtsp, VideoSource


class TestNormalizeRtsp:
    def test_password_with_at(self):
        """密碼含 @ 必須被 percent-encode(以最後一個 @ 切主機)。"""
        url = "rtsp://admin:@admin888@192.168.11.197:554"
        assert normalize_rtsp(url) == \
            "rtsp://admin:%40admin888@192.168.11.197:554"

    def test_idempotent(self):
        """重複呼叫不會二次編碼。"""
        url = "rtsp://admin:%40admin888@192.168.11.197:554"
        assert normalize_rtsp(url) == url

    def test_plain_credentials_unchanged(self):
        url = "rtsp://user:pass123@10.0.0.1:554/stream1"
        assert normalize_rtsp(url) == url

    def test_no_credentials(self):
        url = "rtsp://10.0.0.1:554/stream"
        assert normalize_rtsp(url) == url

    def test_special_chars(self):
        url = "rtsp://user:p#s/w@10.0.0.1:554/ch1"
        out = normalize_rtsp(url)
        assert out == "rtsp://user:p%23s%2Fw@10.0.0.1:554/ch1"

    def test_non_rtsp_passthrough(self):
        assert normalize_rtsp("video.mp4") == "video.mp4"


class TestVideoSourceFile:
    @pytest.fixture
    def video_file(self, tmp_path):
        path = tmp_path / "clip.avi"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                             10.0, (64, 48))
        for i in range(20):
            vw.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
        vw.release()
        return str(path)

    def test_reads_all_frames_with_timestamps(self, video_file):
        vs = VideoSource(video_file)
        assert vs.kind == "file" and not vs.is_live
        assert vs.fps == pytest.approx(10.0)
        count = 0
        while True:
            frame, ts = vs.read()
            if frame is None:
                break
            assert ts == pytest.approx(count / 10.0)
            count += 1
        assert count == 20
        vs.release()

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            VideoSource("no_such_file.mp4")

    def test_kind_detection(self):
        assert VideoSource.__init__ is not None
        # 僅驗證來源型別判斷(不實際連線)
        from inference.stream import _is_rtsp
        assert _is_rtsp("rtsp://x") and _is_rtsp("RTSPS://x")
        assert not _is_rtsp("0") and not _is_rtsp("a.mp4")
