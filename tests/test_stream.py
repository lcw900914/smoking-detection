"""影像來源抽象測試:RTSP URL 正規化、檔案來源讀取、網址判斷與 PTS 時間軸。

網路來源(YouTube / m3u8)不在單元測試裡實際連線 —— 那會讓測試依賴外
部服務與網路,而且直播隨時會結束。這裡測的是**連線之外**的部分:網址
怎麼分類、以及 PTS 時間軸在重連時的行為(那才是會靜靜弄壞判定的地方)。
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from inference.stream import (_MonotonicClock, is_http_url, is_youtube_url,
                              normalize_rtsp, VideoSource)


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

    def test_file_does_not_use_pts(self, video_file):
        """檔案的時間戳是幀序/fps,不該去問串流 PTS。"""
        vs = VideoSource(video_file)
        assert not vs.use_pts
        vs.release()


class TestUrlDetection:
    @pytest.mark.parametrize("url", [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtube.com/live/abc123",
        "http://m.youtube.com/watch?v=abc",
        "https://youtu.be/abc123",
        "https://music.youtube.com/watch?v=abc",
    ])
    def test_youtube_urls(self, url):
        assert is_youtube_url(url) and is_http_url(url)

    @pytest.mark.parametrize("url", [
        # 主機名比對,不是「網址裡有沒有 youtube 這串字」——
        # 後者會把這兩個誤判成 YouTube,然後丟給 yt-dlp 解析失敗
        "https://example.com/?ref=youtube.com",
        "https://notyoutube.com/watch?v=abc",
        "https://cam.example.gov.tw/live.m3u8",
        "rtsp://10.0.0.1:554/stream",
        "video.mp4",
        "0",
    ])
    def test_not_youtube(self, url):
        assert not is_youtube_url(url)

    def test_plain_http_is_http_but_not_youtube(self):
        url = "https://cam.example.gov.tw/live.m3u8"
        assert is_http_url(url) and not is_youtube_url(url)

    def test_non_string_inputs(self):
        assert not is_http_url(0) and not is_youtube_url(None)


class TestQueueDiscipline:
    """HLS 一次送一整段,收幀策略必須是佇列而不是「只保留最新」。

    用錯的代價是實測出來的:一個 burst 送 60 幀進來,消費端只撈得到 2–3
    幀,其餘連同約 2 秒的內容整段丟掉,30 fps 的來源有效取樣剩 2 fps。
    這裡不連網,直接對 VideoSource 的收幀邏輯灌假影格來驗行為。
    """

    def _source(self, sample_fps=10.0, buffer_sec=3.0, fps=30.0,
                prefill_sec=0.0):
        """做一個不開任何來源的 VideoSource(只測收/取幀那一段)。"""
        vs = VideoSource.__new__(VideoSource)
        vs._stop = False
        vs._lock = __import__("threading").Condition()
        vs._frame = None
        vs._frame_ts = 0.0
        vs._frame_id = 0
        vs._last_returned = 0
        vs._queue = __import__("collections").deque()
        vs._dropped = 0
        vs._t0 = 0.0
        vs.fps = fps
        vs.queued = True
        vs.use_pts = True
        vs.is_live = True
        vs.sample_fps = sample_fps
        vs._keep_interval = 0.9 / sample_fps if sample_fps else 0.0
        vs._last_kept = float("-inf")
        eff = sample_fps or fps
        vs.prefill_sec = prefill_sec
        vs._prefill_frames = int(prefill_sec * eff)
        vs._queue_max = max(8, vs._prefill_frames + int(buffer_sec * eff))
        vs._primed = not vs._prefill_frames
        vs.pre_sampled = bool(vs.queued and vs._keep_interval)
        vs.paced = bool(vs.queued and vs._prefill_frames)
        vs._serve_interval = 1.0 / sample_fps if sample_fps else 0.0
        vs._next_serve = 0.0
        vs._first_ts = None
        vs._last_out_ts = 0.0
        vs._wall0 = __import__("time").time()
        return vs

    def _push(self, vs, ts):
        """模擬 reader 收到一幀(抽稀 + 入佇列),回傳有沒有留下。"""
        if ts - vs._last_kept < vs._keep_interval:
            return False
        vs._last_kept = ts
        with vs._lock:
            vs._frame_ts = ts
            vs._frame_id += 1
            if len(vs._queue) >= vs._queue_max:
                vs._queue.popleft()
                vs._dropped += 1
            vs._queue.append((ts, np.full((4, 4, 3), 1, np.uint8)))
        return True

    def test_burst_is_not_collapsed_to_one_frame(self):
        """整段 60 幀(2 秒 @30fps)抵達後,應留下約 2 秒 ×10fps 的量,
        而不是只剩最後一幀。"""
        vs = self._source()
        kept = sum(self._push(vs, i / 30.0) for i in range(60))
        assert 18 <= kept <= 24, kept
        assert len(vs._queue) == kept

    def test_frames_come_out_oldest_first(self):
        """必須照時間順序吐出來:時間軸連續才算得出停留秒數。"""
        vs = self._source()
        for i in range(60):
            self._push(vs, i / 30.0)
        out = []
        while vs._queue:
            _f, ts = vs.read(timeout=0.1)
            out.append(ts)
        assert out == sorted(out)
        assert out[0] < 0.1                       # 從最舊開始,不是最新

    def test_decimates_to_requested_rate(self):
        """收幀端抽稀後就該正好是呼叫端要的取樣率。"""
        vs = self._source(sample_fps=10.0)
        for i in range(90):                  # 3 秒 @30fps
            self._push(vs, i / 30.0)
        assert len(vs._queue) == 30          # 3 秒 × 10 fps

    def test_double_filtering_would_lose_a_third_of_frames(self):
        """釘死這個 aliasing:抽稀後**不可以**再濾一次。

        30 fps 網格抽到 10 fps,相鄰幀間距剛好就是 0.1 秒。呼叫端若再判
        「間隔 ≥ 0.1」,浮點誤差(0.7 - 0.6 = 0.09999999999999998)會讓
        三分之一的幀過不了。所以 VideoSource 用 `pre_sampled` 明講「已經
        抽好了」,呼叫端看到就跳過自己的取樣判斷。
        """
        vs = self._source(sample_fps=10.0)
        for i in range(90):
            self._push(vs, i / 30.0)
        kept = [ts for ts, _ in vs._queue]

        last, passed = float("-inf"), 0      # 模擬呼叫端再濾一次
        for ts in kept:
            if ts - last >= 0.1:
                passed += 1
                last = ts
        assert passed < len(kept), "浮點誤差沒發生的話這個測試就沒有意義了"
        assert passed <= 22                  # 實測 30 → 20

    def test_pre_sampled_flag_is_set(self):
        """呼叫端靠這個旗標決定要不要自己再取樣。"""
        assert self._source(sample_fps=10.0).pre_sampled is True

    def test_not_pre_sampled_without_sample_fps(self):
        """沒告知取樣率就不能宣稱抽好了,呼叫端得自己濾。"""
        vs = self._source(sample_fps=None)
        assert vs.pre_sampled is False

    def test_full_queue_drops_oldest(self):
        """佇列滿 = 消費端長期跟不上:丟最舊的,延遲才有上限。"""
        vs = self._source(buffer_sec=1.0)     # 只放 10 幀
        for i in range(300):
            self._push(vs, i / 30.0)
        assert len(vs._queue) == vs._queue_max
        assert vs.dropped > 0
        newest = vs._queue[-1][0]
        assert newest > 9.0                   # 留下來的是最近的內容

    def test_read_timeout_on_empty_queue(self):
        vs = self._source()
        frame, _ts = vs.read(timeout=0.05)
        assert frame is None


class TestPrefill:
    """開始供幀前先囤幾秒:讓存量吸收 HLS「一段送完就空等」的顛簸。

    實測(YouTube 街景直播 45 秒):無 prefill 8.0 fps、prefill 5 秒
    10.9 fps,而且佇列被丟棄的幀從 238 降到 65。
    """

    def _source(self, **kw):
        return TestQueueDiscipline._source(TestQueueDiscipline(), **kw)

    _push = TestQueueDiscipline._push

    def test_capacity_leaves_room_beyond_the_prefill(self):
        """容量必須 > 囤的量,否則永遠囤不滿就被上限卡住,一直不開跑。"""
        vs = self._source(sample_fps=10.0, buffer_sec=3.0, prefill_sec=5.0)
        assert vs._prefill_frames == 50
        assert vs._queue_max == 80

    def test_waits_until_prefilled(self):
        """囤不夠就先等 —— 這正是「刻意落後直播邊緣」的實作。"""
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=5.0)
        for i in range(30):                  # 只有 1 秒的量,不夠 5 秒
            self._push(vs, i / 30.0)
        a = _t.perf_counter()
        vs.read(timeout=0.1)
        assert _t.perf_counter() - a >= 0.09, "沒有等就直接開跑了"

    def test_serves_once_prefilled(self):
        vs = self._source(sample_fps=10.0, prefill_sec=5.0)
        for i in range(180):                 # 6 秒的量,超過門檻
            self._push(vs, i / 30.0)
        frame, ts = vs.read(timeout=0.05)
        assert frame is not None
        assert ts == 0.0                     # 從最舊的開始供

    def test_prefill_only_blocks_once(self):
        """囤過一次就不再囤:之後短暫見底不該整條停下來重新囤 5 秒。"""
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=0.5)
        for i in range(60):                  # 2 秒的量,超過 0.5 秒門檻
            self._push(vs, i / 30.0)
        vs.read(timeout=0.5)
        assert vs._primed
        vs._queue.clear()                    # 模擬佇列見底
        vs.paced = False                     # 節流另外測,這裡只驗囤積邏輯
        self._push(vs, 100.0)                # 只補一幀
        a = _t.perf_counter()
        frame, _ts = vs.read(timeout=0.5)
        assert frame is not None             # 立刻拿得到,不再等囤滿
        assert _t.perf_counter() - a < 0.05

    def test_timeout_gives_up_and_serves_what_it_has(self):
        """囤不到目標就以逾時收場,然後**照常供幀**。

        這裡不可以回 None:呼叫端會把 None 當成串流故障(印「等待串流影格」
        並重試)。供幀本來就慢、或影片根本沒那麼長的來源,不該被當成壞掉。
        """
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=5.0)
        for i in range(30):
            self._push(vs, i / 30.0)
        a = _t.perf_counter()
        frame, _ts = vs.read(timeout=0.05)
        assert _t.perf_counter() - a >= 0.04       # 有等過(嘗試囤積)
        assert frame is not None                   # 但還是給了手上有的
        assert vs._primed                          # 且不再重試囤積

    def test_disabled_by_default_in_helper(self):
        vs = self._source(sample_fps=10.0)
        assert vs._primed is True                 # prefill_sec=0 → 直接開跑
        assert vs.paced is False                  # 沒囤就沒有存量要保


class TestPacing:
    """照即時速度供幀 —— 沒有這一步,囤積是白囤的。

    消費端每秒能跑 30 幀,直播每秒只生得出 10 幀。不節流的話它會全速衝向
    直播邊緣,把囤好的 5 秒在一兩秒內燒光,然後回到「每幀都要等」的原狀。
    實測就是這樣:佇列從 62 掉到 3,每 5 秒仍卡一次。
    """

    def _source(self, **kw):
        return TestQueueDiscipline._source(TestQueueDiscipline(), **kw)

    _push = TestQueueDiscipline._push

    def test_paces_when_buffer_is_at_or_below_target(self):
        """存量沒到目標水位 → 放慢到 1/sample_fps,讓收幀端補回來。"""
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=5.0)   # 目標 50 幀
        vs._primed = True                        # 囤積邏輯另外測
        for i in range(90):                      # 只有 30 幀,低於水位
            self._push(vs, i / 30.0)
        vs.read(timeout=0.5)                     # 第一次:建立節流基準
        a = _t.perf_counter()
        for _ in range(3):
            vs.read(timeout=0.5)
        elapsed = _t.perf_counter() - a
        assert 0.25 <= elapsed < 0.6, elapsed

    def test_does_not_pace_when_buffer_is_above_target(self):
        """存量高於水位 → 全速消化多出來的。

        這是水位控制的關鍵:丟棄會讓存量下降,而存量下降是**加速**的
        訊號。用絕對時刻表的版本剛好相反(丟棄 → 誤以為超前 → 睡更久
        → 丟更多),實測會螺旋掉到 1.0 fps。
        """
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=0.3)   # 目標 3 幀
        vs._primed = True
        for i in range(90):                      # 30 幀,遠高於水位
            self._push(vs, i / 30.0)
        vs.read(timeout=0.5)
        a = _t.perf_counter()
        for _ in range(3):
            vs.read(timeout=0.5)
        assert _t.perf_counter() - a < 0.05

    def test_unpaced_source_serves_as_fast_as_possible(self):
        """沒有囤積就不節流:RTSP 那種逐幀抵達的來源不需要被拖慢。"""
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=0.0)
        for i in range(90):
            self._push(vs, i / 30.0)
        vs.read(timeout=0.5)
        a = _t.perf_counter()
        for _ in range(3):
            vs.read(timeout=0.5)
        assert _t.perf_counter() - a < 0.05

    def test_pacing_does_not_hold_the_lock_while_sleeping(self):
        """節流必須在鎖外面睡。

        握著條件變數睡會把收幀執行緒一起擋住,佇列就填不進來 —— 正好毀掉
        節流想保住的那段存量。這裡在另一條執行緒裡趁 read() 節流時塞幀,
        塞得進去才算過。
        """
        import threading
        import time as _t
        vs = self._source(sample_fps=10.0, prefill_sec=5.0)   # 水位遠高於存量
        vs._primed = True
        for i in range(30):
            self._push(vs, i / 30.0)
        vs.read(timeout=0.5)

        pushed = threading.Event()

        def writer():
            _t.sleep(0.05)                        # 等 read() 進入節流睡眠
            self._push(vs, 50.0)
            pushed.set()

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        vs.read(timeout=0.5)                      # 這一次會睡約 0.1 秒
        t.join(timeout=1.0)
        assert pushed.is_set(), "收幀端被節流的睡眠擋住了"


class TestLagRatio:
    """lag_ratio 是判斷「囤緩衝救不救得了卡頓」的唯一依據。"""

    def _source(self, **kw):
        return TestQueueDiscipline._source(TestQueueDiscipline(), **kw)

    _push = TestQueueDiscipline._push

    def test_zero_before_any_frame(self):
        assert self._source().lag_ratio == 0.0

    def test_keeping_up_is_about_one(self, monkeypatch):
        vs = self._source()
        clock = {"t": 1000.0}
        monkeypatch.setattr("inference.stream.time.time",
                            lambda: clock["t"])
        for i in range(30):
            self._push(vs, i / 10.0)
        vs.read(timeout=0.01)                 # 第一幀:設定原點
        clock["t"] += 2.0                     # 牆鐘走 2 秒
        for _ in range(20):                   # 消化掉 2 秒的串流時間
            vs.read(timeout=0.01)
        assert vs.lag_ratio == pytest.approx(1.0, abs=0.06)

    def test_falling_behind_is_below_one(self, monkeypatch):
        vs = self._source()
        clock = {"t": 1000.0}
        monkeypatch.setattr("inference.stream.time.time",
                            lambda: clock["t"])
        for i in range(30):
            self._push(vs, i / 10.0)
        vs.read(timeout=0.01)
        clock["t"] += 10.0                    # 牆鐘走 10 秒
        for _ in range(10):                   # 只消化掉 1 秒的串流時間
            vs.read(timeout=0.01)
        assert vs.lag_ratio < 0.5


class TestMonotonicClock:
    """PTS 時間軸:時間一倒退,狀態機的視窗裁剪與停留計時就會默默壞掉。"""

    def test_passes_pts_through(self):
        c = _MonotonicClock(fps=25.0)
        assert c.step(10.0) == pytest.approx(10.0)
        assert c.step(10.04) == pytest.approx(10.04)

    def test_falls_back_to_frame_count(self):
        """有些容器 POS_MSEC 回 0 —— 退回幀數/fps,精度足夠(只用時間差)。"""
        c = _MonotonicClock(fps=10.0)
        assert c.step(0.0) == pytest.approx(0.0)
        assert c.step(None) == pytest.approx(0.1)
        assert c.step(0.0) == pytest.approx(0.2)

    def test_reconnect_reset_does_not_go_backwards(self):
        """重連後 PTS 從 0 重新起算,時間軸必須接著往前長。"""
        c = _MonotonicClock(fps=10.0)
        for t in (5.0, 5.1, 5.2):
            c.step(t)
        after = c.step(0.0)                 # 重連
        assert after > 5.2
        assert c.step(0.1) == pytest.approx(after + 0.1, abs=1e-6)

    def test_monotonic_across_many_resets(self):
        c = _MonotonicClock(fps=10.0)
        last = -1.0
        for _ in range(5):
            for i in range(10):
                t = c.step(i / 10.0)
                assert t > last
                last = t

    def test_never_jumps_backwards_on_jitter(self):
        c = _MonotonicClock(fps=10.0)
        c.step(3.0)
        assert c.step(2.9) > 3.0            # 些微亂序也不許倒退
