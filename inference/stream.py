"""影像來源抽象:檔案 / 本機攝影機 / RTSP / YouTube / HTTP 串流,統一介面。

RTSP 的實務處理:
- 憑證特殊字元自動 percent-encode(密碼含 @ 是最常見的坑,
  例如 rtsp://admin:@admin888@host → 密碼 "@admin888" 會被誤切)
- 強制 TCP 傳輸(預設 UDP 掉包會花屏)與連線逾時
- 背景讀取執行緒「只保留最新影格」:推理速度跟不上串流時
  直接丟舊幀,延遲不會累積
- 斷線自動重連(含退避)

YouTube / HTTP(m3u8)的實務處理:
- YouTube 頁面網址不能直接餵給 OpenCV,要先用 yt-dlp 解析成媒體網址;
  解析出來的 manifest **會過期**(數小時,有時更短),所以重連時一律
  重新解析,不能沿用舊網址
- **時間戳改用串流自己的 PTS,不用牆鐘。** 這是接 HLS 最重要的一點:
  HLS 是一次送一整段(2–5 秒)的影格,牆鐘時間會把「3 秒的動作」壓成
  「0.3 秒內收到的一堆幀」。本系統判「這是抽菸一口還是扶眼鏡」全靠
  停留秒數(`HandToMouthCounter`),用牆鐘會直接把判準弄壞
- PTS 跨重連會從 0 重新起算,`_MonotonicClock` 補偏移保證時間軸單調
  遞增 —— 時間倒退會讓狀態機算出負秒數
- **收幀策略是「佇列」而不是「只保留最新」。** 這兩者對應兩種不同的
  串流行為,用錯會嚴重掉幀:

  | | RTSP / 攝影機 | HLS(YouTube / m3u8) |
  |---|---|---|
  | 影格抵達 | 逐幀、接近等間隔 | 一次一整段(2–5 秒)後空等 |
  | 該用 | 只保留最新(丟舊幀,延遲不累積) | 佇列(照順序消化整段) |

  實測用錯的代價:一個 burst 送 60 幀進來,消費端只來得及撈 2–3 幀,
  **其餘連同約 2 秒的內容整段丟掉**,30 fps 的來源有效取樣只剩 2 fps,
  舉手→停留→放下的轉折直接被跳過。改成佇列後,burst 期間的幀會被
  照順序消化完(管線單步只要 30 ms,一段 2 秒的內容 0.7 秒就吃完),
  剛好趕在下一段抵達前清空
- 收幀時就先抽稀到接近取樣率:整段 30 fps 全存進佇列很吃記憶體,而
  管線本來只要 10 fps。抽稀留 10% 餘裕(`0.9 / sample_fps`),讓消費端
  自己的取樣判斷仍有東西可挑,不會兩層濾波打架把幀濾光
- RTSP 與攝影機仍用牆鐘 + 最新幀:那條路是實地測過的,不動
"""
import os
import re
import threading
import time
from collections import deque
from typing import Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import cv2
import numpy as np

# yt-dlp 解析時偏好的最大高度:1080p 對 6GB VRAM 的推理沒有好處,
# 而且拉高只是讓網路與解碼更吃力
DEFAULT_MAX_HEIGHT = 720

# HLS 佇列保留多少秒的影格。滿了丟最舊的:對直播來說「跳到比較新的地方」
# 比「延遲無限增長」好。抽稀後約 10 fps,3 秒 = 30 幀 ≈ 80 MB @720p
DEFAULT_BUFFER_SEC = 3.0

# 開始供幀前先囤幾秒(刻意落後直播邊緣這麼多)。
#
# HLS 是「一段送完、空等、再一段」,而且收幀執行緒還要跟主執行緒的推理
# 搶 CPU/GIL。貼著直播邊緣跑的話,佇列一空消費端就得等下一段,畫面就
# 一頓一頓的。先囤起來再開始消化,顛簸由存量吸收。
#
# ⚠ 這招只在「收幀端平均跟得上即時」時有效。若平均落後,存量會以固定
# 速率被吃光,卡頓只是延後 prefill_sec 才出現 —— `lag_ratio` 就是用來
# 判斷是哪一種的。代價是多 prefill_sec 的延遲(YouTube 本來就有 5–30 秒,
# 這裡的用途是測試素材,不影響)。
DEFAULT_PREFILL_SEC = 5.0

# FFmpeg 的 av_log 等級(AV_LOG_FATAL = 8)。
#
# 想壓掉的是這一行洗版:
#     [https @ ...] Cannot reuse HTTP connection for different host: ...
# YouTube 每換一次 CDN 主機就印一次,是 AV_LOG_ERROR 但**完全良性**
# (FFmpeg 自己會重連,播放正常)。
#
# ⚠ 實測:**這個環境變數在本機的 OpenCV 5.0.0 上無效**(設 -8 仍照印),
# `OPENCV_FFMPEG_CAPTURE_OPTIONS=http_persistent;0` 也無效。訊息是 FFmpeg
# 在 C 層直接寫 fd 2,Python 層攔不到。留著這行是因為在有支援的 OpenCV
# 4.x build 上確實有效,成本為零;真正的處理在 `啟動GUI.bat`(把 stderr
# 導進 logs/),命令列則自行 `2>nul`。
FFMPEG_LOGLEVEL = "8"


def normalize_rtsp(url: str) -> str:
    """將 RTSP URL 的帳密 percent-encode(冪等,已編碼不會重複編碼)。

    以「最後一個 @」切分憑證與主機:
    rtsp://admin:@admin888@192.168.1.1:554
        → rtsp://admin:%40admin888@192.168.1.1:554
    """
    # 以「最後一個 @」切分:憑證(前)允許含 @ : / 等原始字元,
    # 主機與路徑(後)不含 @
    m = re.match(r"^(rtsps?://)(.*)@([^@]+)$", url, re.DOTALL)
    if not m:
        return url
    scheme, cred, host = m.groups()
    if ":" in cred:
        user, pw = cred.split(":", 1)
        cred = f"{quote(unquote(user), safe='')}:{quote(unquote(pw), safe='')}"
    else:
        cred = quote(unquote(cred), safe="")
    return f"{scheme}{cred}@{host}"


def _is_rtsp(src: str) -> bool:
    return isinstance(src, str) and src.lower().startswith(("rtsp://",
                                                            "rtsps://"))


def is_http_url(src) -> bool:
    return isinstance(src, str) and src.lower().startswith(("http://",
                                                            "https://"))


def is_youtube_url(src) -> bool:
    """是不是 YouTube 的頁面網址(含 youtu.be 短網址與各子網域)。

    比對主機名而不是「網址裡有沒有 youtube 字串」:後者會把
    `https://example.com/?ref=youtube.com` 誤判進來。
    """
    if not is_http_url(src):
        return False
    host = urlparse(src).netloc.lower().split(":")[0]
    host = host[4:] if host.startswith("www.") else host
    return host in ("youtube.com", "youtu.be") or host.endswith(".youtube.com")


def resolve_youtube(url: str, max_height: int = DEFAULT_MAX_HEIGHT) -> str:
    """YouTube 頁面網址 → OpenCV 讀得動的媒體網址(直播為 .m3u8)。

    用 `best[height<=N]` 這種**單一**格式,不用 `bestvideo+bestaudio`:
    後者會回傳影音分離的兩個網址,OpenCV 沒辦法自己合流。
    """
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "讀 YouTube 需要 yt-dlp,請先安裝:pip install yt-dlp") from e
    opts = {"format": f"best[height<={max_height}]/best",
            "quiet": True, "no_warnings": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("entries"):        # 頻道/播放清單網址 → 取第一個
        info = info["entries"][0]
    media = info.get("url")
    if not media:                  # 少數情況只有 requested_formats
        media = next((f["url"] for f in info.get("requested_formats", [])
                      if f.get("vcodec") not in (None, "none")), None)
    if not media:
        raise RuntimeError(f"yt-dlp 未解析出可播放的影像網址:{url}")
    return media


class _MonotonicClock:
    """PTS(秒)→ 單調遞增的時間軸。

    存在的理由:串流重連後 `CAP_PROP_POS_MSEC` 會從 0 重新起算。時間戳
    一倒退,狀態機的視窗裁剪就永遠裁不掉舊紀錄、停留時長會算成負數,
    整條判定鏈默默壞掉且很難查。重連時補一個偏移,讓時間只會往前。

    PTS 拿不到(有些容器回 0)時退回「幀數 ÷ fps」,精度足夠:
    本系統只用時間**差**,不需要絕對時刻。
    """

    def __init__(self, fps: float):
        self.fps = max(float(fps), 1e-3)
        self.frames = 0
        self.offset = 0.0
        self.last = 0.0

    def step(self, pts_sec: Optional[float]) -> float:
        raw = (float(pts_sec) if pts_sec and pts_sec > 0.0
               else self.frames / self.fps)
        self.frames += 1
        t = raw + self.offset
        if self.frames > 1 and t < self.last:
            step = 1.0 / self.fps
            self.offset += (self.last - t) + step
            t = self.last + step
        self.last = t
        return t


class VideoSource:
    """統一影像來源。

    - 檔案:`read()` 逐幀回傳,時間戳 = 幀序 / fps
    - 攝影機 / RTSP(is_live=True):背景執行緒持續讀取,
      `read()` 回傳「最新」影格與牆鐘時間戳;斷線自動重連
    - YouTube / HTTP(is_live=True):同上,但**時間戳來自串流 PTS**
      而非牆鐘(HLS 一次送一整段,牆鐘會把停留秒數壓扁),
      且 YouTube 每次(重新)開啟都重新解析網址(manifest 會過期)

    Args:
        src: 檔案路徑 / 攝影機編號字串 / rtsp:// / youtube 頁面 / http(s) 串流
        use_tcp: RTSP 走 TCP(建議開)
        connect_timeout_sec: 連線/讀取逾時
        reconnect_sec: 斷線重連間隔
        max_height: YouTube 解析時偏好的最大畫面高度
    """

    def __init__(self, src, use_tcp: bool = True,
                 connect_timeout_sec: float = 5.0,
                 reconnect_sec: float = 3.0,
                 max_height: int = DEFAULT_MAX_HEIGHT,
                 sample_fps: Optional[float] = None,
                 buffer_sec: float = DEFAULT_BUFFER_SEC,
                 prefill_sec: float = DEFAULT_PREFILL_SEC):
        self.reconnect_sec = reconnect_sec
        self.max_height = int(max_height)
        self.sample_fps = float(sample_fps) if sample_fps else None
        self.buffer_sec = float(buffer_sec)
        self.prefill_sec = float(prefill_sec)
        self._stop = False
        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._frame_ts = 0.0
        self._frame_id = 0
        self._last_returned = 0
        self._queue: deque = deque()
        self._dropped = 0
        self._t0 = time.time()

        if _is_rtsp(src):
            self.kind = "rtsp"
            self.src = normalize_rtsp(src)
            opts = [f"stimeout;{int(connect_timeout_sec * 1e6)}",
                    "max_delay;500000"]
            if use_tcp:
                opts.insert(0, "rtsp_transport;tcp")
            # 需在 VideoCapture 建立前設定(FFmpeg backend 讀此環境變數)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(opts)
        elif is_youtube_url(src) or is_http_url(src):
            self.kind = "youtube" if is_youtube_url(src) else "http"
            self.src = str(src)      # 保留原始網址:重連要靠它重新解析
            # 明確覆寫 capture options(不只是設定):同一個行程裡先開過
            # RTSP 的話,rtsp_transport 會殘留在環境變數裡
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"timeout;{int(connect_timeout_sec * 1e6)}")
            # setdefault:使用者若自己設了等級(除錯用)就尊重他的設定
            os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", FFMPEG_LOGLEVEL)
        elif str(src).isdigit():
            self.kind = "webcam"
            self.src = int(src)
        else:
            self.kind = "file"
            self.src = str(src)

        self.is_live = self.kind in ("rtsp", "webcam", "youtube", "http")
        # 時間戳用串流 PTS 的來源(見模組 docstring)
        self.use_pts = self.kind in ("youtube", "http")
        # HLS 是整段抵達的,要用佇列消化;RTSP/攝影機逐幀抵達,只留最新
        self.queued = self.use_pts
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟影像來源:{src}" + {
                "rtsp": "(請確認網路、帳密與路徑;密碼含特殊字元已自動編碼)",
                "youtube": "(直播可能已結束,或該影片有地區/年齡限制)",
                "http": "(請確認網址仍有效;m3u8 直播常會換網址)",
            }.get(self.kind, ""))

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 串流常回報 0 或異常值,fallback 25
        self.fps = fps if 1.0 <= fps <= 120.0 else 25.0

        self._file_idx = 0
        self._clock = _MonotonicClock(self.fps)
        # 收幀時就抽稀到 sample_fps(整段 30 fps 全存太吃記憶體)
        self._keep_interval = (0.9 / self.sample_fps
                               if self.queued and self.sample_fps else 0.0)
        self._last_kept = float("-inf")
        eff_fps = self.sample_fps if self._keep_interval else self.fps
        # 容量要放得下「囤的量 + 吸顛簸的量」,否則永遠囤不到目標就開跑
        self._prefill_frames = (int(self.prefill_sec * eff_fps)
                                if self.queued else 0)
        self._queue_max = max(8, self._prefill_frames
                              + int(self.buffer_sec * eff_fps))
        self._primed = not self._prefill_frames
        # 照即時速度供幀(只在有囤積時)。
        #
        # 少了這一步,囤積是白囤的:消費端每秒能跑 30 幀,而直播每秒只
        # 生得出 10 幀,於是它會全速衝向直播邊緣,把囤好的 5 秒在一兩秒
        # 內燒光,然後回到「每一幀都要等」的原狀。實測就是這樣 ——
        # 佇列從 62 掉到 3,每 5 秒仍卡一次。
        #
        # 節流的依據是**存量水位**,不是絕對時刻表:
        #     存量 > 目標 → 不睡,全速消化多出來的
        #     存量 ≤ 目標 → 放慢到 1/sample_fps,讓收幀端補回來
        #
        # 一開始用絕對時刻表(該在 wall0+Δt 供這一幀)的版本會死:佇列滿
        # 時 drop-oldest 讓下一幀的時間戳往前跳,節流誤以為自己超前而睡
        # 更久,於是丟更多、跳更遠 —— 正回饋螺旋,實測掉到 1.0 fps。
        # 水位控制沒有這個問題:丟棄會讓存量下降,而存量下降是加速的訊號。
        self.paced = bool(self.queued and self._prefill_frames)
        self._serve_interval = 1.0 / self.sample_fps if self.sample_fps else 0.0
        self._next_serve = 0.0
        # 落後診斷:消化掉的串流秒數 ÷ 牆鐘秒數。≈1 表示跟得上即時,
        # <1 表示正在被直播拉開距離(囤再多也只是延後卡頓)
        self._first_ts: Optional[float] = None
        self._last_out_ts = 0.0
        self._wall0 = time.time()
        # 已經抽稀到呼叫端要的取樣率 → 呼叫端**不要**再濾一次。
        # 濾兩次會出事:來源是 30 fps 網格,抽到 10 fps 後相鄰幀間距剛好
        # 就是 0.1 秒,呼叫端再判「間隔 ≥ 0.1」時,浮點誤差讓
        # 0.7 - 0.6 = 0.09999999999999998 過不了,實測 30 幀掉到剩 20。
        self.pre_sampled = bool(self.queued and self._keep_interval)
        if self.is_live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "youtube":
            # 每次開啟都重新解析:yt-dlp 給的 manifest 網址會過期,
            # 沿用舊網址的話重連會永遠失敗(而且錯誤看起來像網路問題)
            url = resolve_youtube(self.src, self.max_height)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        if self.kind in ("rtsp", "http"):
            cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小化內部緩衝延遲
            return cap
        return cv2.VideoCapture(self.src)

    # ---------- 背景讀取(僅 live) ----------

    def _reader(self) -> None:
        """持續讀最新影格;失敗即重連(含退避)。"""
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if self._stop:
                    break
                print(f"[stream] 來源中斷,{self.reconnect_sec:.0f} 秒後重連…")
                self.cap.release()
                time.sleep(self.reconnect_sec)
                self._reopen()
                continue
            ts = (self._clock.step(self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                  if self.use_pts else time.time() - self._t0)
            if self.queued:
                # 抽稀:管線只要 sample_fps,整段 30 fps 全存太吃記憶體
                if ts - self._last_kept < self._keep_interval:
                    continue
                self._last_kept = ts
            with self._lock:
                self._frame_ts = ts
                self._frame_id += 1
                if self.queued:
                    if len(self._queue) >= self._queue_max:
                        # 佇列滿 = 消費端長期跟不上。丟最舊的:對直播而言
                        # 「跳到比較新的地方」勝過「延遲無限增長」
                        self._queue.popleft()
                        self._dropped += 1
                    self._queue.append((ts, frame))
                else:
                    self._frame = frame
                self._lock.notify_all()

    def _reopen(self) -> bool:
        """重新開啟來源;失敗時回報而不是讓執行緒死掉。

        YouTube 的重開會呼叫 yt-dlp(網路 I/O),是**會丟例外**的。
        原本的寫法直接 `self.cap = self._open()`,一旦丟例外讀取執行緒
        就靜靜結束,外面只看到「read 一直逾時」,完全沒有線索可查。
        """
        try:
            self.cap = self._open()
            return True
        except Exception as e:
            print(f"[stream] 重新連線失敗({e}),稍後再試…")
            # 佔位物件:下一輪 read() 會立刻失敗,回到上面的重連退避
            self.cap = cv2.VideoCapture()
            return False

    # ---------- 對外介面 ----------

    def read(self, timeout: float = 5.0
             ) -> Tuple[Optional[np.ndarray], float]:
        """讀一幀。回傳 (frame, timestamp);結束/逾時回傳 (None, ts)。

        - RTSP / 攝影機:阻塞等待「新」影格(跳過已回傳過的),時間戳為牆鐘
        - YouTube / HTTP:從佇列取**最舊**的一幀(保持時間連續),
          時間戳為串流 PTS(見模組 docstring)
        - file:逐幀,時間戳 = 幀序 / fps

        時間戳跟著影格一起取,不在這裡現算:影格是背景執行緒收的,
        回傳時的牆鐘已經比它實際抵達的時刻晚了。
        """
        if self.queued:
            with self._lock:
                if not self._primed:
                    # 開跑前先囤滿 prefill_sec。囤不到就以逾時收場後照常
                    # 開跑 —— 短片或供幀本來就慢的來源不該被卡在這裡。
                    self._lock.wait_for(
                        lambda: (len(self._queue) >= self._prefill_frames
                                 or self._stop), timeout=timeout)
                    self._primed = True
                if not self._lock.wait_for(
                        lambda: self._queue or self._stop, timeout=timeout):
                    return None, self._now()             # 逾時
                if not self._queue:
                    return None, self._now()
                ts, frame = self._queue.popleft()
                if self._first_ts is None:
                    self._first_ts = ts
                    self._wall0 = time.time()
                self._last_out_ts = ts
                low = len(self._queue) <= self._prefill_frames
            # 節流一定要在鎖外面睡:握著條件變數睡會把收幀執行緒一起擋住,
            # 佇列就填不進來了——正好毀掉節流想保住的那段存量
            if self.paced and low:
                delay = self._next_serve - time.time()
                if delay > 0:
                    time.sleep(min(delay, 1.0))
            self._next_serve = time.time() + self._serve_interval
            return frame, ts

        if self.is_live:
            with self._lock:
                if not self._lock.wait_for(
                        lambda: self._frame_id > self._last_returned
                        or self._stop,
                        timeout=timeout):
                    return None, self._now()             # 逾時
                if self._stop or self._frame is None:
                    return None, self._now()
                self._last_returned = self._frame_id
                return self._frame.copy(), self._frame_ts

        ok, frame = self.cap.read()
        ts = self._file_idx / self.fps
        self._file_idx += 1
        return (frame if ok else None), ts

    @property
    def dropped(self) -> int:
        """佇列滿而丟掉的幀數(> 0 表示消費端長期跟不上串流)。"""
        return self._dropped

    @property
    def lag_ratio(self) -> float:
        """消化掉的串流秒數 ÷ 牆鐘秒數。

        這是判斷「囤緩衝救不救得了卡頓」的唯一依據:

            ≈ 1.0  跟得上即時 → 顛簸是抖動,囤 prefill_sec 就吸得掉
            < 1.0  正被直播拉開 → 存量會以固定速率見底,囤再多只是把
                   卡頓延後 prefill_sec 才發生,得從別處省 CPU

        還沒取過幀時回 0。
        """
        if self._first_ts is None:
            return 0.0
        wall = max(time.time() - self._wall0, 1e-6)
        return (self._last_out_ts - self._first_ts) / wall

    def _now(self) -> float:
        """沒有影格可回傳時的時間戳(逾時用)。

        PTS 來源不能在這裡憑空生一個牆鐘時間:那會和影格的時間軸對不
        起來。回最後一次收到的影格時間即可 —— 呼叫端拿到 frame=None
        時本來就不會拿這個時間去做判定。
        """
        return self._frame_ts if self.use_pts else time.time() - self._t0

    def release(self) -> None:
        self._stop = True
        if self.is_live:
            with self._lock:
                self._lock.notify_all()
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                # reader 卡在 cap.read()(網路掛死):此時 release cap
                # 會與 reader 併用同一資源而崩潰;留給 daemon 執行緒
                # 隨程序結束回收
                return
        self.cap.release()
