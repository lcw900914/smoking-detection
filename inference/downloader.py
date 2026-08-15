"""下載 YouTube(等)影片存檔,供離線標記與訓練使用。

與 `recorder.py` 的分工,差別只有一件事:**來源會不會結束**。

    直播 → recorder.py   沒有結尾,所以「讀不到東西 = 斷線,該重連」成立
    影片 → 本模組        有結尾,錄影器那個假設會變成無限重錄同一支

所以這裡不自己搬位元組,直接讓 yt-dlp 做它擅長的事(挑格式、續傳、
影音合流),我們只負責介面、路徑與取消。

為什麼不沿用 `VideoSource` 讀 YouTube:那條路徑只拿得到 itag 18(360p)
—— 現代 YouTube 只有那一個影音已合流的漸進式格式,更高畫質是 DASH 影音
分離,OpenCV 沒辦法自己合流。yt-dlp 會叫 ffmpeg 合流,所以下載拿得到
720p/1080p,對之後要看清楚手部動作差很多。
"""
import re
import threading
from pathlib import Path
from typing import Callable, Optional

from inference.recorder import ffmpeg_exe

DEFAULT_OUT_DIR = "downloads"
DEFAULT_MAX_HEIGHT = 720

# 檔名樣板:標題後面接影片 id。同名影片(翻拍、轉載)很常見,加了 id
# 才不會互相覆蓋,而且事後能從檔名追回來源。
OUT_TEMPLATE = "%(title)s [%(id)s].%(ext)s"


def format_selector(max_height: int = DEFAULT_MAX_HEIGHT,
                    audio: bool = True) -> str:
    """畫質上限 → yt-dlp 的格式選擇字串。

    先試影音分離(才有 720p 以上)再退回已合流的,最後退回「有什麼拿
    什麼」——寧可拿到 360p,也不要因為挑不到格式就整個失敗。
    """
    h = int(max_height)
    if not audio:
        return f"bestvideo[height<={h}]/best[height<={h}]/best"
    return (f"bestvideo[height<={h}]+bestaudio/"
            f"best[height<={h}]/best")


def human_size(n: Optional[float]) -> str:
    if not n:
        return "未知"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def human_duration(sec: Optional[float]) -> str:
    if not sec:
        return "未知"
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Cancelled(Exception):
    """使用者按了取消。"""


class VideoDownloader:
    """下載單支影片。probe() 先看清楚再 run(),避免下錯或下爆磁碟。"""

    def __init__(self, url: str, out_dir: str = DEFAULT_OUT_DIR,
                 max_height: int = DEFAULT_MAX_HEIGHT, audio: bool = True,
                 log: Callable = print,
                 on_progress: Optional[Callable] = None):
        self.url = url.strip()
        self.out_dir = Path(out_dir)
        self.max_height = int(max_height)
        self.audio = bool(audio)
        self.log = log
        self.on_progress = on_progress
        self._cancel = threading.Event()
        self.path: Optional[Path] = None

    def cancel(self) -> None:
        self._cancel.set()

    # ---- 選項 ----

    def _opts(self, for_probe: bool = False) -> dict:
        opts = {
            "format": format_selector(self.max_height, self.audio),
            "quiet": True, "no_warnings": True,
            # quiet 擋不掉 yt-dlp 自己的進度條(它走 stdout)。介面已經有
            # 進度條與記錄,讓它安靜下來,主控台才看得到別的訊息
            "noprogress": True,
            "noplaylist": True,       # 貼到播放清單網址時只下這一支,
                                      # 不要默默開始下載 500 支
            "ignoreerrors": False,
        }
        if for_probe:
            return opts
        opts.update({
            "paths": {"home": str(self.out_dir)},
            "outtmpl": OUT_TEMPLATE,
            "windowsfilenames": True,   # 標題常含 Windows 不合法字元
            "continuedl": True,         # 中斷後可續傳
            "merge_output_format": "mp4",
            # ffmpeg 用 imageio-ffmpeg 附帶的那支:本機 PATH 上沒有 ffmpeg,
            # 不指定的話影音合流會失敗,而且錯誤訊息看起來像格式問題
            "ffmpeg_location": ffmpeg_exe(),
            "progress_hooks": [self._hook],
            "postprocessor_hooks": [self._pp_hook],
        })
        return opts

    # ---- 進度 ----

    def _raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def _hook(self, d: dict) -> None:
        self._raise_if_cancelled()
        if self.on_progress is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            self.on_progress({
                "frac": (done / total) if total else 0.0,
                "text": (f"下載中 {human_size(done)}"
                         f" / {human_size(total)}"
                         f"  {(d.get('speed') or 0) / 2**20:.1f} MB/s"),
            })
        elif d.get("status") == "finished":
            self.on_progress({"frac": 1.0, "text": "下載完成,處理中…"})

    def _pp_hook(self, d: dict) -> None:
        """後處理(影音合流)也要能取消,不然按了取消還要等它跑完。"""
        self._raise_if_cancelled()
        if d.get("status") == "started" and self.on_progress:
            self.on_progress({"frac": 1.0,
                              "text": f"{d.get('postprocessor', '處理')}…"})

    # ---- 動作 ----

    def probe(self) -> dict:
        """先查資訊:標題、長度、是否為直播、預估大小。

        `is_live` 一定要查:直播沒有結尾,丟給下載器會一直下到磁碟滿。
        那種來源該走「直播錄影」分頁。
        """
        import yt_dlp
        with yt_dlp.YoutubeDL(self._opts(for_probe=True)) as ydl:
            info = ydl.extract_info(self.url, download=False)
        if info.get("entries"):
            info = info["entries"][0]
        size = (info.get("filesize") or info.get("filesize_approx")
                or sum(f.get("filesize") or f.get("filesize_approx") or 0
                       for f in info.get("requested_formats", [])) or None)
        return {
            "title": info.get("title") or "(無標題)",
            "id": info.get("id") or "",
            "duration": info.get("duration"),
            "height": info.get("height"),
            "is_live": bool(info.get("is_live")
                            or info.get("live_status") == "is_live"),
            "size": size,
            "uploader": info.get("uploader") or "",
        }

    def run(self) -> Optional[Path]:
        """下載;回傳最終檔案路徑。使用者取消時回傳 None。"""
        import yt_dlp
        self.out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with yt_dlp.YoutubeDL(self._opts()) as ydl:
                info = ydl.extract_info(self.url, download=True)
        except Cancelled:
            self.log("[下載] 已取消")
            return None
        except yt_dlp.utils.DownloadError as e:
            # yt-dlp 會把我們的 Cancelled 包進 DownloadError
            if self._cancel.is_set():
                self.log("[下載] 已取消")
                return None
            raise RuntimeError(_clean_error(str(e))) from e
        if info.get("entries"):
            info = info["entries"][0]
        name = info.get("requested_downloads", [{}])[0].get("filepath")
        self.path = Path(name) if name else None
        return self.path


def _clean_error(msg: str) -> str:
    """把 yt-dlp 的 ANSI 色碼與前綴清掉,錯誤訊息才看得懂。"""
    msg = re.sub(r"\x1b\[[0-9;]*m", "", msg)
    return msg.replace("ERROR: ", "").strip()
