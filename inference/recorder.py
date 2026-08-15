"""直播錄影:把 YouTube / m3u8 直播原樣存到硬碟,供事後離線分析。

**全程不解碼**(ffmpeg `-c copy`)。這是整個模組的核心決定,其他所有性質
都是它推出來的:

- **不會掉幀**:寫進檔案的就是伺服器送來的封包本身,沒有解碼、沒有重編碼,
  中間沒有任何「來不及處理就丟掉」的環節。即時管線之所以會掉幀,是因為它
  必須解碼 + 推理才跟得上;錄影不需要,所以那個問題整個消失
- **速度穩定**:CPU 幾乎是零(只有搬位元組),瓶頸只剩磁碟與網路
- **殘檔仍可播**:輸出 mpegts(`.ts`)而不是 mp4。mp4 的索引寫在檔尾,
  行程被強制結束時整個檔案報廢;`.ts` 隨時砍掉都還播得動,對「一直開著錄」
  的用途這點很重要

目錄長這樣:

    recordings/
      youtube_3nyPER2kzqk/          ← 依直播網址建立
        20260813/                   ← 每天一個
          20260813_211500.ts        ← 預設每 10 分鐘一段
          20260813_212500.ts
        20260814/
      cam_example_gov_tw_live/
        ...

跨日不重啟 ffmpeg:輸出路徑用 ffmpeg 的 `-strftime`,午夜一到它自己就寫進
新的日期資料夾(資料夾由本模組**預先**建好,包含明天的)。重啟只發生在
連線真的斷掉時 —— 每次重啟都會重新解析 YouTube 網址,因為 manifest 會過期。

保留天數到期的舊資料夾會被刪掉,刪除邏輯刻意寫得很保守,見 `prune_days()`。
"""
import re
import shutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from inference.stream import (DEFAULT_MAX_HEIGHT, is_http_url, is_youtube_url,
                              resolve_youtube)

DAY_FMT = "%Y%m%d"
# 日期資料夾必須「剛好」是 8 位數字。刪除只認這個樣式,見 prune_days()
DAY_RE = re.compile(r"^\d{8}$")

DEFAULT_ROOT = "recordings"
DEFAULT_SEGMENT_SEC = 600      # 每段 10 分鐘
DEFAULT_KEEP_DAYS = 3


def ffmpeg_exe() -> str:
    """找一支可用的 ffmpeg。

    優先用 imageio-ffmpeg 附帶的:它隨 pip 裝好,不必要求使用者另外安裝
    ffmpeg 也不必動 PATH(本機實測系統上並沒有 ffmpeg)。
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "找不到 ffmpeg。請安裝其一:pip install imageio-ffmpeg,"
        "或把 ffmpeg 放進 PATH")


def site_slug(url: str) -> str:
    """直播網址 → 檔案系統安全、且同一個直播固定不變的資料夾名。

    YouTube 取影片 id(網址上的追蹤參數不影響),其他來源取主機名 + 路徑。
    「固定不變」是重點:每次啟動要能接續寫進同一個資料夾,不是每次都開新的。
    """
    if is_youtube_url(url):
        p = urlparse(url)
        vid = ""
        host = p.netloc.lower()
        if host.endswith("youtu.be"):
            vid = p.path.strip("/").split("/")[0]
        else:
            from urllib.parse import parse_qs
            vid = (parse_qs(p.query).get("v") or [""])[0]
            if not vid:                      # /live/<id>、/embed/<id>
                parts = [s for s in p.path.split("/") if s]
                vid = parts[-1] if parts else ""
        if vid:
            return f"youtube_{_safe(vid)}"
    if is_http_url(url):
        p = urlparse(url)
        stem = (p.netloc + p.path).rsplit(".", 1)[0]
        return _safe(stem) or "stream"
    return _safe(url) or "stream"


def _safe(text: str) -> str:
    """壓成 [A-Za-z0-9_-],避免 Windows 不合法字元與空白。"""
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_-]", "_", text)).strip("_")


def day_name(when: Optional[datetime] = None) -> str:
    return (when or datetime.now()).strftime(DAY_FMT)


def ensure_day_dirs(site_root: Path,
                    when: Optional[datetime] = None) -> List[Path]:
    """建好今天與**明天**的資料夾,回傳兩者。

    要先建明天的:輸出路徑交給 ffmpeg 的 strftime 展開,午夜一到它就直接
    往新日期的路徑寫。那個資料夾若還不存在,ffmpeg 會開檔失敗而中斷錄影
    —— 正好在無人看顧的半夜。
    """
    now = when or datetime.now()
    dirs = [site_root / day_name(now),
            site_root / day_name(now + timedelta(days=1))]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def prune_days(site_root: Path, keep: int = DEFAULT_KEEP_DAYS,
               dry_run: bool = False,
               when: Optional[datetime] = None) -> List[Path]:
    """只保留最新的 keep 個日期資料夾,其餘刪除;回傳被刪的清單。

    這個函式會**遞迴刪除目錄**,所以每一條限制都是刻意的:

    - 只看 `site_root` 的直接子項,不遞迴尋找
    - 名字必須**剛好**符合 8 位數字(`DAY_RE`)。任何其他東西——設定檔、
      筆記、手動整理的資料夾——一律不碰
    - 跳過符號連結(不跟著連結刪到別的地方去)
    - `keep < 1` 直接拒絕:那等於要求清空,不會是有意的
    - **未來日期的資料夾不算進額度也不刪**:`ensure_day_dirs()` 會預建
      明天,那個空資料夾若佔掉一格,`keep=3` 實際只留得到 2 天錄影
    """
    if keep < 1:
        raise ValueError(f"keep 至少要 1,收到 {keep}")
    if not site_root.is_dir():
        return []
    days = sorted(
        (d for d in site_root.iterdir()
         if d.is_dir() and not d.is_symlink() and DAY_RE.match(d.name)),
        key=lambda d: d.name)
    # 未來日期(ensure_day_dirs 預建的明天)不算進額度,也絕不刪。
    # 不排除的話,那個空資料夾會佔掉一格:keep=3 實際只留得到 2 天錄影
    today = day_name(when or datetime.now())
    future = [d for d in days if d.name > today]
    past = [d for d in days if d.name <= today]
    doomed = past[:-keep] if keep < len(past) else []
    del future
    if not dry_run:
        for d in doomed:
            shutil.rmtree(d, ignore_errors=True)
    return doomed


def build_command(media_url: str, out_template: str, ffmpeg: str,
                  segment_sec: int = DEFAULT_SEGMENT_SEC,
                  audio: bool = False) -> List[str]:
    """組出「原樣複製 + 定時切段」的 ffmpeg 命令。

    `-c copy` 是關鍵:不解碼、不重編碼,所以不可能掉幀,CPU 也接近零。
    `-strftime 1` 讓 ffmpeg 自己把輸出路徑裡的 %Y%m%d 展開,跨日不必重啟。
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin"]
    if media_url.lower().startswith(("http://", "https://")):
        # 斷線自動重連(這幾個選項只對 http(s) 輸入有效,且必須放在 -i 前)
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10"]
    cmd += ["-i", media_url, "-c", "copy"]
    if not audio:
        # 預設不錄音:對動作分析沒有用處,而且公共場所的談話內容
        # 比影像更敏感。要錄請明確指定 --audio
        cmd += ["-an"]
    cmd += ["-f", "segment",
            "-segment_time", str(int(segment_sec)),
            "-segment_format", "mpegts",
            "-reset_timestamps", "1",
            "-strftime", "1",
            out_template]
    return cmd


class StreamRecorder:
    """看顧一支 ffmpeg,斷了就重連,並維護資料夾與保留天數。"""

    def __init__(self, url: str, root: str = DEFAULT_ROOT,
                 segment_sec: int = DEFAULT_SEGMENT_SEC,
                 keep_days: int = DEFAULT_KEEP_DAYS,
                 max_height: int = DEFAULT_MAX_HEIGHT,
                 audio: bool = False, restart_sec: float = 5.0,
                 log=print):
        self.url = url
        self.slug = site_slug(url)
        self.site_root = Path(root) / self.slug
        self.segment_sec = int(segment_sec)
        self.keep_days = int(keep_days)
        self.max_height = int(max_height)
        self.audio = audio
        self.restart_sec = float(restart_sec)
        self.log = log
        self.ffmpeg = ffmpeg_exe()
        self.log_path = self.site_root / "ffmpeg.log"
        self._stop = False
        self.restarts = 0

    LOG_MAX_BYTES = 5_000_000

    def _rotate_log(self) -> None:
        """ffmpeg 的紀錄檔滿了就輪替一份,免得長年累月長到幾百 MB。"""
        try:
            if self.log_path.exists() and \
                    self.log_path.stat().st_size > self.LOG_MAX_BYTES:
                old = self.log_path.with_suffix(".log.old")
                old.unlink(missing_ok=True)
                self.log_path.rename(old)
        except OSError:
            pass                      # 紀錄檔輪替失敗不該影響錄影

    # ---- 單次連線 ----

    def _media_url(self) -> str:
        """取得可直接餵給 ffmpeg 的網址。

        每次(重)連都重新解析:yt-dlp 給的 manifest 網址會過期,沿用舊的
        會一直重連失敗,而且錯誤看起來像網路問題。
        """
        if is_youtube_url(self.url):
            return resolve_youtube(self.url, self.max_height)
        return self.url

    def _out_template(self) -> str:
        # ffmpeg 的 strftime 展開:午夜自動換到新的日期資料夾
        return str(self.site_root / "%Y%m%d" / "%Y%m%d_%H%M%S.ts")

    def stop(self) -> None:
        self._stop = True

    # ---- 主迴圈 ----

    def run(self, duration_sec: Optional[float] = None) -> int:
        """持續錄影直到 stop() 或 duration_sec 到期,回傳重啟次數。"""
        t0 = time.time()
        self.site_root.mkdir(parents=True, exist_ok=True)
        self.log(f"[錄影] 來源 {self.url}")
        self.log(f"[錄影] 存放 {self.site_root}  每段 {self.segment_sec}s  "
                 f"保留 {self.keep_days} 天")
        # 空間預估先講在前面:720p 實測約 21 MB/分 ≈ 30 GB/天,保留三天
        # 就要 90 GB。錄到一半磁碟滿了,ffmpeg 只會安靜地停住
        try:
            free = shutil.disk_usage(self.site_root).free / 2**30
            need = 30.0 * self.keep_days
            self.log(f"[錄影] 磁碟可用 {free:.0f} GB;720p 約 30 GB/天,"
                     f"保留 {self.keep_days} 天約需 {need:.0f} GB"
                     + ("  ⚠ 空間可能不足" if free < need else ""))
        except OSError:
            pass
        while not self._stop:
            self._housekeep()
            try:
                media = self._media_url()
            except Exception as e:
                self.log(f"[錄影] 解析網址失敗:{e},{self.restart_sec:.0f} 秒後重試")
                if self._sleep(self.restart_sec, t0, duration_sec):
                    break
                continue

            cmd = build_command(media, self._out_template(), self.ffmpeg,
                                self.segment_sec, self.audio)
            # stderr 一定要導進檔案,**不可以**用 subprocess.PIPE。
            # 沒人讀的 PIPE 只有 64 KB,HLS 的警告(每換一次 CDN 主機就一行)
            # 很快塞爆它,ffmpeg 就卡在寫 stderr 上不動了 —— 實測錄了 118 秒
            # 只寫出 19 秒、檔案剛好停在 4 MiB,而且行程還活著,看起來像
            # 「正在錄」。導進檔案沒有大小限制,也順便留下可查的紀錄。
            self._rotate_log()
            with open(self.log_path, "a", encoding="utf-8", errors="replace") \
                    as errlog:
                errlog.write(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"{' '.join(cmd[-6:])} ===\n")
                errlog.flush()
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL,
                                        stderr=errlog)
                self.log(f"[錄影] 開始({time.strftime('%H:%M:%S')})")
                # 每 30 秒醒來一次:維護資料夾(跨日要先備好明天的)與保留天數
                while proc.poll() is None and not self._stop:
                    if self._sleep(30.0, t0, duration_sec):
                        break
                    self._housekeep()
                if proc.poll() is None:
                    self._terminate(proc)
                    break
            self.restarts += 1
            self.log(f"[錄影] 連線中斷(第 {self.restarts} 次)"
                     f",詳見 {self.log_path}")
            if self._sleep(self.restart_sec, t0, duration_sec):
                break
        self.log(f"[錄影] 結束,共重連 {self.restarts} 次")
        return self.restarts

    def _housekeep(self) -> None:
        ensure_day_dirs(self.site_root)
        for gone in prune_days(self.site_root, self.keep_days):
            self.log(f"[錄影] 已刪除逾期資料夾 {gone.name}")

    def _sleep(self, seconds: float, t0: float,
               duration_sec: Optional[float]) -> bool:
        """分段小睡,回傳「是否該收工」(可被 stop() 或總時長打斷)。"""
        end = time.time() + seconds
        while time.time() < end:
            if self._stop:
                return True
            if duration_sec is not None and time.time() - t0 >= duration_sec:
                self._stop = True
                return True
            time.sleep(min(0.5, end - time.time()))
        return False

    def _terminate(self, proc) -> None:
        """收掉 ffmpeg。`.ts` 就算被砍在半途也還播得動。"""
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---- 檢查錄下來的東西 ----

def probe_duration(path: Path, ffmpeg: Optional[str] = None) -> float:
    """用 ffmpeg 讀出一個檔案的實際長度(秒);讀不到回 0。"""
    ff = ffmpeg or ffmpeg_exe()
    out = subprocess.run([ff, "-hide_banner", "-i", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)",
                  out.stderr.decode("utf-8", "replace"))
    if not m:
        return 0.0
    h, mm, s = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(s)


def day_stats(day_dir: Path, ffmpeg: Optional[str] = None) -> dict:
    """一天的錄影統計:段數、總長、位元組、以及**缺口**。

    缺口是「這一天實際錄到的秒數」與「第一段開始到最後一段結束的時間跨度」
    的差:錄影本身不會掉幀,但連線中斷期間的內容是真的不存在,這個數字
    就是在量那個。
    """
    ff = ffmpeg or ffmpeg_exe()
    files = sorted(p for p in day_dir.glob("*.ts") if p.is_file())
    if not files:
        return {"files": 0, "seconds": 0.0, "bytes": 0, "span": 0.0,
                "gap": 0.0}
    total = sum(probe_duration(p, ff) for p in files)
    stamps = []
    for p in files:
        m = re.search(r"(\d{8})_(\d{6})", p.name)
        if m:
            stamps.append(datetime.strptime(m.group(0), "%Y%m%d_%H%M%S"))
    span = ((max(stamps) - min(stamps)).total_seconds()
            + probe_duration(files[-1], ff)) if stamps else total
    return {"files": len(files), "seconds": total,
            "bytes": sum(p.stat().st_size for p in files),
            "span": span, "gap": max(0.0, span - total)}
