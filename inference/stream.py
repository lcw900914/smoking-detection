"""影像來源抽象:檔案 / 本機攝影機 / RTSP 串流,統一介面。

RTSP 的實務處理:
- 憑證特殊字元自動 percent-encode(密碼含 @ 是最常見的坑,
  例如 rtsp://admin:@admin888@host → 密碼 "@admin888" 會被誤切)
- 強制 TCP 傳輸(預設 UDP 掉包會花屏)與連線逾時
- 背景讀取執行緒「只保留最新影格」:推理速度跟不上串流時
  直接丟舊幀,延遲不會累積
- 斷線自動重連(含退避)
"""
import os
import re
import threading
import time
from typing import Optional, Tuple
from urllib.parse import quote, unquote

import cv2
import numpy as np


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


class VideoSource:
    """統一影像來源。

    - 檔案:`read()` 逐幀回傳,時間戳 = 幀序 / fps
    - 攝影機 / RTSP(is_live=True):背景執行緒持續讀取,
      `read()` 回傳「最新」影格與牆鐘時間戳;斷線自動重連

    Args:
        src: 檔案路徑 / 攝影機編號字串 / rtsp:// URL
        use_tcp: RTSP 走 TCP(建議開)
        connect_timeout_sec: RTSP 連線/讀取逾時
        reconnect_sec: 斷線重連間隔
    """

    def __init__(self, src, use_tcp: bool = True,
                 connect_timeout_sec: float = 5.0,
                 reconnect_sec: float = 3.0):
        self.reconnect_sec = reconnect_sec
        self._stop = False
        self._lock = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._last_returned = 0
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
        elif str(src).isdigit():
            self.kind = "webcam"
            self.src = int(src)
        else:
            self.kind = "file"
            self.src = str(src)

        self.is_live = self.kind in ("rtsp", "webcam")
        self.cap = self._open()
        if not self.cap.isOpened():
            raise RuntimeError(f"無法開啟影像來源:{src}"
                               +("(請確認網路、帳密與路徑;密碼含特殊字元"
                                  "已自動編碼)" if self.kind == "rtsp" else ""))

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # 串流常回報 0 或異常值,fallback 25
        self.fps = fps if 1.0 <= fps <= 120.0 else 25.0

        self._file_idx = 0
        if self.is_live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        if self.kind == "rtsp":
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
                self.cap = self._open()
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._lock.notify_all()

    # ---------- 對外介面 ----------

    def read(self, timeout: float = 5.0
             ) -> Tuple[Optional[np.ndarray], float]:
        """讀一幀。回傳 (frame, timestamp);結束/逾時回傳 (None, ts)。

        - live:阻塞等待「新」影格(跳過已回傳過的),時間戳為牆鐘
        - file:逐幀,時間戳 = 幀序 / fps
        """
        if self.is_live:
            with self._lock:
                if not self._lock.wait_for(
                        lambda: self._frame_id > self._last_returned
                        or self._stop,
                        timeout=timeout):
                    return None, time.time() - self._t0  # 逾時
                if self._stop or self._frame is None:
                    return None, time.time() - self._t0
                self._last_returned = self._frame_id
                return self._frame.copy(), time.time() - self._t0

        ok, frame = self.cap.read()
        ts = self._file_idx / self.fps
        self._file_idx += 1
        return (frame if ok else None), ts

    def release(self) -> None:
        self._stop = True
        if self.is_live:
            with self._lock:
                self._lock.notify_all()
            self._thread.join(timeout=2.0)
        self.cap.release()
