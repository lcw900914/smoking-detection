"""直播錄影 CLI:把 YouTube / m3u8 直播存成每日資料夾裡的分段 .ts。

錄影全程不解碼(ffmpeg -c copy),所以不會掉幀、CPU 幾乎是零,
也就不會有即時管線那種「跟不上就掉幀」的問題。細節見 inference/recorder.py。

用法:
    python scripts/record_stream.py "https://www.youtube.com/watch?v=xxxx"
    python scripts/record_stream.py <url> --keep-days 3 --segment-sec 600
    python scripts/record_stream.py <url> --hours 8        # 錄 8 小時後自動停
    python scripts/record_stream.py <url> --stats          # 只看已錄內容的統計
    python scripts/record_stream.py <url> --prune-dry-run  # 只列出會刪哪些

Ctrl-C 可隨時中止;`.ts` 就算被砍在半途也還播得動。
"""
import argparse
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from inference.recorder import (DEFAULT_KEEP_DAYS,  # noqa: E402
                                DEFAULT_ROOT, DEFAULT_SEGMENT_SEC,
                                StreamRecorder, day_stats, prune_days,
                                site_slug)


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


def cmd_stats(site_root: Path) -> int:
    if not site_root.is_dir():
        print(f"[統計] 還沒有任何錄影:{site_root}")
        return 1
    days = sorted(d for d in site_root.iterdir() if d.is_dir())
    if not days:
        print(f"[統計] {site_root} 底下沒有日期資料夾")
        return 1
    print(f"[統計] {site_root}")
    print("  日期     | 段數 | 實錄時長  | 涵蓋跨度  | 缺口     | 大小")
    for d in days:
        s = day_stats(d)
        print(f"  {d.name} | {s['files']:4d} | "
              f"{s['seconds']/3600:6.2f} 小時 | "
              f"{s['span']/3600:6.2f} 小時 | "
              f"{s['gap']/60:6.1f} 分 | {_human(s['bytes'])}")
    print("  缺口 = 涵蓋跨度 - 實錄時長,也就是連線中斷期間真正缺掉的內容。"
          "\n  錄影本身不解碼,不會有『來不及處理而掉幀』的損失。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="直播錄影(不解碼,不掉幀)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("url", help="YouTube 網址或 m3u8 直播網址")
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"存放根目錄(預設 {DEFAULT_ROOT})")
    ap.add_argument("--segment-sec", type=int, default=DEFAULT_SEGMENT_SEC,
                    help=f"每段秒數(預設 {DEFAULT_SEGMENT_SEC})")
    ap.add_argument("--keep-days", type=int, default=DEFAULT_KEEP_DAYS,
                    help=f"保留幾天(預設 {DEFAULT_KEEP_DAYS},超過自動刪除)")
    ap.add_argument("--max-height", type=int, default=720,
                    help="YouTube 畫質上限(預設 720)")
    ap.add_argument("--audio", action="store_true",
                    help="連音訊一起錄(預設不錄:對動作分析無用,"
                         "且公共場所談話比影像敏感)")
    ap.add_argument("--hours", type=float, default=None,
                    help="錄滿幾小時後自動停止(預設一直錄)")
    ap.add_argument("--stats", action="store_true",
                    help="只顯示已錄內容的統計,不錄影")
    ap.add_argument("--prune-dry-run", action="store_true",
                    help="只列出保留天數會刪掉哪些資料夾,不實際刪除")
    args = ap.parse_args()

    site_root = Path(args.root) / site_slug(args.url)

    if args.stats:
        return cmd_stats(site_root)

    if args.prune_dry_run:
        doomed = prune_days(site_root, args.keep_days, dry_run=True)
        if not doomed:
            print(f"[保留] {site_root} 沒有超出 {args.keep_days} 天的資料夾")
        for d in doomed:
            print(f"[保留] 會刪除 {d}")
        return 0

    rec = StreamRecorder(
        args.url, root=args.root, segment_sec=args.segment_sec,
        keep_days=args.keep_days, max_height=args.max_height,
        audio=args.audio)

    def on_signal(_sig, _frm):
        print("\n[錄影] 收到中止訊號,收尾中…")
        rec.stop()
    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except (AttributeError, ValueError):
        pass                    # Windows 上不一定有 SIGTERM

    rec.run(duration_sec=args.hours * 3600 if args.hours else None)
    cmd_stats(site_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
