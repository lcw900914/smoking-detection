"""建立 demo 測試影片:單片複製 + 蒙太奇串接。

蒙太奇設計重點(踩過的坑):
- 片段間插 4 秒黑幕(> track 回收 3 秒),讓每段的緩衝/EMA/track 完全重置,
  避免跨片段污染
- 全部原生尺寸置中、只補黑邊(縮放會改變模型分數)
- 用 ffmpeg x264 crf 12 近無損編碼(cv2 的 mp4v 壓縮失真會改變模型分數)

用法:python scripts/make_demo_videos.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = Path("D:/datasets/hmdb51/videos/hmdb51")
OUT = Path("demo_videos")
W, H = 560, 240   # 所選片段最大寬度
GAP_SEC = 4.0

SINGLES = {
    "抽菸_車庫9秒.avi":
        "smoke/After_work_smoke_in_the_garage_smoke_h_nm_np1_fr_med_2.avi",
    "抽菸_女子8秒.avi":
        "smoke/nice_smoking_girl_smoke_h_nm_np1_le_med_2.avi",
    "抽菸_特寫8秒.avi":
        "smoke/OSSER_-_Qualboro_light_-_Marlboro_Verarschung_smoke_h_cm_np1_le_bad_0.avi",
    "抽菸_困難樣本_模型會漏掉.avi":
        "smoke/American_History_X_smoke_h_nm_np1_fr_goo_29.avi",
    "喝飲料6秒.avi": "drink/BATMAN_BEGINS_drink_h_nm_np1_fr_goo_9.avi",
    "吃口香糖5秒.avi": "chew/Big_League_Chew_chew_h_nm_np1_fr_goo_1.avi",
    "講話6秒.avi":
        "talk/jonhs_netfreemovies_holygrail_talk_h_cm_np1_ri_med_14.avi",
}

MONTAGES = {
    "抽菸_蒙太奇.mp4": [
        "smoke/After_work_smoke_in_the_garage_smoke_h_nm_np1_fr_med_2.avi",
        "smoke/nice_smoking_girl_smoke_h_nm_np1_le_med_2.avi",
        "smoke/OSSER_-_Qualboro_light_-_Marlboro_Verarschung_smoke_h_cm_np1_le_bad_0.avi",
        "smoke/girl_smoking_a_cigarette_smoke_h_nm_np1_fr_med_0.avi",
    ],
    "混淆動作_蒙太奇.mp4": [
        "drink/BATMAN_BEGINS_drink_h_nm_np1_fr_goo_9.avi",
        "drink/AllThePresidentMen_drink_h_nm_np1_fr_goo_5.avi",
        "drink/AmericanGangster_drink_u_nm_np1_fr_goo_67.avi",
        "chew/Big_League_Chew_chew_h_nm_np1_fr_goo_2.avi",
        "chew/Blowing_Bubbles!_chew_h_nm_np1_fr_goo_2.avi",
        "talk/jonhs_netfreemovies_holygrail_talk_h_nm_np1_fr_med_7.avi",
    ],
}


def pad_center(frame: np.ndarray) -> np.ndarray:
    """原生尺寸置中,黑邊補滿(不縮放、不變形)。"""
    h, w = frame.shape[:2]
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = (W - w) // 2, (H - h) // 2
    canvas[y:y + h, x:x + w] = frame
    return canvas


def build_montage(name: str, rels: list) -> None:
    """以 ffmpeg x264 crf 12 近無損寫出串接影片。"""
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", "30", "-i", "-",
         "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p",
         str(OUT / name)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    black = np.zeros((H, W, 3), dtype=np.uint8)
    total = 0
    for k, rel in enumerate(rels):
        if k > 0:
            for _ in range(int(GAP_SEC * 30)):
                proc.stdin.write(black.tobytes())
                total += 1
        cap = cv2.VideoCapture(str(SRC / rel))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            proc.stdin.write(pad_center(f).tobytes())
            total += 1
        cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"{name}: {total / 30:.0f} 秒")


def main():
    OUT.mkdir(exist_ok=True)
    for name, rel in SINGLES.items():
        shutil.copy(SRC / rel, OUT / name)
        print("複製", name)
    for name, rels in MONTAGES.items():
        build_montage(name, rels)


if __name__ == "__main__":
    main()
