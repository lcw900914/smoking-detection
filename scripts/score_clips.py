"""量測多段影片在串流管線下的最高 P_t(挑 demo 素材用)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from utils import load_config
from inference.pipeline import SmokingDetectionPipeline

SRC = Path("D:/datasets/hmdb51/videos/hmdb51")
CANDS = {
    "smoke/After_work_smoke_in_the_garage_smoke_h_nm_np1_fr_med_2.avi": "抽菸",
    "smoke/nice_smoking_girl_smoke_h_nm_np1_le_med_2.avi": "抽菸",
    "smoke/OSSER_-_Qualboro_light_-_Marlboro_Verarschung_smoke_h_cm_np1_le_bad_0.avi": "抽菸",
    "smoke/American_History_X_smoke_h_nm_np1_fr_goo_29.avi": "抽菸",
    "smoke/girl_smoking_a_cigarette_smoke_h_nm_np1_fr_med_0.avi": "抽菸",
    "drink/BATMAN_BEGINS_drink_h_nm_np1_fr_goo_9.avi": "喝",
    "drink/American_History_X_drink_h_nm_np1_fr_goo_46.avi": "喝",
    "drink/AllThePresidentMen_drink_h_nm_np1_fr_goo_5.avi": "喝",
    "drink/AmericanGangster_drink_u_nm_np1_fr_goo_67.avi": "喝",
    "chew/Big_League_Chew_chew_h_nm_np1_fr_goo_2.avi": "嚼",
    "chew/Blowing_Bubbles!_chew_h_nm_np1_fr_goo_2.avi": "嚼",
    "talk/jonhs_netfreemovies_holygrail_talk_h_nm_np1_fr_med_7.avi": "講話",
    "talk/jonhs_netfreemovies_holygrail_talk_u_nm_np1_le_med_17.avi": "講話",
}


def max_p(pipeline: SmokingDetectionPipeline, path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / 10))
    best, i = 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            for _, r in pipeline.step(frame, i / fps).items():
                best = max(best, r["P"])
        i += 1
    cap.release()
    return best


def main():
    icfg = load_config("configs/inference_hmdb.yaml")
    mcfg = load_config("configs/model.yaml")
    for rel, lab in CANDS.items():
        p = SmokingDetectionPipeline(
            icfg, mcfg, ckpt_path="checkpoints/hmdb_e2e_best.pt")
        print(f"{lab}\t最高P={max_p(p, SRC / rel):.3f}\t{rel}")


if __name__ == "__main__":
    main()
