"""端到端 demo:webcam 或影片檔即時偵測。

用法(在專案根目錄執行):
    python scripts/demo.py --source 0                       # webcam,M1 骨架
    python scripts/demo.py --source video.mp4 --ckpt ckpt.pt --save out.mp4
"""
import argparse
import sys
from pathlib import Path

# 讓 scripts/ 下可直接執行(把專案根目錄加入 path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.pipeline import SmokingDetectionPipeline  # noqa: E402
from utils import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="抽菸偵測 demo")
    parser.add_argument("--source", default="0", help="影片路徑或攝影機編號")
    parser.add_argument("--ckpt", default=None, help="模型權重;不給則只跑偵測+追蹤")
    parser.add_argument("--save", default=None, help="輸出 mp4 路徑")
    parser.add_argument("--infer-config", default="configs/inference.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()

    use_model = args.ckpt is not None
    infer_cfg = load_config(args.infer_config)
    model_cfg = load_config(args.model_config) if use_model else None

    pipeline = SmokingDetectionPipeline(
        infer_cfg, model_cfg, ckpt_path=args.ckpt, use_model=use_model)
    pipeline.run(args.source, display=True, save_video=args.save)


if __name__ == "__main__":
    main()
