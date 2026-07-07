"""Teacher 離線推理:對全訓練集推理一遍,存 soft labels 與中間層特徵。

Teacher:pytorchvideo X3D-M(Kinetics-400 預訓練;建議先在抽菸資料微調)。
存檔為 fp16 .npy,student 訓練時直接讀檔——teacher 與 student
絕不同時佔用 VRAM(6GB 限制下的關鍵設計)。

輸出(每 clip):
    {out}/{clip_id}_logits.npy   teacher 2 類 soft logits(smoking / 非)
    {out}/{clip_id}_feat.npy     指定中間層特徵(GAP 後向量,fp16)

用法:
    python -m training.distill_precompute --data datasets/processed/train \
        --out datasets/teacher --teacher-ckpt x3d_m_finetuned.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.dataset import SmokingClipDataset
from utils import load_config, resolve_device, check_vram_budget


def load_teacher(name: str = "x3d_m", ckpt: str = None,
                 num_classes: int = 2, device=None) -> nn.Module:
    """載入 pytorchvideo teacher 模型。

    ckpt 為 None 時使用 Kinetics 預訓練 + 隨機分類頭(僅供管線驗證;
    正式蒸餾前請先微調 teacher)。
    """
    model = torch.hub.load("facebookresearch/pytorchvideo", name,
                           pretrained=(ckpt is None))
    # 將 Kinetics 400 類分類頭換成 2 類(smoking / 非)
    head_proj = model.blocks[-1].proj
    model.blocks[-1].proj = nn.Linear(head_proj.in_features, num_classes)
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("model", state))
        print(f"[distill] 已載入微調後 teacher:{ckpt}")
    else:
        print("[distill] 警告:teacher 分類頭未微調,soft labels 不可靠,"
              "僅供管線驗證")
    return model.to(device).eval()


class _FeatureHook:
    """抓取 teacher 中間層輸出(GAP 成向量)。"""

    def __init__(self, module: nn.Module):
        self.feat = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # (B, C, T, H, W) → GAP → (B, C)
        if out.dim() == 5:
            self.feat = out.mean(dim=(2, 3, 4))
        elif out.dim() == 4:
            self.feat = out.mean(dim=(2, 3))
        else:
            self.feat = out


@torch.no_grad()
def precompute(data_root: str, out_root: str, teacher_name: str,
               teacher_ckpt: str, feature_block: int = 4,
               clip_T: int = 16, device: str = "auto") -> None:
    """對每個 clip 均勻取 clip_T 幀,teacher 一次前向,存 logits 與特徵。"""
    dev = resolve_device(device)
    # X3D-M 推理 batch 1 約 1.5GB,單獨執行,不與 student 同時
    check_vram_budget(2.0, context="X3D-M teacher 推理")

    teacher = load_teacher(teacher_name, teacher_ckpt, device=dev)
    hook = _FeatureHook(teacher.blocks[feature_block])

    ds = SmokingClipDataset(data_root, augment=None)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    for clip in tqdm(ds.clips, desc="teacher 推理"):
        n = clip["num_frames"]
        # 均勻取 clip_T 幀(涵蓋整個 clip 的時間範圍)
        idxs = np.linspace(0, n - 1, clip_T).round().astype(int)
        frames = torch.stack([
            torch.from_numpy(ds._load_frame(clip, int(j), None)
                             .transpose(2, 0, 1))
            for j in idxs
        ])  # (T, 3, H, W)
        # pytorchvideo 輸入格式 (B, 3, T, H, W)
        video = frames.permute(1, 0, 2, 3).unsqueeze(0).float().to(dev)

        logits = teacher(video)[0].cpu().numpy().astype(np.float16)
        feat = hook.feat[0].cpu().numpy().astype(np.float16)

        np.save(out / f"{clip['clip_id']}_logits.npy", logits)
        np.save(out / f"{clip['clip_id']}_feat.npy", feat)
    print(f"[distill] 完成,輸出:{out}")


def main():
    parser = argparse.ArgumentParser(description="teacher 離線推理預計算")
    parser.add_argument("--data", required=True, help="processed clip 目錄")
    parser.add_argument("--out", required=True, help="輸出目錄")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--teacher-ckpt", default=None,
                        help="微調後的 teacher 權重")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.train_config)["distill"]
    precompute(args.data, args.out, cfg.get("teacher", "x3d_m"),
               args.teacher_ckpt, device=args.device)


if __name__ == "__main__":
    main()
