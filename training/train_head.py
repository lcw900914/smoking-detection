"""階段一:凍結 backbone,只訓練時序頭 + 融合層(讀離線特徵)。

- 資料:FeatureClipDataset(extract_features.py 輸出的 .npy)
- 完全不碰影像與 backbone,batch 256,VRAM 佔用極低
- 支援 resume(--resume ckpt 路徑)、tensorboard logging
- 支援 KD:config 中 distill.enabled=true 且 teacher_root 有預計算檔

用法:
    python -m training.train_head --train-features datasets/features/train \
        --val-features datasets/features/val
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.full_model import build_model
from data.feature_dataset import FeatureClipDataset
from training.losses import TotalLoss, FeatureKD
from training.common import (build_optimizer, build_scheduler,
                             save_checkpoint, load_checkpoint)
from utils import (load_config, set_seed, resolve_device,
                   check_vram_budget, count_parameters)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """驗證:視窗階段 accuracy 與 clip 二元 accuracy。"""
    model.eval()
    stage_correct = clip_correct = total = 0
    for batch in loader:
        out = model.forward_features(batch["short_feats"].to(device),
                                     batch["long_feats"].to(device))
        stage_pred = out["stage_logits"].argmax(1).cpu()
        clip_pred = (torch.sigmoid(out["fusion_logit"]).cpu() > 0.5).float()
        stage_correct += (stage_pred == batch["stage_label"]).sum().item()
        clip_correct += (clip_pred == batch["clip_label"]).sum().item()
        total += len(stage_pred)
    return {"stage_acc": stage_correct / max(1, total),
            "clip_acc": clip_correct / max(1, total)}


def main():
    parser = argparse.ArgumentParser(description="階段一:訓練時序頭")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--val-features", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint 路徑")
    parser.add_argument("--tag", default="head", help="實驗名稱")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)
    cfg = train_cfg["head"]
    set_seed(train_cfg.get("seed", 42))
    device = resolve_device("auto")

    buf = model_cfg["buffer"]
    kd_cfg = train_cfg.get("distill", {})
    kd_root = (train_cfg["paths"]["teacher_root"]
               if kd_cfg.get("enabled") else None)

    train_ds = FeatureClipDataset(
        args.train_features,
        short_T=buf["short"]["T"], long_T=buf["long"]["T"],
        long_stride=buf["long"]["stride"], train=True, kd_root=kd_root)
    val_ds = FeatureClipDataset(
        args.val_features,
        short_T=buf["short"]["T"], long_T=buf["long"]["T"],
        long_stride=buf["long"]["stride"], train=False)

    # pin_memory 關閉:特徵樣本大(數十 MB/batch),多 worker 預取的
    # 鎖頁記憶體會在 Windows WDDM 下觸發 cudaErrorAlreadyMapped
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(model_cfg).to(device)
    model.freeze_backbone()  # 階段一不碰 backbone(其實根本不會 forward 它)
    print(f"[train_head] 時序頭+融合層參數量:"
          f"{sum(p.numel() for p in model.head_parameters()):,}")

    # 特徵 batch 256:約 (256, 2048, 28, 28) fp32 ≈ 1.6GB,在預算內
    check_vram_budget(2.5, train_cfg.get("vram_warn_gb", 4.0),
                      context=f"train_head batch {cfg['batch_size']}")

    loss_weights = dict(cfg["loss_weights"])
    feature_kd = None
    if kd_root:
        loss_weights.update(kd_cfg["loss_weights"])
        feature_kd = FeatureKD(
            student_dim=model_cfg["temporal_head"]["mid_channels"],
            teacher_dim=kd_cfg["teacher_feature_dim"]).to(device)
    criterion = TotalLoss(loss_weights,
                          kd_temperature=kd_cfg.get("temperature", 4.0),
                          feature_kd=feature_kd,
                          pos_weight=cfg.get("pos_weight")).to(device)

    params = list(model.head_parameters())
    if feature_kd is not None:
        params += list(feature_kd.parameters())
    optimizer = build_optimizer(params, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch, best = 0, 0.0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler,
                                device=str(device))
        start_epoch, best = state["epoch"] + 1, state["best_metric"]
        print(f"[train_head] resume 自 epoch {start_epoch}")

    ckpt_dir = Path(train_cfg["paths"]["ckpt_dir"])
    writer = SummaryWriter(
        Path(train_cfg["paths"]["log_dir"]) / args.tag)

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        model.backbone.eval()  # 凍結的 backbone 維持 eval(BN 統計不更新)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model.forward_features(batch["short_feats"],
                                         batch["long_feats"])
            losses = criterion(out, batch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()

            if step % 20 == 0:
                gstep = epoch * len(train_loader) + step
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), gstep)
                pbar.set_postfix(loss=f"{losses['total'].item():.4f}")
        scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        metrics = evaluate(model, val_loader, device)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[train_head] epoch {epoch}: {metrics}")

        save_checkpoint(ckpt_dir / f"{args.tag}_last.pt",
                        model, optimizer, scheduler, epoch, best)
        if metrics["clip_acc"] > best:
            best = metrics["clip_acc"]
            save_checkpoint(ckpt_dir / f"{args.tag}_best.pt",
                            model, optimizer, scheduler, epoch, best)
            print(f"[train_head] 新最佳 clip_acc={best:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
