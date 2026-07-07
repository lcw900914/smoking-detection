"""階段二:端到端微調(解凍 backbone,AMP 混合精度)。

- 資料:SmokingClipDataset(讀 jpg 序列)
- batch 8 + gradient accumulation ×4(有效 batch 32)
- backbone lr = head lr × 0.1(差分學習率)
- torch.cuda.amp 混合精度,適配 6GB VRAM
- 支援 resume、tensorboard logging

用法:
    python -m training.train_e2e --train-data datasets/processed/train \
        --val-data datasets/processed/val --init-ckpt checkpoints/head_best.pt
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.full_model import build_model
from data.dataset import SmokingClipDataset
from training.losses import TotalLoss
from training.common import (build_optimizer, build_scheduler,
                             save_checkpoint, load_checkpoint)
from utils import load_config, set_seed, resolve_device, check_vram_budget


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    stage_correct = clip_correct = total = 0
    for batch in loader:
        out = model(batch["short_images"].to(device),
                    batch["long_images"].to(device))
        stage_pred = out["stage_logits"].argmax(1).cpu()
        clip_pred = (torch.sigmoid(out["fusion_logit"]).cpu() > 0.5).float()
        stage_correct += (stage_pred == batch["stage_label"]).sum().item()
        clip_correct += (clip_pred == batch["clip_label"]).sum().item()
        total += len(stage_pred)
    return {"stage_acc": stage_correct / max(1, total),
            "clip_acc": clip_correct / max(1, total)}


def main():
    parser = argparse.ArgumentParser(description="階段二:端到端微調")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--init-ckpt", default=None,
                        help="階段一權重(head_best.pt)")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tag", default="e2e")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)
    cfg = train_cfg["e2e"]
    set_seed(train_cfg.get("seed", 42))
    device = resolve_device("auto")

    buf = model_cfg["buffer"]
    aug = train_cfg.get("augment", {})
    train_ds = SmokingClipDataset(
        args.train_data,
        short_T=buf["short"]["T"], long_T=buf["long"]["T"],
        long_stride=buf["long"]["stride"],
        augment={"bbox_jitter": aug.get("bbox_jitter", 0.08),
                 "hflip": aug.get("hflip", 0.5),
                 "color_jitter": aug.get("color_jitter", 0.3)},
        samples_per_clip=2)
    val_ds = SmokingClipDataset(
        args.val_data,
        short_T=buf["short"]["T"], long_T=buf["long"]["T"],
        long_stride=buf["long"]["stride"], augment=None)

    # pin_memory 關閉:Windows WDDM 下大 batch 鎖頁記憶體易觸發 CUDA 錯誤
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=cfg["num_workers"],
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["num_workers"])

    model = build_model(model_cfg).to(device)
    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location=device,
                           weights_only=True)
        model.load_state_dict(state.get("model", state))
        print(f"[train_e2e] 已載入階段一權重:{args.init_ckpt}")
    model.unfreeze_backbone()

    # VRAM 估計:batch 8 × 32 幀 ResNet-18 前向+反向(AMP)約 3.5-4GB
    check_vram_budget(4.0, train_cfg.get("vram_warn_gb", 4.0),
                      context=f"train_e2e batch {cfg['batch_size']}"
                              f"(AMP={cfg.get('amp', True)})")

    criterion = TotalLoss(dict(cfg["loss_weights"]),
                          pos_weight=cfg.get("pos_weight")).to(device)

    # 差分學習率:backbone lr = head lr × backbone_lr_mult
    head_params = list(model.head_parameters())
    optimizer = build_optimizer([
        {"params": model.backbone.parameters(),
         "lr": cfg["lr"] * cfg["backbone_lr_mult"]},
        {"params": head_params, "lr": cfg["lr"]},
    ], cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accum = cfg.get("grad_accum", 1)

    start_epoch, best = 0, 0.0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler,
                                device=str(device))
        start_epoch, best = state["epoch"] + 1, state["best_metric"]
        print(f"[train_e2e] resume 自 epoch {start_epoch}")

    ckpt_dir = Path(train_cfg["paths"]["ckpt_dir"])
    writer = SummaryWriter(Path(train_cfg["paths"]["log_dir"]) / args.tag)

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(batch["short_images"].to(device),
                            batch["long_images"].to(device))
                losses = criterion(out, {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()})
                loss = losses["total"] / accum
            scaler.scale(loss).backward()

            # gradient accumulation:每 accum 步更新一次
            if (step + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 20 == 0:
                gstep = epoch * len(train_loader) + step
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), gstep)
                pbar.set_postfix(loss=f"{losses['total'].item():.4f}")
        scheduler.step()
        writer.add_scalar("train/lr", scheduler.get_last_lr()[-1], epoch)

        metrics = evaluate(model, val_loader, device)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        print(f"[train_e2e] epoch {epoch}: {metrics}")

        save_checkpoint(ckpt_dir / f"{args.tag}_last.pt",
                        model, optimizer, scheduler, epoch, best)
        if metrics["clip_acc"] > best:
            best = metrics["clip_acc"]
            save_checkpoint(ckpt_dir / f"{args.tag}_best.pt",
                            model, optimizer, scheduler, epoch, best)
            print(f"[train_e2e] 新最佳 clip_acc={best:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
