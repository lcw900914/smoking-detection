"""訓練共用工具:optimizer / scheduler 建立、checkpoint 存讀(resume)。"""
import math
import os
from pathlib import Path
from typing import Iterable, Optional

import torch


def build_optimizer(params: Iterable, cfg: dict,
                    lr: Optional[float] = None) -> torch.optim.Optimizer:
    """依 config 建立 optimizer(目前支援 adamw / sgd)。"""
    lr = lr if lr is not None else cfg["lr"]
    name = cfg.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr,
                                 weight_decay=cfg.get("weight_decay", 0.05))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=cfg.get("weight_decay", 5e-4))
    raise ValueError(f"未知 optimizer: {name}")


def cosine_warmup_lambda(epochs: int, warmup_epochs: int):
    """cosine schedule + 線性 warmup 的 LambdaLR 函式。"""
    def fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    """依 config 建立 scheduler(目前支援 cosine + warmup)。"""
    if cfg.get("scheduler", "cosine") == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            cosine_warmup_lambda(cfg["epochs"], cfg.get("warmup_epochs", 0)))
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


def save_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler,
                    epoch: int, best_metric: float,
                    extra: Optional[dict] = None) -> None:
    """儲存 checkpoint(支援 resume 所需的全部狀態)。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, scheduler=None,
                    device: str = "cpu") -> dict:
    """讀取 checkpoint;回傳 {"epoch", "best_metric"} 供 resume。"""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {"epoch": ckpt.get("epoch", 0),
            "best_metric": ckpt.get("best_metric", 0.0)}
