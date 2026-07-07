"""共用工具:設定檔載入、隨機種子、VRAM 保護檢查。"""
import os
import random
import warnings

import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    """讀取 yaml 設定檔,回傳 dict。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """固定所有隨機來源,確保實驗可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """解析裝置字串;auto 時優先使用 CUDA。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def check_vram_budget(estimated_gb: float, warn_gb: float = 4.0,
                      context: str = "") -> bool:
    """VRAM 保護:估計佔用超過 warn_gb 時發出警告並提示降級選項。

    回傳 True 表示在預算內,False 表示超標(呼叫端可據此降級)。
    """
    if estimated_gb <= warn_gb:
        return True
    warnings.warn(
        f"[VRAM 警告] {context} 估計佔用約 {estimated_gb:.1f} GB,"
        f"超過預算 {warn_gb:.1f} GB(RTX 2060 僅 6GB)。\n"
        f"降級選項:(1) 縮小 batch size;(2) 開啟 AMP fp16;"
        f"(3) 改用離線特徵(extract_features.py)訓練時序頭。",
        stacklevel=2,
    )
    return False


def count_parameters(model: torch.nn.Module) -> int:
    """計算可訓練參數量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- 影像檔讀寫(支援含中文等非 ASCII 的 Windows 路徑) ----------
# cv2.imread / cv2.imwrite 在 Windows 上無法處理非 ASCII 路徑,
# 全專案的圖片檔讀寫一律走以下兩個函式。

def imread(path) -> "np.ndarray | None":
    """讀取影像(BGR);失敗回傳 None。"""
    import cv2
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path, img, jpg_quality: int = 95) -> bool:
    """寫出影像;依副檔名編碼。回傳是否成功。"""
    import cv2
    ext = os.path.splitext(str(path))[1] or ".jpg"
    params = ([cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
              if ext.lower() in (".jpg", ".jpeg") else [])
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True
