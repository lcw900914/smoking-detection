"""M1 節點正規化(無參數):消除攝影機距離、人物位置與解析度的影響。

輸入:節點序列 (T, 17, 3) — 影像座標 x, y 與置信度 conf
輸出:每幀特徵 (T, 85) = 17×2 正規化座標 + 17×2 速度 + 17 conf

正規化定義(與第一階段 SkeletonStageEstimator 的尺度語意一致):
  中心 = 雙肩中點(缺則有效節點質心)
  尺度 = max(肩寬, 0.55×軀幹高)(側面視角肩寬被壓縮,取大者)
  conf < conf_thresh 的節點座標置 0(遮擋),conf 保留為輸入通道
"""
from typing import Optional

import numpy as np

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
FEATURE_DIM = 17 * 2 + 17 * 2 + 17  # 座標 + 速度 + conf = 85


def _frame_center_scale(kpts: np.ndarray, conf_thresh: float):
    """單幀的中心與尺度;肩不可見時退回有效節點質心與外接框。"""
    vis = kpts[:, 2] >= conf_thresh
    sho_ok = vis[L_SHO] and vis[R_SHO]
    if sho_ok:
        center = (kpts[L_SHO, :2] + kpts[R_SHO, :2]) / 2
        scale = float(np.linalg.norm(kpts[L_SHO, :2] - kpts[R_SHO, :2]))
        if vis[L_HIP] and vis[R_HIP]:
            mid_hip = (kpts[L_HIP, :2] + kpts[R_HIP, :2]) / 2
            scale = max(scale, 0.55 * float(np.linalg.norm(center - mid_hip)))
    elif vis.any():
        pts = kpts[vis, :2]
        center = pts.mean(axis=0)
        span = pts.max(axis=0) - pts.min(axis=0)
        scale = float(max(span.max(), 1.0))
    else:
        return None, None
    return center, max(scale, 1e-3)


def normalize_sequence(seq: np.ndarray, conf_thresh: float = 0.3,
                       fps: float = 10.0) -> np.ndarray:
    """(T, 17, 3) → (T, 85) 正規化特徵序列。

    - 座標:逐幀減中心、除尺度;低置信節點置 0
    - 速度:一階差分(第 0 幀為 0)——正規化後才差分,
      使速度同樣具距離不變性
    - conf:原樣保留,模型自行學會忽略不可靠節點
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    coords = np.zeros((T, 17, 2), dtype=np.float32)
    confs = seq[:, :, 2].astype(np.float32)

    for t in range(T):
        center, scale = _frame_center_scale(seq[t], conf_thresh)
        if center is None:
            continue  # 整幀無效:座標留 0
        vis = seq[t, :, 2] >= conf_thresh
        coords[t, vis] = (seq[t, vis, :2] - center) / scale

    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    # 兩側任一幀節點不可見 → 該速度不可信,置 0
    vis_pair = (confs[1:] >= conf_thresh) & (confs[:-1] >= conf_thresh)
    vel[1:][~vis_pair] = 0.0

    return np.concatenate([
        coords.reshape(T, -1), vel.reshape(T, -1), confs
    ], axis=1)  # (T, 85)
