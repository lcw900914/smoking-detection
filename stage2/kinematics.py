"""淺層特徵:身體座標系、圖節點特徵、可解釋運動學量。

兩條輸入流,刻意用不同的不變性:

  圖流 graph_features() —— 平移+尺度正規化,**不做旋轉對齊**。
      攝影機固定俯視,人朝哪邊站本身就是資訊(背對時腕-鼻距離無意義),
      旋轉掉會把這個資訊洗掉。給 ST-GCN 吃。

  運動學流 kinematic_features() —— 在**身體座標系**裡算角度與距離,
      對相機滾轉、人物側傾完全不變。這是使用者要的「手的相對角度」:
      手臂抬到哪、前臂指向哪、手腕在鼻子的哪個方位,全部是人自己的
      參考系,不是畫面的。淺層基元(舉手/放下)就從這裡判。

身體座標系定義:
    原點 = 雙肩中點
    尺度 = max(肩寬, 0.55×軀幹高)   ← 沿用第一階段定義,側面肩寬會被壓縮
    up   = 單位向量(肩中點 − 髖中點),髖不可見時退回畫面正上方
    right= up 順時針轉 90°
    某點的身體座標 = ((p − 原點) / 尺度) 投影到 (right, up)
    → by > 0 代表該點高於肩線
"""
from typing import Optional

import numpy as np

from inference.skeleton import estimate_orientation
from stage2.graph import (ARM_CHAIN, COCO_SUBSET, L_EAR, L_EYE, L_HIP,
                          L_SHO, NOSE, NUM_NODES, R_EAR, R_EYE, R_HIP,
                          R_SHO, SIDES)

# ---- 圖節點特徵通道 ----------------------------------------------------
GRAPH_CHANNELS = 5          # x, y, vx, vy, conf

# ---- 運動學特徵版面(每側 18 維 × 2 側 + 全域 7 維 = 43)----------------
K_D_NOSE = 0        # 腕-鼻距離(身體尺度正規化)
K_PHI_X = 1         # 腕相對鼻的方位單位向量(身體座標 right 分量)
K_PHI_Y = 2         # 同上 up 分量:>0 表示手高過鼻
K_D_EAR = 3         # 腕-耳距離(講電話的關鍵量)
K_D_EYE = 4         # 腕-眼距離(扶眼鏡)
K_H_WRI = 5         # 腕高(相對肩線,>0 高於肩)
K_H_OVER_EYE = 6    # 腕高 − 眼高(>0 表示手在眼睛以上 → 抓頭髮/戴帽)
K_LEN_UPPER = 7     # 上臂長 / 尺度(透視壓縮的指標:手往前伸會變短)
K_UPPER_X = 8       # 上臂方向(肩→肘)單位向量
K_UPPER_Y = 9
K_LEN_FORE = 10     # 前臂長 / 尺度
K_FORE_X = 11       # 前臂方向(肘→腕)單位向量
K_FORE_Y = 12
K_COS_ELBOW = 13    # 肘角餘弦:+1 完全彎曲,−1 完全打直
K_V_DNOSE = 14      # d(腕-鼻)/dt,>0 遠離、<0 接近(每秒)
K_V_H = 15          # d(腕高)/dt(每秒)
K_SPEED = 16        # 腕速度大小(身體尺度/秒)
K_VALID = 17        # 幾何可用:腕點可見 + 身體座標系成立
K_FACE_OK = 18      # 臉部量可用:另外要求鼻點可信
SIDE_DIM = 19

G_TILT_X = 0        # 軀幹傾斜(身體 up 相對畫面上方)
G_TILT_Y = 1
G_LOG_SCALE = 2     # log(身體尺度像素 / 100):遠近的代理量
G_FRONT = 3         # 朝向 one-hot
G_BACK = 4
G_UNKNOWN = 5
G_VALID = 6         # 本幀是否有可用的身體座標系
GLOBAL_DIM = 7

KIN_DIM = SIDE_DIM * 2 + GLOBAL_DIM      # 43
SIDE_SLICE = {"L": slice(0, SIDE_DIM),
              "R": slice(SIDE_DIM, 2 * SIDE_DIM)}
GLOBAL_SLICE = slice(2 * SIDE_DIM, KIN_DIM)

KIN_NAMES = [
    "腕鼻距", "方位x", "方位y", "腕耳距", "腕眼距", "腕高", "腕高減眼高",
    "上臂長", "上臂x", "上臂y", "前臂長", "前臂x", "前臂y", "肘角cos",
    "腕鼻距變率", "腕高變率", "腕速", "幾何有效", "臉部可用",
]

# 速度類特徵的平滑窗(幀)。姿態模型的關鍵點逐幀抖動 2–4 px,
# 直接差分算出來的腕速中位數會高到 1 個身體尺度/秒——比真實的手部
# 移動還大,「靜止」這個概念就沒了。先平滑再差分。
VEL_SMOOTH_WIN = 3


def smooth1d(x: np.ndarray, win: int = 3) -> np.ndarray:
    """邊緣延伸的移動平均;win 為奇數。"""
    if win <= 1:
        return x
    pad = win // 2
    xp = np.pad(np.asarray(x, np.float32), (pad, pad), mode="edge")
    k = np.ones(win, np.float32) / win
    return np.convolve(xp, k, mode="valid").astype(np.float32)


def _unit(v: np.ndarray, eps: float = 1e-6):
    """單位化 (…,2);長度過小回傳零向量與 0 長度。"""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / np.maximum(n, eps), 0.0), n[..., 0]


def body_frames(seq: np.ndarray, conf_thresh: float = 0.3):
    """逐幀身體座標系。

    Args:
        seq: (T, 17, 3) 或 (T, 13, 3) 影像座標關鍵點
    Returns:
        dict — origin (T,2)、scale (T,)、up (T,2)、right (T,2)、valid (T,)
    """
    seq = np.asarray(seq, dtype=np.float32)
    T = seq.shape[0]
    xy, conf = seq[:, :, :2], seq[:, :, 2]
    vis = conf >= conf_thresh

    origin = np.zeros((T, 2), np.float32)
    scale = np.ones((T,), np.float32)
    up = np.tile(np.array([0.0, -1.0], np.float32), (T, 1))
    valid = np.zeros((T,), bool)

    sho_ok = vis[:, L_SHO] & vis[:, R_SHO]
    hip_ok = vis[:, L_HIP] & vis[:, R_HIP]
    sho_mid = (xy[:, L_SHO] + xy[:, R_SHO]) / 2
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2

    origin[sho_ok] = sho_mid[sho_ok]
    sho_w = np.linalg.norm(xy[:, L_SHO] - xy[:, R_SHO], axis=1)
    torso = np.linalg.norm(sho_mid - hip_mid, axis=1)
    s = np.where(sho_ok & hip_ok, np.maximum(sho_w, 0.55 * torso), sho_w)
    scale[sho_ok] = np.maximum(s[sho_ok], 1e-3)
    valid |= sho_ok

    # 肩不可見:退回有效節點質心 + 外接尺度(勉強可用,valid 仍為 True
    # 但朝向會是 unknown,下游的側別特徵多半也算不出來)
    for t in np.where(~sho_ok)[0]:
        if not vis[t].any():
            continue
        pts = xy[t, vis[t]]
        origin[t] = pts.mean(axis=0)
        scale[t] = max(float((pts.max(0) - pts.min(0)).max()), 1e-3)
        valid[t] = True

    u, ulen = _unit(sho_mid - hip_mid)
    use = sho_ok & hip_ok & (ulen > 1e-3)
    up[use] = u[use]
    right = np.stack([-up[:, 1], up[:, 0]], axis=1)   # up 順時針 90°
    return {"origin": origin, "scale": scale, "up": up,
            "right": right, "valid": valid}


def to_body(seq: np.ndarray, frames: Optional[dict] = None,
            conf_thresh: float = 0.3):
    """關鍵點 → 身體座標 (T, V, 2);另回傳可見遮罩 (T, V)。"""
    seq = np.asarray(seq, dtype=np.float32)
    frames = body_frames(seq, conf_thresh) if frames is None else frames
    rel = (seq[:, :, :2] - frames["origin"][:, None, :]) \
        / frames["scale"][:, None, None]
    bx = (rel * frames["right"][:, None, :]).sum(-1)
    by = (rel * frames["up"][:, None, :]).sum(-1)
    return np.stack([bx, by], axis=-1), seq[:, :, 2] >= conf_thresh


def graph_features(seq: np.ndarray, conf_thresh: float = 0.3
                   ) -> np.ndarray:
    """ST-GCN 的節點特徵:(T, 13, 5) = 正規化 xy + 速度 + conf。

    平移與尺度正規化在**畫面座標軸**上做(不旋轉,理由見模組說明)。
    低置信節點座標歸零,conf 留作通道讓模型自己學會忽略。
    """
    seq = np.asarray(seq, dtype=np.float32)[:, COCO_SUBSET]
    frames = body_frames(seq, conf_thresh)
    vis = seq[:, :, 2] >= conf_thresh

    coords = (seq[:, :, :2] - frames["origin"][:, None, :]) \
        / frames["scale"][:, None, None]
    coords = np.where(vis[:, :, None], coords, 0.0).astype(np.float32)

    vel = np.zeros_like(coords)
    vel[1:] = coords[1:] - coords[:-1]
    pair = vis[1:] & vis[:-1]
    vel[1:] = np.where(pair[:, :, None], vel[1:], 0.0)

    return np.concatenate(
        [coords, vel, seq[:, :, 2:3]], axis=-1).astype(np.float32)


def _diff(x: np.ndarray, ok: np.ndarray, fps: float) -> np.ndarray:
    """一階差分 ×fps;兩側任一幀無效則為 0。"""
    d = np.zeros_like(x)
    d[1:] = (x[1:] - x[:-1]) * fps
    d[1:] = np.where(ok[1:] & ok[:-1], d[1:], 0.0)
    return d


def kinematic_features(seq: np.ndarray, fps: float = 10.0,
                       conf_thresh: float = 0.3,
                       nose_conf: float = 0.5) -> np.ndarray:
    """(T, 17|13, 3) → (T, 43) 可解釋運動學特徵。

    版面見模組頂部常數。所有距離以身體尺度正規化(遠近不變),
    所有方向以身體座標系表示(相機滾轉、人物側傾不變)。
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.shape[1] > NUM_NODES:
        full = seq                      # 朝向估計需要 COCO 原編號
        seq = seq[:, COCO_SUBSET]
    else:
        full = seq
    T = seq.shape[0]
    frames = body_frames(seq, conf_thresh)
    body, vis = to_body(seq, frames, conf_thresh)
    out = np.zeros((T, KIN_DIM), np.float32)

    nose_vis = seq[:, NOSE, 2] >= nose_conf
    eye_ok = vis[:, L_EYE] | vis[:, R_EYE]
    eye_mid = np.where(
        (vis[:, L_EYE] & vis[:, R_EYE])[:, None],
        (body[:, L_EYE] + body[:, R_EYE]) / 2,
        np.where(vis[:, L_EYE][:, None], body[:, L_EYE], body[:, R_EYE]))
    eye_mid = np.where(eye_ok[:, None], eye_mid, body[:, NOSE])

    for side in SIDES:
        sho_i, elb_i, wri_i = ARM_CHAIN[side]
        sl = SIDE_SLICE[side]
        blk = np.zeros((T, SIDE_DIM), np.float32)

        wri, elb, sho = body[:, wri_i], body[:, elb_i], body[:, sho_i]
        # 兩級有效性:幾何(手在哪、手臂什麼姿勢)只需要腕點與身體座標系;
        # 臉部相關量(腕-鼻/腕-耳距離)才另外要求鼻點可信。
        # 綁在一起會出事——俯視角有七成的幀鼻點不可信,若一併棄權,
        # 連「手放在腿上沒動」這種確定的事實都標不出來。
        ok = vis[:, wri_i] & frames["valid"]
        face_ok = ok & nose_vis

        # 腕-鼻:距離與方位(使用者要的「手的相對角度」)
        rel_nose = wri - body[:, NOSE]
        dir_nose, d_nose = _unit(rel_nose)
        blk[:, K_D_NOSE] = d_nose
        blk[:, K_PHI_X] = dir_nose[:, 0]
        blk[:, K_PHI_Y] = dir_nose[:, 1]

        # 腕-耳:同側優先,同側不可見取另一側(耳點常被頭遮),
        # 兩耳皆不可見時退回腕-鼻距離(不會憑空造出「手在耳邊」)
        ears = [(L_EAR if side == "L" else R_EAR),
                (R_EAR if side == "L" else L_EAR)]
        d_ear = d_nose.copy()
        for e in ears:
            de = np.linalg.norm(wri - body[:, e], axis=1)
            d_ear = np.where(vis[:, e], np.minimum(d_ear, de), d_ear)
        blk[:, K_D_EAR] = d_ear
        blk[:, K_D_EYE] = np.linalg.norm(wri - eye_mid, axis=1)

        blk[:, K_H_WRI] = wri[:, 1]
        blk[:, K_H_OVER_EYE] = wri[:, 1] - eye_mid[:, 1]

        arm_ok = vis[:, sho_i] & vis[:, elb_i] & vis[:, wri_i]
        upper, len_up = _unit(elb - sho)
        fore, len_fo = _unit(wri - elb)
        blk[:, K_LEN_UPPER] = np.where(arm_ok, len_up, 0)
        blk[:, K_UPPER_X] = np.where(arm_ok, upper[:, 0], 0)
        blk[:, K_UPPER_Y] = np.where(arm_ok, upper[:, 1], 0)
        blk[:, K_LEN_FORE] = np.where(arm_ok, len_fo, 0)
        blk[:, K_FORE_X] = np.where(arm_ok, fore[:, 0], 0)
        blk[:, K_FORE_Y] = np.where(arm_ok, fore[:, 1], 0)
        # 肘角:以肘為頂點,(肩−肘)與(腕−肘)的夾角餘弦
        blk[:, K_COS_ELBOW] = np.where(
            arm_ok, (-upper * fore).sum(axis=1), 0)

        # 速度一律先平滑再差分(見 VEL_SMOOTH_WIN 說明)
        blk[:, K_V_DNOSE] = _diff(
            smooth1d(blk[:, K_D_NOSE], VEL_SMOOTH_WIN), face_ok, fps)
        blk[:, K_V_H] = _diff(
            smooth1d(blk[:, K_H_WRI], VEL_SMOOTH_WIN), ok, fps)
        wri_s = np.stack([smooth1d(wri[:, 0], VEL_SMOOTH_WIN),
                          smooth1d(wri[:, 1], VEL_SMOOTH_WIN)], axis=1)
        sp = np.zeros(T, np.float32)
        sp[1:] = np.linalg.norm(wri_s[1:] - wri_s[:-1], axis=1) * fps
        blk[:, K_SPEED] = np.where(
            np.concatenate([[False], ok[1:] & ok[:-1]]), sp, 0)

        # 無效幀不留殘值:避免模型把幻覺數字當證據
        blk[~ok] = 0.0
        # 鼻點不可信 → 只清掉臉部相關量,手臂幾何仍然有效
        for c in (K_D_NOSE, K_PHI_X, K_PHI_Y, K_D_EAR, K_D_EYE,
                  K_H_OVER_EYE, K_V_DNOSE):
            blk[~face_ok, c] = 0.0
        blk[:, K_VALID] = ok.astype(np.float32)
        blk[:, K_FACE_OK] = face_ok.astype(np.float32)
        out[:, sl] = blk

    g = out[:, GLOBAL_SLICE]
    g[:, G_TILT_X] = frames["up"][:, 0]
    g[:, G_TILT_Y] = -frames["up"][:, 1]
    g[:, G_LOG_SCALE] = np.log(np.maximum(frames["scale"], 1e-3) / 100.0)
    for t in range(T):
        ori = (estimate_orientation(full[t], conf_thresh, nose_conf)
               if full.shape[1] >= 13 else "unknown")
        g[t, G_FRONT] = ori == "front"
        g[t, G_BACK] = ori == "back"
        g[t, G_UNKNOWN] = ori == "unknown"
    g[:, G_VALID] = frames["valid"].astype(np.float32)
    out[:, GLOBAL_SLICE] = g
    return out


SIDE_INPUT_DIM = SIDE_DIM + GLOBAL_DIM   # 25

# 鏡像正規化的符號向量:把左手的橫向分量取負,左手就長得跟右手一樣。
# 這樣 L1 的基元分類頭可以左右共用權重(等於訓練資料加倍),而不是
# 各學一套。縱向量(高度、距離、肘角)本來就左右同義,不動。
_x_components = [K_PHI_X, K_UPPER_X, K_FORE_X, SIDE_DIM + G_TILT_X]
SIDE_VIEW_MIRROR = np.ones(SIDE_INPUT_DIM, np.float32)
SIDE_VIEW_MIRROR[_x_components] = -1.0


def side_view(kin: np.ndarray, side: str,
              canonical: bool = True) -> np.ndarray:
    """取單側的運動學輸入 (T, 25) = 該側 18 維 + 全域 7 維。

    canonical=True 時對左手做鏡像正規化(見 SIDE_VIEW_MIRROR)。
    """
    v = np.concatenate([kin[..., SIDE_SLICE[side]],
                        kin[..., GLOBAL_SLICE]], axis=-1)
    if canonical and side == "L":
        v = v * SIDE_VIEW_MIRROR
    return v


def relative_angle_deg(kin: np.ndarray, side: str) -> np.ndarray:
    """腕相對鼻的方位角(度),0° = 正右方、90° = 正上方。給人看的。"""
    blk = kin[:, SIDE_SLICE[side]]
    return np.degrees(np.arctan2(blk[:, K_PHI_Y], blk[:, K_PHI_X]))
