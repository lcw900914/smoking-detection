"""骨架拓樸圖:節點子集、邊集合與 ST-GCN 用的分區鄰接矩陣。

為什麼要用圖而不是把 17×3 攤平成向量(舊 stage2/normalize.py 的做法):
攤平之後「腕」跟「膝」在特徵向量裡只是相鄰的兩個數字,模型得自己從
資料學出「腕接肘、肘接肩」這件事——資料只有幾十段時學不出來。
把拓樸寫進鄰接矩陣,等於免費送給模型一個正確的歸納偏置。

節點子集(13 點):COCO 17 點的前 13 個,砍掉膝與踝。
理由:本專案的判別動作全部發生在手-臉之間,腿部節點在辦公室場景常被
桌子遮擋,留著只會帶進噪聲。前 13 個索引與 COCO 原編號完全相同,
不需要重新編號對照表。

邊分兩種:
  ANATOMICAL 解剖邊——真實骨頭連接,決定「向心/離心」分區。
  FUNCTIONAL 功能邊——腕↔鼻、腕↔耳、腕↔腕。這幾條不是骨頭,但正是
    本任務的判別關係(手到嘴 / 手到耳)。加進圖裡,訊息一步就能傳到,
    不必靠腕→肘→肩→頸→鼻繞四層。
"""
from typing import List, Tuple

import numpy as np

# ---- 節點(索引與 COCO 相同)-------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP = 11, 12

NUM_NODES = 13
COCO_SUBSET = list(range(NUM_NODES))   # 由 (T,17,3) 取前 13 點

NODE_NAMES = [
    "鼻", "左眼", "右眼", "左耳", "右耳",
    "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
    "左髖", "右髖",
]

# 左右對調索引(水平翻轉增強用)
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

# 每一側的手臂鏈(L2 取片段特徵、L1 取側別節點時用)
ARM_CHAIN = {
    "L": (L_SHO, L_ELB, L_WRI),
    "R": (R_SHO, R_ELB, R_WRI),
}
SIDE_EAR = {"L": L_EAR, "R": R_EAR}
SIDE_EYE = {"L": L_EYE, "R": R_EYE}
SIDES = ("L", "R")

# ---- 邊 ---------------------------------------------------------------
ANATOMICAL_EDGES: List[Tuple[int, int]] = [
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    (NOSE, L_SHO), (NOSE, R_SHO),          # 頸部(COCO 無頸點,以鼻代)
    (L_SHO, R_SHO),
    (L_SHO, L_ELB), (L_ELB, L_WRI),
    (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
]

FUNCTIONAL_EDGES: List[Tuple[int, int]] = [
    (L_WRI, NOSE), (R_WRI, NOSE),          # 手到嘴(抽菸/喝水/吃)
    (L_WRI, L_EAR), (R_WRI, R_EAR),        # 手到同側耳(講電話)
    (L_WRI, R_EAR), (R_WRI, L_EAR),        # 跨側(左手接右耳也常見)
    (L_WRI, L_EYE), (R_WRI, R_EYE),        # 手到眼(扶眼鏡)
    (L_WRI, R_WRI),                        # 雙手互動(點菸、捧杯)
]

# 中心節點集合:向心/離心分區以「到雙肩的最短跳數」定義。
# 用集合而非單一節點,避免左右不對稱。
CENTER_NODES = (L_SHO, R_SHO)


def hop_distance(edges=None, num_nodes: int = NUM_NODES) -> np.ndarray:
    """各節點到中心(雙肩)的最短跳數;不連通為 inf。"""
    edges = ANATOMICAL_EDGES if edges is None else edges
    adj = [[] for _ in range(num_nodes)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    dist = np.full(num_nodes, np.inf)
    frontier = list(CENTER_NODES)
    for n in frontier:
        dist[n] = 0
    d = 0
    while frontier:
        d += 1
        nxt = []
        for n in frontier:
            for m in adj[n]:
                if dist[m] == np.inf:
                    dist[m] = d
                    nxt.append(m)
        frontier = nxt
    return dist


def build_adjacency(include_functional: bool = True,
                    num_nodes: int = NUM_NODES) -> np.ndarray:
    """ST-GCN 空間分區鄰接矩陣 (3, V, V),已做度正規化。

    三個分區(ST-GCN 的 spatial configuration partitioning):
      A[0] 自身      —— 節點本身(含自環)
      A[1] 向心      —— 鄰居比自己更靠近軀幹中心(手臂收回的方向)
      A[2] 離心      —— 鄰居比自己更遠離中心(手臂伸出的方向)

    分區用「解剖跳數」決定,功能邊也依同一套跳數歸位——腕(跳數 3)
    連到鼻(跳數 1)自然落進向心分區,語意正確。

    每個分區各自以 D^-1 A 正規化(行和為 1),避免度大的節點(如肩)
    在聚合時壓過其他節點。
    """
    edges = list(ANATOMICAL_EDGES)
    if include_functional:
        edges += FUNCTIONAL_EDGES
    hop = hop_distance(num_nodes=num_nodes)

    A = np.zeros((3, num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        A[0, i, i] = 1.0                      # 自環
    for i, j in edges:
        for a, b in ((i, j), (j, i)):         # 無向:兩個方向都建
            # A[k, a, b] = b 對 a 的貢獻,分區看的是「來源 b 相對 a」
            if hop[b] == hop[a]:
                A[0, a, b] = 1.0
            elif hop[b] < hop[a]:
                A[1, a, b] = 1.0
            else:
                A[2, a, b] = 1.0

    for k in range(A.shape[0]):               # 度正規化
        deg = A[k].sum(axis=1, keepdims=True)
        A[k] = A[k] / np.maximum(deg, 1e-6)
    return A


def edge_list(include_functional: bool = True) -> List[Tuple[int, int]]:
    """繪圖/除錯用的邊清單。"""
    return (list(ANATOMICAL_EDGES) +
            (list(FUNCTIONAL_EDGES) if include_functional else []))
