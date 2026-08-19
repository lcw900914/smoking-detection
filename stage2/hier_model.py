"""兩層骨架時序模型。

    L1 PrimitiveNet   逐幀 → 淺層基元(靜置/舉手/停留/放下/其他活動)
        輸入:骨架拓樸圖序列 (B,5,T,13) + 可解釋運動學 (B,T,45)
        輸出:每側每幀的基元機率 + 每側每幀的嵌入
        監督:規則偽標籤(見 primitives.py),模糊幀 ignore

    L2 CompositionNet 片段序列 → 具體動作(抽菸/喝水/講電話/扶眼鏡/抓頭髮)
        輸入:L1 切出來的片段 token 序列(基元 + 屬性 + 嵌入)+ 節律統計
        輸出:六類機率
        監督:片段級人工標籤(annotations/clip_labels.json)

**為什麼分兩層,而不是一個網路直接吃骨架吐「抽菸」?**
1. 資料量。直接端到端要學「像素/座標 → 抽菸」,幾十段片段遠遠不夠。
   拆開之後,L1 的監督是逐幀的(64 段 × 約 134 幀 × 2 手 = 17,338 筆)
   而且標籤免費;L2 的輸入已經被壓成十幾個 token,參數量壓到 3 萬以下。
2. 可解釋。誤判時看得到是哪一層錯:基元時間軸畫出來,是「停留」判錯,
   還是「停留 1.2 秒 × 3 次」被組成了抽菸,一眼分得出來。
3. 可換。抽菸/喝水的判準要調,改 L2;姿態模型換掉,只需重訓 L1。

參考文獻
────────
本檔的兩個網路都由標準積木組成,論文裡要照實引用:

L1 PrimitiveNet 的主幹
  Yan, S., Xiong, Y., Lin, D. (2018). Spatial Temporal Graph
  Convolutional Networks for Skeleton-Based Action Recognition.
  AAAI 2018. https://doi.org/10.1609/aaai.v32i1.12328
  (實作見 stage2/stgcn.py;本專案加的功能邊見 stage2/graph.py)

L2 CompositionNet 的編碼器與位置編碼
  Vaswani, A. et al. (2017). Attention Is All You Need.
  NeurIPS 2017. arXiv:1706.03762

位置編碼吃**真實秒數**而非 token 序號,這個想法也不是本專案首創:
  Kazemi, S.M. et al. (2019). Time2Vec: Learning a Vector
  Representation of Time. arXiv:1907.05321
  Shukla, S.N., Marlin, B.M. (2021). Multi-Time Attention Networks
  for Irregularly Sampled Time Series. ICLR 2021. arXiv:2101.10318
本專案的作法是把 Vaswani 的正弦編碼直接餵秒數(週期 0.5–60 秒),
比上面兩篇單純:沒有可學的時間嵌入,也沒有時間注意力。

「原子動作 → 複合行為」這種兩層切法在時序動作分割裡有大量前作,
主張新穎前請先查文獻,例如:
  Abu Farha, Y., Gall, J. (2019). MS-TCN: Multi-Stage Temporal
  Convolutional Network for Action Segmentation. CVPR 2019.

**本專案自己的部分**(這些才是可以主張的):五個基元的詞彙表與
「標籤全由幾何規則自動產生、L1 不需人標」的作法(stage2/primitives.py)、
21 維片段屬性與 16 維節律統計(stage2/composition.py)、
以及零參數的片段文法 grammar_scores。
"""
import numpy as np
import torch
import torch.nn as nn

from stage2.graph import ARM_CHAIN, SIDES, build_adjacency
from stage2.kinematics import (GLOBAL_SLICE, GRAPH_CHANNELS,
                               SIDE_INPUT_DIM, SIDE_SLICE,
                               SIDE_VIEW_MIRROR)
from stage2.primitives import NUM_PRIMITIVES, SEG_ATTR_DIM
from stage2.stgcn import STGCNTrunk
from stage2.taxonomy import DEEP_CLASSES

PRIM_EMBED_DIM = 16          # L1 送給 L2 的每幀嵌入寬度


class PrimitiveNet(nn.Module):
    """L1:骨架拓樸圖 + 運動學 → 逐幀基元。

    兩側共用同一顆分類頭:左手的運動學先做鏡像正規化
    (SIDE_VIEW_MIRROR),看起來就跟右手一樣,等於訓練樣本加倍;
    另外餵一個側別旗標,讓模型仍可表達左右差異(慣用手)。
    """

    def __init__(self, num_primitives: int = NUM_PRIMITIVES,
                 channels=(32, 48, 64), kt: int = 5,
                 embed_dim: int = PRIM_EMBED_DIM, dropout: float = 0.2,
                 include_functional_edges: bool = True):
        super().__init__()
        A = torch.from_numpy(build_adjacency(include_functional_edges))
        self.trunk = STGCNTrunk(GRAPH_CHANNELS, A, channels=channels,
                                kt=kt, dropout=dropout)
        c = self.trunk.out_channels

        self.node_proj = nn.Sequential(          # 該側 肩/肘/腕 三個節點
            nn.Conv1d(c * 3, 48, 1), nn.BatchNorm1d(48), nn.ReLU(True))
        self.kin_enc = nn.Sequential(            # 可解釋運動學支流
            nn.Conv1d(SIDE_INPUT_DIM, 32, 5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(True))
        self.fuse = nn.Sequential(
            nn.Conv1d(48 + 32 + 1, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(True), nn.Dropout(dropout))
        self.head = nn.Conv1d(64, num_primitives, 1)
        self.embed = nn.Conv1d(64, embed_dim, 1)
        self.embed_dim = embed_dim

        self.register_buffer("mirror",
                             torch.from_numpy(SIDE_VIEW_MIRROR.copy()))
        idx = {s: torch.tensor(
            list(range(SIDE_SLICE[s].start, SIDE_SLICE[s].stop)) +
            list(range(GLOBAL_SLICE.start, GLOBAL_SLICE.stop)))
            for s in SIDES}
        self.register_buffer("idx_L", idx["L"])
        self.register_buffer("idx_R", idx["R"])

    def _side_kin(self, kin: torch.Tensor, side: str) -> torch.Tensor:
        """(B,T,45) → (B,25,T),左手做鏡像正規化。"""
        idx = self.idx_L if side == "L" else self.idx_R
        v = kin.index_select(-1, idx)
        if side == "L":
            v = v * self.mirror
        return v.transpose(1, 2)

    def forward(self, graph: torch.Tensor, kin: torch.Tensor):
        """graph (B,5,T,13)、kin (B,T,45)
        → logits (B,T,2,P)、embed (B,T,2,E)。第 2 維順序為 (左, 右)。
        """
        B, _, T, _ = graph.shape
        h = self.trunk(graph)                              # (B,c,T,V)
        logits, embeds = [], []
        for si, side in enumerate(SIDES):
            nodes = torch.tensor(ARM_CHAIN[side], device=graph.device)
            g = h.index_select(3, nodes)                   # (B,c,T,3)
            g = g.permute(0, 1, 3, 2).reshape(B, -1, T)    # (B,3c,T)
            k = self._side_kin(kin, side)                  # (B,25,T)
            flag = torch.full((B, 1, T), float(si), device=graph.device)
            z = self.fuse(torch.cat(
                [self.node_proj(g), self.kin_enc(k), flag], dim=1))
            logits.append(self.head(z).transpose(1, 2))    # (B,T,P)
            embeds.append(self.embed(z).transpose(1, 2))   # (B,T,E)
        return torch.stack(logits, dim=2), torch.stack(embeds, dim=2)


# ---- L2 -------------------------------------------------------------

TOKEN_DIM = NUM_PRIMITIVES + 2 + SEG_ATTR_DIM + PRIM_EMBED_DIM   # 43


def _time_sinusoid(times: torch.Tensor, dim: int) -> torch.Tensor:
    """以「秒」為單位的正弦位置編碼 (B,N) → (B,N,dim)。

    片段不是等距的:兩段 hold 差 2 秒還是 40 秒,是抽菸與否的關鍵。
    用序號當位置會把這個資訊丟掉,所以直接編碼真實時間。
    週期涵蓋 0.5–60 秒,對應「一口之內」到「一根菸之間」的尺度。
    """
    half = dim // 2
    freqs = torch.exp(torch.linspace(np.log(2 * np.pi / 0.5),
                                     np.log(2 * np.pi / 60.0), half,
                                     device=times.device))
    ang = times.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


ENCODERS = ("transformer", "gru", "bag")


class CompositionNet(nn.Module):
    """L2:淺層片段序列 → 具體動作。

    序列編碼器可換(`encoder=`),**其餘全部固定** —— token 怎麼組、
    時間怎麼編碼、怎麼池化、怎麼接統計、分類頭長什麼樣,三種選項
    一模一樣。這樣「換編碼器」就是單一變因,比較才有意義:

        transformer  自注意力 ×2 層(預設)
        gru          雙向 GRU ×2 層,hidden = d_model/2 使輸出維度不變
        bag          不做任何 token 之間的互動,直接送去池化

    `bag` 不是湊數的:它回答「片段之間到底需不需要互動」。如果 bag 就
    打平另外兩個,代表判別訊息全在單一片段的屬性與整段的節律統計裡,
    那整個序列編碼器可以拆掉。這種事在資料只有幾十段時很常發生。

    原本選 Transformer 的理由(現在應該用實測檢驗,不是相信它):
    片段序列很短(通常 5–30 個 token),但判準是「兩兩之間像不像」
    (抽菸的每一口長得幾乎一樣)——自注意力天生在算兩兩關係,
    GRU 得靠隱狀態繞。

    注意:**三種編碼器的參數量不同**(Transformer 約 17k、GRU 約 9.6k、
    bag 為 0),比較時要一起報,不然分不出是「架構比較好」還是
    「容量比較合適」。
    """

    def __init__(self, num_classes: int = len(DEEP_CLASSES),
                 token_dim: int = TOKEN_DIM, stat_dim: int = 16,
                 d_model: int = 32, nhead: int = 4, layers: int = 2,
                 dropout: float = 0.3, encoder: str = "transformer"):
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(
                f"未知的編碼器 {encoder!r};可用:{', '.join(ENCODERS)}")
        self.proj = nn.Linear(token_dim, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.encoder_kind = encoder

        if encoder == "transformer":
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, norm_first=True)
            # norm_first 與 nested tensor 併用時 PyTorch 會自行停用後者並
            # 發警告;序列本來就只有十幾個 token,直接關掉省事
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers,
                                                 enable_nested_tensor=False)
        elif encoder == "gru":
            # hidden = d_model/2 且雙向 → 輸出仍是 d_model,
            # 下游的池化與分類頭一行都不用改
            if d_model % 2:
                raise ValueError("gru 編碼器需要偶數 d_model")
            self.encoder = nn.GRU(
                d_model, d_model // 2, num_layers=layers, batch_first=True,
                bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        else:
            self.encoder = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(d_model + stat_dim, 48), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(48, num_classes))
        self.d_model = d_model

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B,N,d) + 有效遮罩 → (B,N,d)。三種編碼器的唯一分歧點。"""
        if self.encoder_kind == "transformer":
            return self.encoder(x, src_key_padding_mask=~mask)
        if self.encoder_kind == "gru":
            # 補齊用的空位不能餵進 GRU,否則末端隱狀態被零向量洗掉;
            # pack 的長度為 0 會直接報錯,所以下限夾 1
            lens = mask.sum(1).clamp(min=1).to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens, batch_first=True, enforce_sorted=False)
            out, _ = self.encoder(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                out, batch_first=True, total_length=x.shape[1])
            return out
        return x

    def forward(self, tokens: torch.Tensor, times: torch.Tensor,
                mask: torch.Tensor, stats: torch.Tensor,
                return_attn: bool = False):
        """tokens (B,N,D)、times (B,N) 秒、mask (B,N) True=有效、
        stats (B,S) 已由 composition.normalize_stats 標準化
        → logits (B,C)。"""
        x = self.token_norm(self.proj(tokens)) + \
            _time_sinusoid(times, self.d_model)
        h = self.encode(x, mask)
        w = mask.float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        z = torch.cat([pooled, stats], dim=1)
        logits = self.head(z)
        return (logits, h) if return_attn else logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
