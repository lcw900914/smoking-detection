# 抽菸行為偵測系統(Channel-as-Temporal-Buffer)

不依賴香菸物件、**純動作時序分析**的抽菸行為偵測系統。不偵測「菸」這個
在監控距離下只有幾個像素的小物件,而是偵測「抽菸這個動作序列」:
舉手靠近嘴 → 停留吸食 → 放下 → 週期性重複。

開發環境:單張 NVIDIA RTX 2060(6GB VRAM),Python + PyTorch 2.x。
架構全程只用 2D 卷積算子,保留向邊緣 NPU(Jetson Nano / STM32N6)移植的相容性。

---

## 系統功能

| 功能 | 說明 |
|------|------|
| 多人即時偵測 | YOLOv8-pose 偵測 + ByteTrack 追蹤,每人獨立維護狀態,實測 1080p 串流 8+ fps(RTX 2060) |
| RTSP 串流 | 支援 IP 攝影機(憑證特殊字元自動編碼、TCP 傳輸、只保留最新影格不累積延遲、斷線自動重連) |
| 次數警戒等級 | 以「手到嘴事件次數」逐級升高:1 次低(0.2)/ 2 次中(0.5)/ ≥3 次高(0.8),而非單一動作就判定 |
| 通報條件(可調) | 紅色警報 = 非背向 + 非移動中 + 事件次數 ≥ N(GUI 可輸入)+ 融合分數持續超過觸發線 |
| 骨架可視化 | 畫面即時繪製骨架、腕-鼻距離線(手在嘴部時紅色高亮)、動作階段(S1 舉手 / S2 嘴部停留 / S3 放下) |
| 背向處理 | 背對鏡頭時骨架分支「棄權」(2D 腕鼻距離不可信),不誤報;網路分數仍高者轉橘色「無法確認」警示並自動存 hard case |
| 移動排除(可開關) | 視窗內累積移動 ≥ 3 倍自身身高 → 走動中不視為抽菸(以人物自身尺寸為單位,與鏡頭距離無關) |
| 逗留警告 | 長時間在場 + 手部關鍵點不可見 + 位移小 → 橘色逗留提示(背向漏檢不靜默消失) |
| 距離自適應 | 腕-鼻距離以身體尺度正規化(先天距離不變);太遠(尺度 < 24px)棄權;門檻附帶像素誤差餘裕,越遠自動放寬 |
| Demo GUI | Tkinter 介面:影像隨視窗縮放、固定 8 格追蹤狀態面板、門檻/次數/移動排除即時調整、警報記錄與截圖 |

## 方法

### 1. 核心:Channel-as-Temporal-Buffer 時序建模

每幀影像經共享權重的 2D backbone(ResNet-18 改 stride-8)產生 `C×H×W = 128×28×28`
特徵圖,推入**環形緩衝區**;推理時將最近 T 幀沿通道軸疊成 `(T·C)×H×W` 厚特徵圖:

```
通道排列(interleave):[c0t0, c0t1, ..., c0t15, c1t0, ...]
→ Conv2d(T·C → T·C, kernel=1, groups=C)   每組恰為同一特徵通道的 16 個時刻
                                            = depthwise temporal conv(攤平為 2D)
→ Conv2d(T·C → 256, kernel=1)              跨特徵語意混合
→ 2 個 3×3 conv block → GAP → 分類頭
```

- **純 2D 算子**:無 Conv3D / LSTM / attention(單元測試強制白名單)
- **Continual inference**:新幀只跑一次 backbone,僅重算時序頭,算力為滑動視窗重算法的 1/T
- **雙時間尺度**:短尺度 T=16, stride=1(單次動作約 2 秒)+ 長尺度 T=16, stride=8
  (週期重複約 13 秒),兩頭 embedding 融合輸出 cycle score

### 2. 骨架輔助分支(late fusion)

YOLOv8s-pose 一次前向同時取得人物框與 17 個 COCO 關鍵點:

- **腕-鼻正規化距離** `d = dist(腕, 鼻) / max(肩寬, 0.55×軀幹高)`
  (軀幹高對水平旋轉不變,補側面視角肩寬被壓縮的問題)
- 規則推斷階段:d 快速縮小 → S1;d < 0.9+誤差餘裕 → S2 手在臉部;自嘴部快速拉大 → S3
- **朝向判斷**:COCO 關鍵點有左右語意,背對時左右肩 x 順序顛倒(+ 鼻點置信度)
  → 背向時棄權,不產生 S2 訊號
- S2 停留(≥0.5 秒,容忍 0.5 秒偵測中斷)結算為一次「手到嘴事件」進入次數計數器

### 3. 融合與警報

```
單週期分數 = 0.4 × 次數警戒分數(1次0.2 / 2次0.5 / ≥3次0.8, 90 秒視窗)
           + 0.6 × 網路 cycle score
P_t = 0.9·P_{t-1} + 0.1·單週期分數          (EMA,per-track)
紅色警報:非背向 ∧ 非移動中 ∧ 次數 ≥ N ∧ P_t > 觸發線持續 2 秒
解除:P_t < 解除線(雙門檻 hysteresis)
```

### 4. 訓練(6GB VRAM 配方)

1. **離線特徵**:凍結 backbone 抽全資料集特徵存 fp16 .npy
2. **階段一**:只訓練時序頭(batch 128-256,不碰影像)
3. **階段二**:端到端微調(AMP + batch 8 × grad accum 4 + backbone lr ×0.1)
4. **離線蒸餾**(已備妥):X3D-M teacher 預先推理存 soft labels,師生絕不同時佔 VRAM

## 目前成果與已知問題

**資料**:HMDB51 子集(smoke/drink/eat/chew/talk 共 610 段),YOLO+ByteTrack
自動標註(僅 clip 級標籤,無幀級階段標籤)。

**Clip 級(驗證集 105 段)**:端到端微調後 accuracy 93.3%、macro-F1 0.86、
AUC 0.85;smoking recall 11/18;喝水/吃東西/講話 hard negative 零誤判。

**已知問題:實地(監控視角)不漏測但誤測偏多。** 根因分析:

1. **Domain gap**:網路以 HMDB51(電影片段、多為正面近景)訓練,對監控俯視角
   的一般動作(摸臉、喝水、講電話、托腮)常給出高分;實測背向辦公人員的網路
   分數常達 0.7–0.95(訓練集中沒有這類負樣本)
2. **骨架幾何的先天歧義**:「手在臉部區域」是抽菸的必要非充分條件,
   規則分支無法單獨區分摸臉/托腮
3. 已有的緩解(背向棄權、次數門檻、移動排除)壓低了誤報頻率,
   但**治本需要目標場景的自錄資料重訓**——系統已自動把「背向高分」案例
   存進 `hard_cases/`,即為重訓資料的來源

**Roadmap**:自錄資料集(幀級 S1–S3 標籤 + 監控視角 hard negatives)→ 重訓
外觀網路與階段頭 → 啟用狀態機順序驗證 → KD 蒸餾(M5)→ 事件級評估 FP/h(M6)。

## 安裝與使用

```bash
pip install -r requirements.txt        # GPU 請裝對應 CUDA 的 torch/torchvision
# Windows 若遇 OMP Error #15:set KMP_DUPLICATE_LIB_OK=TRUE

pytest tests/ -v                       # 101 個單元測試
python scripts/smoke_test.py           # 合成資料端到端煙霧測試(CPU 可跑)

# Demo GUI(來源可為攝影機編號、影片檔或 rtsp:// URL)
python scripts/gui.py

# 命令列管線
python -m inference.pipeline --source "rtsp://user:pass@ip:554" \
    --model-ckpt checkpoints/hmdb_e2e_best.pt
```

訓練流程(前處理 → 特徵抽取 → 兩階段訓練 → 評估)詳見各腳本 docstring:
`data/preprocess.py` → `training/extract_features.py` → `training/train_head.py`
→ `training/train_e2e.py` → `eval/clip_eval.py` / `eval/event_eval.py`。

模型權重與資料集不進版控(見 `.gitignore`);HMDB51 取得方式與自動標註:
`scripts/auto_annotate.py`。

## 專案結構

```
configs/        yaml 設定(model / train / inference,超參數零寫死)
models/         backbone、ring_buffer(通道排列唯一定義)、temporal_head、full_model
tracking/       YOLOv8(-pose)偵測、ByteTrack、ROI 平滑裁切
inference/      即時管線、狀態機/計數器/移動閘門、骨架分支、警報、RTSP 串流
data/           影片離線前處理、clip dataset、離線特徵 dataset
training/       兩階段訓練、特徵抽取、離線蒸餾、losses
eval/           clip 級與事件級評估
scripts/        GUI、demo、自動標註、煙霧測試
tests/          101 個單元測試(通道排列與 continual/offline 一致性為核心)
```
