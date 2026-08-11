"""標記類別表與訓練用的合併對應。

**類別粒度的取捨(重要,2026-08 修訂)**

手臂在 COCO 17 點裡只有肩、肘、腕三個節點,頭部只有鼻、眼、耳。
沒有手指、沒有嘴巴。舊版據此把「推眼鏡 / 摸鼻子 / 擦臉 / 抓頭」全部
併成一類——在**逐幀攤平特徵**上這是對的,那時候它們確實是同一團點。

兩層模型改變了這個結論。深層看的不是幀,是**片段**:
    扶眼鏡 = 手停在眼睛高度、0.5–1 秒、幾乎不動
    抓頭髮 = 手停在**眼睛以上**、1–3 秒、來回摩擦
    托腮   = 手停在下巴、5 秒以上、完全不動
這三件事在「停留位置 × 停留時長 × 停留期間的速度」這個空間裡是分開的,
所以現在拆成三顆按鈕。合併與否的判準沒有變,變的是特徵的解析度:

  1. 骨架分不出來的 -> 合併
  2. 人標的時候會猶豫的 -> 合併(擦臉 vs 摸鼻子,標記者自己就不一致)

「擦臉 vs 摸鼻子」仍然併在 face_touch,因為第 2 條還是成立。

**移動型態不在這裡。** 經過 / 徘徊 / 等待是從框的軌跡算出來的
(見 inference/state_machine.py 的 MovementGate、LoiterDetector),
不需要人標,也不該讓模型學 —— 門檻要調時改一個數字就好,模型得重訓。

**室內/室外也不在這裡。** 那是整批影片的屬性,不是單段的動作類別。
"""

# 按鈕分組(決定標記工具的版面順序)
GROUPS = ["動作", "排除"]

# (快捷鍵, 標籤代碼, 顯示名, 分組) —— 全部用數字鍵,單手就能標完
#
# 2026-08 改版:原本「手碰臉」一個按鈕涵蓋推眼鏡/抓頭/托腮,是因為舊的
# 攤平 TCN 分不出來,分開標只會製造雜訊。兩層模型改吃片段序列之後,
# 「手停在眼睛高度 0.8 秒」與「手停在頭頂上方 2 秒」在片段層次是分得開的,
# 所以把它拆成扶眼鏡 / 抓頭髮 / 其他碰臉三顆。深層詞彙見 DEEP_CLASSES。
CATEGORIES = [
    ("1", "smoking",      "抽菸",       "動作"),
    ("2", "drinking",     "喝水",       "動作"),
    ("3", "eating",       "吃東西",     "動作"),
    ("4", "phone_call",   "講電話",     "動作"),
    ("5", "glasses",      "扶眼鏡",     "動作"),
    ("6", "scratch_head", "抓頭髮",     "動作"),
    ("7", "face_touch",   "其他碰臉",   "動作"),
    ("8", "no_contact",   "手沒碰臉",   "動作"),

    ("9", "bad_pose",     "骨架錯誤",   "排除"),
    ("0", "back_view",    "背對看不清", "排除"),
    ("-", "other",        "其他/不確定", "排除"),
]

# 概括型類別涵蓋什麼(標記工具顯示成提示,避免標記者自己解讀)
EXAMPLES = {
    "smoking": "含點菸、電子菸",
    "phone_call": "手機貼耳;滑手機請按 8",
    "glasses": "推眼鏡、口罩、耳機 —— 手停在眼睛高度、時間很短",
    "scratch_head": "抓頭、撥瀏海、戴脫帽 —— 手高過眼睛",
    "face_touch": "托腮、摸鼻子、擦臉、撐下巴",
    "no_contact": "滑手機、比手勢、拿東西、站著沒事、撐傘",
}

NAME_OF = {code: name for _, code, name, _ in CATEGORIES}
CODES = [code for _, code, _, _ in CATEGORIES]

# ---------------------------------------------------------------------
# 訓練用的合併類別(smoking 固定為 index 0,評估時直接取這一維)
#
# 目前資料極少,先併成 4 類。等喝水/吃東西/講電話各自累積夠了,
# 把下面 MERGE 裡那三行改成各自獨立即可 —— 不必重標。
# ---------------------------------------------------------------------
TRAIN_CLASSES = ["smoking", "hand_to_mouth", "face_touch", "no_contact"]

MERGE = {
    "smoking": "smoking",

    # 這三類手都真的到嘴,骨架難分;先合併,資料夠了再拆
    "drinking": "hand_to_mouth",
    "eating": "hand_to_mouth",
    "phone_call": "hand_to_mouth",

    "face_touch": "face_touch",
    "no_contact": "no_contact",

    # --- 舊標籤:既有標記不作廢,但標記工具會標示「建議複標」-------
    "phone": "hand_to_mouth",       # 2026-07 版「講電話」
    "desk_work": "face_touch",      # 2026-07 版「桌面工作/摸臉」
    # 2026-07 版「other_neg(其他非抽菸)」是雜項,無法誠實歸類 -> 不對應
    # 過渡期的細類表(已收斂成上面 6 類)
    "lighting": "smoking", "ecig": "smoking",
    "cough": "hand_to_mouth", "yawn": "hand_to_mouth",
    "radio": "hand_to_mouth",
    "glasses": "face_touch", "mask": "face_touch",
    "scratch_head": "face_touch", "wipe_face": "face_touch",
    "chin_rest": "face_touch", "earphone": "face_touch",
    "touch_nose": "face_touch", "hat": "face_touch",
    "phone_scroll": "no_contact", "gesture": "no_contact",
    "carrying": "no_contact", "idle": "no_contact",
    "umbrella": "no_contact",
}

# 明確排除(標了但不進訓練)
EXCLUDED = {"bad_pose", "back_view", "other", "unsure", "other_neg"}

# 已不再提供按鈕的代碼:標記工具會標示「建議複標」
LEGACY_CODES = ({"phone", "desk_work", "other_neg", "unsure"}
                | (set(MERGE) - set(CODES)))
LEGACY_NAME = {
    "phone": "講電話(舊)", "desk_work": "桌面工作/摸臉(舊)",
    "other_neg": "其他非抽菸(舊)", "unsure": "不確定(舊)",
}


# ---------------------------------------------------------------------
# 深層(L2)類別 —— 兩層骨架時序模型的輸出詞彙
#
# 與上面的 TRAIN_CLASSES 差別:TRAIN_CLASSES 是舊 stage2 攤平 TCN 用的
# 保守合併(把喝水/吃/講電話併成 hand_to_mouth,因為攤平特徵分不開)。
# 新的兩層模型吃的是「片段序列」——停留在耳邊 8 秒 vs 停留在嘴邊 1 秒
# 兩次,在片段層次是完全不同的東西,所以可以分開。
#
# 要加類別(例如把「吃東西」獨立出來)只需要動這兩個常數 + CATEGORIES。
# ---------------------------------------------------------------------
DEEP_CLASSES = ["smoking", "drinking", "phone_call", "glasses",
                "hair", "other"]
DEEP_NAMES = {"smoking": "抽菸", "drinking": "喝水", "phone_call": "講電話",
              "glasses": "扶眼鏡", "hair": "抓頭髮", "other": "其他"}

DEEP_MERGE = {
    "smoking": "smoking", "lighting": "smoking", "ecig": "smoking",
    "drinking": "drinking",
    "phone_call": "phone_call", "phone": "phone_call",
    "glasses": "glasses", "mask": "glasses",     # 都是手停在眼鼻區、極短
    "scratch_head": "hair", "hat": "hair",       # 都是手高過眼睛
    "no_contact": "other", "idle": "other", "gesture": "other",
    "carrying": "other", "umbrella": "other", "phone_scroll": "other",
    # 托腮/摸鼻/擦臉:不屬於五個具體動作,歸「其他」。
    # 注意這個代碼的語意在 2026-08 變窄了——舊版的「手碰臉」還涵蓋
    # 推眼鏡與抓頭,那時候它橫跨兩個深層類別,不能這樣對應。
    # 目前標籤檔裡沒有任何 face_touch,所以沒有舊資料受影響。
    "face_touch": "other",

    # 粗標負樣本:2026-07 標的時候只分「其他非抽菸 / 桌面工作」,
    # 裡面混著扶眼鏡、抓頭髮、托腮。當「other」用不會教壞二分類,
    # 但要訓練六分類前應該用新按鈕複標(見 COARSE_NEGATIVE_CODES)。
    "other_neg": "other", "desk_work": "other",
}

# 這些代碼進得了訓練,但語意太粗,六分類前建議複標
COARSE_NEGATIVE_CODES = {"other_neg", "desk_work"}

# face_touch 概括了扶眼鏡/抓頭髮/托腮/摸鼻——正好橫跨 glasses 與 hair
# 兩個深層類別,無法誠實對應,一律不進 L2 訓練(標記工具會提示複標)。
# eating 同理:使用者指定的深層詞彙沒有「吃東西」,要加的話在
# DEEP_CLASSES 補一格、這裡補一行 "eating": "eating" 即可。
#
# 注意:這裡**不能**直接沿用上面的 EXCLUDED。EXCLUDED 把 other_neg
# 排掉,是因為舊的四類 TRAIN_CLASSES 沒有一格放得下「其他非抽菸」;
# 深層有 other 這一類,other_neg 正好對得上,而且那是 48 段免費的
# 負樣本——照抄 EXCLUDED 會把資料集從 64 段縮成 16 段。
DEEP_EXCLUDED = {"eating", "chew", "cough", "yawn", "radio",
                 "touch_nose", "wipe_face", "chin_rest", "earphone",
                 "bad_pose", "back_view", "unsure", "other"}


def deep_index(code: str):
    """標籤代碼 → 深層類別索引;不進 L2 訓練者回傳 None。"""
    if code in DEEP_EXCLUDED:
        return None
    group = DEEP_MERGE.get(code)
    return DEEP_CLASSES.index(group) if group is not None else None


def deep_display(code_or_class: str) -> str:
    return DEEP_NAMES.get(code_or_class, display_name(code_or_class))


def train_index(code: str):
    """標籤代碼 -> 訓練類別索引;不進訓練者回傳 None。"""
    group = MERGE.get(code)
    return TRAIN_CLASSES.index(group) if group is not None else None


def display_name(code: str) -> str:
    """標籤代碼 -> 顯示名(含舊代碼)。"""
    return NAME_OF.get(code) or LEGACY_NAME.get(code) or code
