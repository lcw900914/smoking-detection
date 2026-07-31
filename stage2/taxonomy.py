"""標記類別表與訓練用的合併對應。

**類別粒度的取捨(重要)**

手臂在 COCO 17 點裡只有肩、肘、腕三個節點,頭部只有鼻、眼、耳。
沒有手指、沒有嘴巴。所以「推眼鏡 / 摸鼻子 / 擦臉 / 抓頭」在特徵空間
裡幾乎是同一個東西 —— 分開標只是製造雜訊,不會讓模型學到更多。

合併的判準是兩件事的聯集:
  1. 骨架分不出來的 -> 合併(推眼鏡 vs 摸鼻子)
  2. 人標的時候會猶豫的 -> 合併(擦臉 vs 摸鼻子,標記者自己就不一致)

反過來,人一眼就能判、骨架也有機會分的(抽菸 / 喝水 / 講電話),
就留著 —— 這些是誤報分析時最想單獨看的幾類。

**移動型態不在這裡。** 經過 / 徘徊 / 等待是從框的軌跡算出來的
(見 inference/state_machine.py 的 MovementGate、LoiterDetector),
不需要人標,也不該讓模型學 —— 門檻要調時改一個數字就好,模型得重訓。

**室內/室外也不在這裡。** 那是整批影片的屬性,不是單段的動作類別。
"""

# 按鈕分組(決定標記工具的版面順序)
GROUPS = ["動作", "排除"]

# (快捷鍵, 標籤代碼, 顯示名, 分組) —— 全部用數字鍵,單手就能標完
CATEGORIES = [
    ("1", "smoking",     "抽菸",      "動作"),
    ("2", "drinking",    "喝水",      "動作"),
    ("3", "eating",      "吃東西",    "動作"),
    ("4", "phone_call",  "講電話",    "動作"),
    ("5", "face_touch",  "手碰臉",    "動作"),
    ("6", "no_contact",  "手沒碰臉",  "動作"),

    ("8", "bad_pose",    "骨架錯誤",   "排除"),
    ("9", "back_view",   "背對看不清", "排除"),
    ("0", "other",       "其他/不確定", "排除"),
]

# 概括型類別涵蓋什麼(標記工具顯示成提示,避免標記者自己解讀)
EXAMPLES = {
    "smoking": "含點菸、電子菸",
    "phone_call": "手機貼耳;滑手機請按 6",
    "face_touch": "推眼鏡、口罩、抓頭、擦臉、托腮、耳機、摸鼻子、帽子",
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


def train_index(code: str):
    """標籤代碼 -> 訓練類別索引;不進訓練者回傳 None。"""
    group = MERGE.get(code)
    return TRAIN_CLASSES.index(group) if group is not None else None


def display_name(code: str) -> str:
    """標籤代碼 -> 顯示名(含舊代碼)。"""
    return NAME_OF.get(code) or LEGACY_NAME.get(code) or code
