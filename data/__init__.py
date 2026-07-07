"""資料套件:離線前處理、clip dataset、離線特徵 dataset。"""

# 階段標籤映射(全專案唯一定義;S4 吐煙僅為輔助訊號,併入 background)
STAGE_MAP = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "none": 3}
STAGE_NAMES = ["S1", "S2", "S3", "background"]

# clip 行為標籤(smoking 為正類,其餘皆為 hard negative / background)
LABEL_SET = ["smoking", "drinking", "eating", "phone", "wiping", "background"]
