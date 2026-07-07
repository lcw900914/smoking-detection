"""pytest 共用設定:路徑與環境變數。"""
import os
import sys
from pathlib import Path

# Windows 上 torch 與其他套件的 OpenMP runtime 衝突之標準 workaround
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 專案根目錄加入 path,使 `models.` 等套件可直接 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
