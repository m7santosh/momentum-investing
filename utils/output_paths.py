"""Shared locations for script outputs."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINAL_RESULT_DIR = REPO_ROOT / "final_result"
FINAL_RESULT_STOCK_DIR = FINAL_RESULT_DIR / "stock"
FINAL_RESULT_ETF_DIR = FINAL_RESULT_DIR / "etf"
