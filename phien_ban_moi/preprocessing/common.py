"""Đọc/ghi CSV dùng chung; không phụ thuộc code crawl hoặc tự gán nhãn."""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(sorted(missing))}")


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Giữ nhãn đã duyệt và kết quả cũ: chọn --output mới khi cần chạy lại.
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        frame.to_csv(handle, index=False)


def run_cli(action):
    try:
        action()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Lỗi: {error}") from None
