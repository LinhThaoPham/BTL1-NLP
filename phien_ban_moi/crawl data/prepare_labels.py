"""Bước 3: python prepare_labels.py

Xuất Comment, label, post_url, comment_id. Nhãn để trống cho người gán:
0 = Non Toxic; 1 = Toxic. Giữ dấu tiếng Việt, emoji và các câu ngắn.
Không ghi đè file đã có để tránh mất nhãn đang duyệt; dùng --output cho đợt mới.
"""
import argparse
from pathlib import Path
import unicodedata

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def clean_text(value):
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def main():
    parser = argparse.ArgumentParser(description="Làm sạch cơ bản và tạo CSV chờ gán nhãn.")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "facebook_comments_all.csv")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "comments_with_label.csv")
    args = parser.parse_args()
    if not args.input.exists():
        parser.error("Chưa có CSV bình luận. Chạy crawl_comments.py trước.")
    if args.output.exists():
        parser.error("File đầu ra đã tồn tại; dùng --output tên_mới.csv để giữ nhãn cũ.")
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    required = {"comment", "post_url", "comment_id"}
    if not required.issubset(frame.columns):
        parser.error(f"CSV thiếu cột: {', '.join(sorted(required - set(frame.columns)))}")
    original_count = len(frame)
    frame["Comment"] = frame["comment"].map(clean_text)
    frame = frame.loc[frame["Comment"].ne("")].copy()
    frame = frame.drop_duplicates(subset=["post_url", "comment_id"])
    frame["label"] = ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Chế độ x bảo đảm không ghi đè nếu file xuất hiện trong lúc xử lý.
    with args.output.open("x", encoding="utf-8-sig", newline="") as handle:
        frame[["Comment", "label", "post_url", "comment_id"]].to_csv(handle, index=False)
    print(f"Ban đầu: {original_count}; còn lại: {len(frame)}; bỏ rỗng/trùng ID: {original_count - len(frame)}")
    print(f"Đã lưu: {args.output}")


if __name__ == "__main__":
    main()
