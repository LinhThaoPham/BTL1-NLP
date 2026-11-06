"""Bước 1: làm sạch cơ bản; sau đó người dùng điền label trong file đầu ra."""
import argparse
from pathlib import Path
import unicodedata

import pandas as pd

from common import BASE_DIR, DATA_DIR, read_csv, run_cli, write_csv


def clean_text(text):
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def clean_data(frame):
    frame = frame.copy()
    source = next((col for col in ("text_raw", "comment", "Comment") if col in frame), None)
    if source is None:
        raise ValueError("Cần cột comment, Comment hoặc text_raw.")
    frame["text_raw"] = frame[source]
    frame["text_clean"] = frame["text_raw"].map(clean_text)
    if "label" not in frame:
        frame["label"] = ""
    frame = frame.loc[frame["text_clean"].ne("")].copy()
    if {"comment_id", "post_url"}.issubset(frame.columns):
        identified = frame["comment_id"].str.strip().ne("") & frame["post_url"].str.strip().ne("")
        known = frame.loc[identified]
        for _, group in known.groupby(["post_url", "comment_id"]):
            labels = set(group["label"].str.strip()) - {""}
            if len(labels) > 1:
                raise ValueError("Cùng ID bình luận có nhãn khác nhau; duyệt lại trước khi làm sạch.")
        # Ưu tiên dòng đã có nhãn; không gộp nhầm các dòng thiếu ID/nguồn.
        known = known.assign(_labeled=known["label"].str.strip().ne(""))
        known = known.sort_values("_labeled").drop_duplicates(["post_url", "comment_id"], keep="last")
        frame = pd.concat([known.drop(columns="_labeled"), frame.loc[~identified]]).sort_index()
    order = ["text_raw", "text_clean", "label"]
    return frame[order + [col for col in frame if col not in order]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=BASE_DIR.parent / "crawl data" / "facebook_comments_all.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "01_clean.csv")
    args = parser.parse_args()
    original = read_csv(args.input)
    cleaned = clean_data(original)
    write_csv(cleaned, args.output)
    print(f"Ban đầu {len(original)} dòng; giữ {len(cleaned)}; bỏ rỗng/trùng ID {len(original) - len(cleaned)}.")
    print(f"Đã lưu {args.output}. Đọc text_raw và điền label: 0 = Non Toxic, 1 = Toxic.")


if __name__ == "__main__":
    run_cli(main)
