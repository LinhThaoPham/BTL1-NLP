"""Bước 2: chuẩn hóa văn bản cho BoW/TF-IDF; giữ cột gốc và nhãn."""
import argparse
import json
from pathlib import Path
import re
import unicodedata

import emoji

if __package__:
    from .common import BASE_DIR, DATA_DIR, read_csv, require_columns, run_cli, write_csv
else:
    from common import BASE_DIR, DATA_DIR, read_csv, require_columns, run_cli, write_csv


def load_rules(path=BASE_DIR / "text_rules.json"):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_text(text, rules):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " url_token ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " email_token ", text)
    text = re.sub(r"(?<!\w)@[\w.]+", " user_token ", text)
    text = emoji.demojize(text, delimiters=(" emoji_", " "), language="en")
    # Tên emoji có dấu nối/dấu câu cần thành một token duy nhất.
    text = re.sub(r"emoji_(\S+)", lambda m: "emoji_" + re.sub(r"\W", "_", m[1]), text)
    tokens = text.split()
    abbreviations = rules["abbreviations"]
    out = []
    for token in tokens:
        # Giữ dấu câu để còn kiểm tra sắc thái; chuẩn hóa phần từ bên trong.
        match = re.fullmatch(r"([^\w]*)(\w+)([^\w]*)", token)
        if match:
            left, word, right = match.groups()
            token = left + abbreviations.get(word, word) + right
        out.append(token)
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "01_clean.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "02_normalized.csv")
    parser.add_argument("--rules", type=Path, default=BASE_DIR / "text_rules.json")
    args = parser.parse_args()
    frame = read_csv(args.input)
    require_columns(frame, ["text_raw", "text_clean", "label"])
    rules = load_rules(args.rules)
    frame["text_normalized"] = frame["text_clean"].map(lambda text: normalize_text(text, rules))
    write_csv(frame, args.output)
    print(f"Đã chuẩn hóa {len(frame)} dòng: {args.output}")


if __name__ == "__main__":
    run_cli(main)
