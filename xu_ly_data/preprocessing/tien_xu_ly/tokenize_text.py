
import argparse
from pathlib import Path
import re

if __package__:
    from .common import DATA_DIR, read_csv, require_columns, run_cli, write_csv
    from .normalize_text import load_rules
else:
    from common import DATA_DIR, read_csv, require_columns, run_cli, write_csv
    from normalize_text import load_rules

SPECIAL_TOKENS = re.compile(r"(\bemoji_\w+\b|\b(?:url|email|user)_token\b)")


def tokenize_text(text, rules, remove_stopwords=False, tokenizer=None):
    if tokenizer is None:
        from underthesea import word_tokenize
        tokenizer = lambda part: word_tokenize(part, format="text")
    parts = SPECIAL_TOKENS.split(text)
    tokens = []
    for part in parts:
        if not part.strip():
            continue
        segmented = part if SPECIAL_TOKENS.fullmatch(part) else tokenizer(part)
        tokens.extend(segmented.split())
    if remove_stopwords:
        stop = set(rules["stopwords"]) - set(rules["protected_words"])
        tokens = [token for token in tokens if token not in stop]
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "02_normalized.csv")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "03_processed.csv")
    parser.add_argument("--remove-stopwords", action="store_true", help="Tùy chọn, mặc định giữ stop words")
    args = parser.parse_args()
    frame = read_csv(args.input)
    require_columns(frame, ["text_normalized", "label"])
    rules = load_rules()
    frame["text_processed"] = frame["text_normalized"].map(
        lambda text: tokenize_text(text, rules, args.remove_stopwords))
    write_csv(frame, args.output)
    print(f"Đã tách từ {len(frame)} dòng: {args.output}")


if __name__ == "__main__":
    run_cli(main)
