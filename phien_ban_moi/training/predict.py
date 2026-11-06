"""Dự đoán văn bản mới bằng model do flow này tạo ra; không nạp joblib không rõ nguồn."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from ..preprocessing.normalize_text import normalize_text
from ..preprocessing.tokenize_text import tokenize_text


def predict(model_path, texts):
    if not texts or any(not str(text).strip() for text in texts):
        raise ValueError("Cần bình luận không rỗng.")
    bundle = joblib.load(model_path)
    rules = bundle["rules"]
    processed = [tokenize_text(normalize_text(text, rules), rules) for text in texts]
    labels = bundle["pipeline"].predict(processed)
    return pd.DataFrame({"text_raw": texts, "text_processed": processed, "predicted_label": labels,
                         "prediction": [bundle["label_mapping"][int(label)] for label in labels]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(predict(args.model, [args.text]).to_string(index=False))
