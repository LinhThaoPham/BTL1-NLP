"""Đọc nhãn được cung cấp, chuẩn hóa, cách ly mâu thuẫn; không đổi cờ duyệt nhãn."""
import hashlib
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .normalize_text import load_rules, normalize_text
from .tokenize_text import tokenize_text

LABEL_MAP = {"0": 0, "1": 1, "non-toxic": 0, "non toxic": 0, "nontoxic": 0, "toxic": 1}


def post_identity(url):
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("story_fbid", "fbid"):
        if query.get(key):
            return "facebook:" + query[key][0]
    parts = parsed.path.strip("/").split("/")
    for name in ("posts", "reel", "videos", "permalink"):
        if name in parts and parts.index(name) + 1 < len(parts):
            return "facebook:" + parts[parts.index(name) + 1]
    return url.strip()


def feature_key(text):
    # Khớp cách vectorizer bỏ dấu câu; bắt cả trùng do khác hoa/thường, dấu câu.
    return " ".join(re.findall(r"(?u)\b\w+\b", text.casefold()))


def prepare_data(path, output, tokenizer=None):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"text_raw", "label", "post_url", "comment_id", "is_synthetic"}
    if not required.issubset(frame):
        raise ValueError(f"Thiếu cột: {sorted(required - set(frame))}")
    frame["input_row"] = range(2, len(frame) + 2)
    labels = frame.label.str.strip().str.lower().map(LABEL_MAP)
    if labels.isna().any():
        raise ValueError("Có nhãn trống hoặc khác toxic/non-toxic/0/1; cần kiểm tra đầu vào.")
    frame["label_original"] = frame.label
    frame["label"] = labels.astype(int)
    flags = frame.is_synthetic.str.strip().str.lower()
    if not flags.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Cột is_synthetic cần True/False hoặc 1/0.")
    frame["is_synthetic"] = flags.isin(["true", "1"])
    for name in ("needs_review", "review_status", "source_comment_id", "synthetic_id", "split_group_id"):
        if name not in frame:
            frame[name] = ""
    frame["post_id"] = frame.post_url.map(post_identity)
    frame["source_key"] = frame.post_id + "|" + frame.comment_id
    frame["record_id"] = [f"input:{i}" for i in frame.input_row]
    frame["text_clean"] = frame.text_raw.map(lambda s: " ".join(str(s).split()))
    rules = load_rules()
    print(f"Chuẩn hóa/tách từ {len(frame)} dòng (quy tắc cố định, không học từ test)...", flush=True)
    cache = {}
    for i, text in enumerate(frame.text_clean, 1):
        if text not in cache:
            normalized = normalize_text(text, rules)
            cache[text] = (normalized, tokenize_text(normalized, rules, tokenizer=tokenizer))
        if i % 1000 == 0:
            print(f"  Đã xử lý {i}/{len(frame)} dòng", flush=True)
    frame["text_normalized"] = frame.text_clean.map(lambda s: cache[s][0])
    frame["text_processed"] = frame.text_clean.map(lambda s: cache[s][1])
    frame["text_key"] = frame.text_processed.map(feature_key)
    frame["text_hash"] = frame.text_key.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    quarantine = []

    def exclude(mask, reason):
        nonlocal frame
        rejected = frame.loc[mask].copy()
        rejected["excluded_reason"] = reason
        quarantine.append(rejected)
        frame = frame.loc[~mask].copy()

    exclude(frame.text_key.eq("") | frame.post_url.str.strip().eq("") | frame.comment_id.str.strip().eq(""), "empty_text_or_source_or_id")
    real = frame.loc[~frame.is_synthetic].copy()
    conflicting_text = set(real.groupby("text_key").label.nunique().loc[lambda s: s.gt(1)].index)
    conflicting_id = set(real.groupby("source_key").label.nunique().loc[lambda s: s.gt(1)].index)
    exclude((~frame.is_synthetic) & (frame.text_key.isin(conflicting_text) | frame.source_key.isin(conflicting_id)), "conflicting_original_labels")
    real = frame.loc[~frame.is_synthetic].copy()
    # Giữ bản đầu theo thứ tự đầu vào, không lựa nhãn dựa trên tần suất.
    duplicate_ids = real.loc[real.duplicated("source_key")].record_id
    exclude(frame.record_id.isin(duplicate_ids), "duplicate_original_id")
    real = frame.loc[~frame.is_synthetic].copy()
    duplicate_text = real.loc[real.duplicated("text_key")].record_id
    exclude(frame.record_id.isin(duplicate_text), "duplicate_original_content")
    real = frame.loc[~frame.is_synthetic].copy()
    synthetic = frame.loc[frame.is_synthetic].copy()
    # Dữ liệu tổng hợp có ID riêng; comment_id trong file có thể là ID cha.
    synthetic["parent_source_key"] = synthetic.post_id + "|" + synthetic.source_comment_id
    synthetic.to_csv(output / "synthetic_candidates.csv", index=False, encoding="utf-8-sig")
    real.to_csv(output / "original_clean.csv", index=False, encoding="utf-8-sig")
    excluded = pd.concat(quarantine, ignore_index=True)
    excluded.to_csv(output / "excluded_rows.csv", index=False, encoding="utf-8-sig")
    stats = {
        "input_rows": len(labels), "original_input": int((~flags.isin(["true", "1"])).sum()),
        "synthetic_input": int(flags.isin(["true", "1"]).sum()),
        "original_used": len(real), "synthetic_candidates": len(synthetic),
        "original_label_counts": {str(k): int(v) for k, v in real.label.value_counts().items()},
        "exclusions": excluded.excluded_reason.value_counts().to_dict(),
        "needs_review_original_input": int(((~flags.isin(["true", "1"])) &
            pd.read_csv(path, dtype=str, keep_default_na=False).get("needs_review", pd.Series("", index=flags.index)).str.lower().eq("true")).sum()),
        "label_status": "Nhãn do file cung cấp; có nhãn ChatGPT và needs_review, chưa xem là ground truth do con người xác nhận.",
    }
    if set(real.label) != {0, 1}:
        raise ValueError("Sau làm sạch không đủ hai lớp gốc; xem excluded_rows.csv.")
    return real.reset_index(drop=True), synthetic.reset_index(drop=True), stats, rules
