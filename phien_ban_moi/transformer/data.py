"""Đọc bộ chia hiện có trong data/processed; không chia lại, không đổi nhãn."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
TEXT_COLUMN = "text_clean"  # ViSoBERT có tokenizer riêng, không dùng text_processed đã tách từ.


def load_splits(data_dir=BASE_DIR / "data" / "processed"):
    pieces = {}
    for name in SPLITS:
        frame = pd.read_csv(Path(data_dir) / f"{name}.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        missing = {TEXT_COLUMN, "text_processed", "label", "is_synthetic", "record_id", "post_id"} - set(frame)
        if missing:
            raise ValueError(f"{name}.csv thiếu cột: {sorted(missing)}")
        frame["label"] = frame.label.astype(int)
        frame["is_synthetic"] = frame.is_synthetic.str.strip().str.lower().isin(["true", "1"])
        if set(frame.label) != {0, 1}:
            raise ValueError(f"{name}.csv cần đủ hai lớp.")
        pieces[name] = frame.reset_index(drop=True)
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(pieces[left].post_id) & set(pieces[right].post_id):
            raise ValueError(f"{left} và {right} có chung bài viết; bộ chia không hợp lệ.")
    return pieces


class Encoded:
    """Token hóa một lần; trả về chỉ số để DataLoader lấy theo thứ tự ROS hoặc gốc."""

    def __init__(self, texts, labels, tokenizer, max_length):
        encoded = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.input_ids = encoded["input_ids"]
        self.labels = np.asarray(labels, dtype=int)
        self.truncated = int(sum(len(ids) >= max_length for ids in self.input_ids))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], int(self.labels[index])


def make_collate(pad_id):
    import torch

    def collate(batch):
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (tokens, _) in enumerate(batch):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

    return collate
