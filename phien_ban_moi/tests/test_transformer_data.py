"""Kiểm tra phần dữ liệu của ViSoBERT; không tải mô hình."""
import numpy as np
import pandas as pd
import pytest

from phien_ban_moi.transformer.data import Encoded, load_splits, make_collate


def write_split(folder, name, posts, labels):
    pd.DataFrame({
        "text_clean": [f"câu {name} {i}" for i in range(len(labels))],
        "text_processed": [f"câu {name} {i}" for i in range(len(labels))],
        "label": labels, "is_synthetic": ["False"] * len(labels),
        "record_id": [f"{name}:{i}" for i in range(len(labels))], "post_id": posts,
    }).to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")


def test_load_splits_parses_types_and_rejects_shared_posts(tmp_path):
    write_split(tmp_path, "train", ["a", "a"], [0, 1])
    write_split(tmp_path, "validation", ["b", "b"], [0, 1])
    write_split(tmp_path, "test", ["c", "c"], [1, 0])
    pieces = load_splits(tmp_path)
    assert pieces["train"].label.dtype.kind == "i"
    assert pieces["test"].is_synthetic.dtype == bool
    write_split(tmp_path, "test", ["a", "c"], [1, 0])
    with pytest.raises(ValueError, match="chung bài"):
        load_splits(tmp_path)


def test_encoded_and_collate_pad_to_longest():
    tokenizer = lambda texts, truncation, max_length: {"input_ids": [[5] * min(len(t), max_length) for t in texts]}
    encoded = Encoded(["ab", "abcdef"], [0, 1], tokenizer, max_length=4)
    assert encoded.truncated == 1
    batch = make_collate(pad_id=1)([encoded[0], encoded[1]])
    assert batch["input_ids"].tolist() == [[5, 5, 1, 1], [5, 5, 5, 5]]
    assert batch["attention_mask"].tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert np.array_equal(batch["labels"].numpy(), [0, 1])


def test_real_processed_splits_are_disjoint_by_post():
    pieces = load_splits()
    assert {name: len(part) for name, part in pieces.items()} == {"train": 5292, "validation": 605, "test": 656}
