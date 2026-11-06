"""Dữ liệu tự soạn để kiểm tra cách ly nhãn, lineage và chống rò rỉ; không dùng báo cáo thật."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from phien_ban_moi.preprocessing.prepare_labeled_data import prepare_data
from phien_ban_moi.preprocessing.split_labeled_data import make_splits
from phien_ban_moi.training.sampling import random_oversample_indices
from phien_ban_moi.training.experiments import run_training


def real_fixture():
    return pd.DataFrame([{
        "record_id": f"r:{post}:{label}:{j}", "source_key": f"p{post}|{label}:{j}",
        "comment_id": f"{label}:{j}", "post_id": f"p{post}", "post_url": f"https://www.facebook.com/page/posts/{post}/",
        "text_processed": f"{'xúc_phạm' if label else 'bình_thường'} độc_lập_{post}_{label}_{j}",
        "text_key": f"gốc {post} {label} {j}", "text_hash": f"hash{post}_{label}_{j}",
        "label": label, "is_synthetic": False,
    } for post in range(10) for label in (0, 1) for j in range(6)])


def test_sampling_balances_train_without_altering_original_array():
    y = np.array([0] * 20 + [1] * 2)
    original = y.copy()
    indices = random_oversample_indices(y, 42)
    assert np.array_equal(y, original)
    assert len(indices) == 40
    assert set(indices) == set(range(len(y)))
    assert (y[indices] == 0).sum() == (y[indices] == 1).sum()
    assert np.array_equal(indices, random_oversample_indices(y, 42))


def test_preparation_keeps_review_flags_and_quarantines_conflicts(tmp_path):
    data = pd.DataFrame([
        {"text_raw": text, "label": label, "comment_id": ident,
         "post_url": "https://www.facebook.com/page/posts/123/", "is_synthetic": "False",
         "needs_review": "True", "review_status": "nhãn máy cần duyệt"}
        for text, label, ident in [("Phim hay ❤️", "non-toxic", "1"), ("Phim hay ❤️", "non-toxic", "2"),
                                  ("Mày ngu", "toxic", "3"), ("Mâu thuẫn", "toxic", "4"),
                                  ("Mâu thuẫn", "non-toxic", "5")]
    ])
    path = tmp_path / "labels.csv"
    data.to_csv(path, index=False)
    real, synthetic, audit, rules = prepare_data(path, tmp_path / "prepared", tokenizer=lambda text: text)
    assert len(real) == 2 and synthetic.empty
    assert set(real.label) == {0, 1}
    assert real.needs_review.eq("True").all()
    assert real.review_status.eq("nhãn máy cần duyệt").all()
    assert audit["exclusions"]["conflicting_original_labels"] == 2
    assert audit["exclusions"]["duplicate_original_content"] == 1
    assert "emoji_red_heart" in real.text_processed.iloc[0]
    assert path.read_text().count("Mâu thuẫn") == 2


def test_split_keeps_all_synthetic_out_of_eval_and_uses_only_train_parents(tmp_path):
    real = real_fixture()
    synthetic = pd.DataFrame([
        {**row.to_dict(), "record_id": f"s:{i}", "is_synthetic": True,
         "parent_source_key": row.source_key, "text_key": f"biến thể {i}",
         "text_processed": f"biến_thể_{i}"}
        for i, row in real.loc[real.label.eq(1)].iterrows()
    ])
    pieces, eligible, stats = make_splits(real, synthetic, tmp_path)
    assert len(eligible) > 0
    assert eligible.parent_source_key.isin(pieces["train"].source_key).all()
    assert set(eligible.post_id) <= set(pieces["train"].post_id)
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(pieces[left].post_id) & set(pieces[right].post_id)
        assert not set(pieces[left].text_key) & set(pieces[right].text_key)
    assert not pieces["test"].is_synthetic.any()
    assert not pieces["validation"].is_synthetic.any()
    assert sum(len(piece) for piece in pieces.values()) == len(real)
    assert set(eligible.parent_source_key).isdisjoint(set(pieces["test"].source_key))


def test_vectorizers_never_learn_validation_tokens(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    real = real_fixture()
    pieces = {"train": real.iloc[:96].copy(), "validation": real.iloc[96:].copy()}
    pieces["validation"]["text_processed"] += " unseen_validation_token"
    synthetic = real.iloc[:0].copy()
    choices, models, best = run_training(pieces, synthetic, tmp_path, {"test_rules": True})
    assert len(choices) == len(models) == 12
    assert best == choices.loc[choices.validation_macro_f1.idxmax(), "model"]
    for model in models.values():
        assert "unseen_validation_token" not in model["vectorizer"].vocabulary_
    marker = json.loads((tmp_path / "selected_model_before_test.json").read_text())
    assert marker["test_used_for_selection"] is False
