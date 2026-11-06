"""Dữ liệu tự soạn cho các bước tiền xử lý riêng; không ghi vào dữ liệu thật."""
from pathlib import Path
import sys

import pandas as pd
import pytest

DIRECTORY = Path(__file__).resolve().parents[2] / "phien_ban_moi" / "preprocessing"
sys.path.insert(0, str(DIRECTORY))
import clean_data
import normalize_text
import tokenize_text
import split_data
from common import read_csv, write_csv


def test_clean_preserves_raw_short_text_labels_and_unknown_ids():
    frame = pd.DataFrame({
        "comment": ["Dở", "", "Dở", "  Hay\n😀  ", "Khác", "Khác nữa"],
        "label": ["1", "", "", "0", "", ""],
        "comment_id": ["1", "2", "1", "3", "", ""],
        "post_url": ["p", "p", "p", "p", "p", "p"],
    })
    result = clean_data.clean_data(frame)
    assert result.text_clean.tolist() == ["Dở", "Hay 😀", "Khác", "Khác nữa"]
    assert result.text_raw.iloc[1] == "  Hay\n😀  "
    assert result.label.iloc[0] == "1"
    assert frame.comment.iloc[3] == "  Hay\n😀  "


def test_clean_accepts_previous_annotation_file_and_rejects_label_conflict():
    original = pd.DataFrame({"Comment": ["Dở"], "label": ["0"]})
    assert clean_data.clean_data(original).text_raw.tolist() == ["Dở"]
    conflict = pd.DataFrame({"comment": ["A", "A"], "comment_id": ["1", "1"],
                             "post_url": ["p", "p"], "label": ["0", "1"]})
    with pytest.raises(ValueError, match="nhãn khác nhau"):
        clean_data.clean_data(conflict)


def test_normalization_preserves_negation_emoji_numbers_and_profanity():
    result = normalize_text.normalize_text("Ko! ngu 100k ❤️ https://example.com", normalize_text.load_rules())
    assert result.startswith("không!")
    assert "ngu" in result and "100k" in result
    assert "emoji_red_heart" in result and "url_token" in result
    assert "http" not in result


def test_special_tokens_survive_tokenization_and_protected_words_survive_stopwords():
    rules = {"stopwords": ["không", "và"], "protected_words": ["không"]}
    seen = []
    def tokenizer(text):
        seen.append(text)
        return text.replace("mọi người", "mọi_người")
    result = tokenize_text.tokenize_text("không và mọi người emoji_red_heart url_token", rules, True, tokenizer)
    assert result == "không mọi_người emoji_red_heart url_token"
    assert all("emoji_red_heart" not in piece and "url_token" not in piece for piece in seen)


def sample_dataset():
    return pd.DataFrame([{"text_raw": f"Gốc {post} {label}", "text_clean": f"Gốc {post} {label}",
        "text_processed": f"nội_dung_{post}_{label}", "post_url": f"https://www.facebook.com/page/posts/{post}/",
        "comment_id": f"{post}_{label}", "label": str(label)}
        for post in range(10) for label in (0, 1)])


def test_shared_splits_have_no_group_or_content_leakage_and_are_repeatable():
    frame = sample_dataset()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    pieces = split_data.split_data(frame)
    assert sum(map(len, pieces.values())) == 20
    groups, contents = [], []
    for name, piece in pieces.items():
        assert set(piece.label) == {"0", "1"}
        groups.append(set(piece.post_id))
        contents.append(set(piece.text_processed))
        pd.testing.assert_frame_equal(piece, split_data.split_data(frame)[name])
    for i in range(3):
        for j in range(i + 1, 3):
            assert not groups[i] & groups[j]
            assert not contents[i] & contents[j]
    assert [len(pieces[name]) for name in ("train", "validation", "test")] == [16, 2, 2]


def test_split_blocks_unlabeled_conflicts_and_insufficient_groups():
    frame = sample_dataset()
    frame.loc[0, "label"] = ""
    with pytest.raises(ValueError, match="nhãn trống"):
        split_data.split_data(frame)
    frame = sample_dataset()
    frame.loc[1, "text_processed"] = frame.loc[0, "text_processed"]
    with pytest.raises(ValueError, match="nhãn khác nhau"):
        split_data.split_data(frame)
    with pytest.raises(ValueError, match="ít nhất 6 bài"):
        split_data.split_data(sample_dataset().iloc[:8])


def test_csv_keeps_empty_labels_and_does_not_overwrite(tmp_path):
    path = tmp_path / "clean.csv"
    write_csv(pd.DataFrame({"text_raw": ["😀"], "label": [""]}), path)
    assert read_csv(path).iloc[0].to_dict() == {"text_raw": "😀", "label": ""}
    with pytest.raises(FileExistsError):
        write_csv(pd.DataFrame(), path)
