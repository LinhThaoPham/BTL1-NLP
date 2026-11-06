"""Chia dữ liệu gốc theo bài; chỉ chọn biến thể có cha còn trong train."""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_splits(real, synthetic, output, seed=42):
    best, best_score = None, float("inf")
    for attempt in range(300):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        a, b = next(outer.split(real, groups=real.post_id))
        holdout = real.iloc[b]
        if holdout.post_id.nunique() < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + attempt)
        c, d = next(inner.split(holdout, groups=holdout.post_id))
        pieces = {"train": real.iloc[a].copy(), "validation": holdout.iloc[c].copy(), "test": holdout.iloc[d].copy()}
        if any(set(part.label) != {0, 1} for part in pieces.values()):
            continue
        if min(int(pieces[name].label.eq(1).sum()) for name in ("validation", "test")) < 5:
            continue
        score = 0
        for name, ratio in zip(("train", "validation", "test"), (0.8, 0.1, 0.1)):
            part = pieces[name]
            score += abs(len(part) / len(real) - ratio)
            for label in (0, 1):
                score += abs(part.label.eq(label).sum() / real.label.eq(label).sum() - ratio)
        if score < best_score:
            best, best_score = pieces, score
    if best is None:
        raise ValueError("Không chia được tập gốc theo bài với ít nhất 5 Toxic ở validation và test; cần thêm dữ liệu gốc.")
    train = best["train"]
    parent_info = train.set_index("source_key")
    original_texts = set(real.text_key)
    candidates = synthetic.copy()
    def reason(row):
        if row.parent_source_key not in parent_info.index:
            return "source_parent_not_in_original_train"
        if row.label != parent_info.loc[row.parent_source_key, "label"]:
            return "label_differs_from_source_parent"
        if row.text_key in original_texts:
            return "duplicates_original_content"
        return "eligible"
    candidates["eligibility"] = candidates.apply(reason, axis=1) if len(candidates) else pd.Series(dtype=str)
    usable = candidates.loc[candidates.eligibility.eq("eligible")].copy()
    conflicting = set(usable.groupby("text_key").label.nunique().loc[lambda s: s.gt(1)].index)
    conflict_ids = set(usable.loc[usable.text_key.isin(conflicting), "record_id"])
    candidates.loc[candidates.record_id.isin(conflict_ids), "eligibility"] = "conflicting_synthetic_labels"
    usable = usable.loc[~usable.text_key.isin(conflicting)]
    duplicate_ids = set(usable.loc[usable.duplicated("text_key"), "record_id"])
    candidates.loc[candidates.record_id.isin(duplicate_ids), "eligibility"] = "duplicate_synthetic_content"
    usable = usable.drop_duplicates("text_key")
    assert not usable.text_key.isin(pd.concat([best["validation"], best["test"]]).text_key).any()
    assert usable.parent_source_key.isin(train.source_key).all()
    assert set(usable.post_id).issubset(set(train.post_id))
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not set(best[left].post_id) & set(best[right].post_id)
        assert not set(best[left].text_key) & set(best[right].text_key)
    manifest = []
    for name, part in best.items():
        part["split"] = name
        part.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
        manifest.append(part[["record_id", "source_key", "post_id", "text_hash", "label", "is_synthetic", "split"]])
    candidates.to_csv(output / "synthetic_eligibility.csv", index=False, encoding="utf-8-sig")
    usable.to_csv(output / "synthetic_train.csv", index=False, encoding="utf-8-sig")
    pd.concat(manifest).to_csv(output / "split_manifest.csv", index=False, encoding="utf-8-sig")
    stats = {name: {"rows": len(part), "posts": part.post_id.nunique(),
                    "labels": {str(k): int(v) for k, v in part.label.value_counts().items()}}
             for name, part in best.items()}
    stats["synthetic_eligibility"] = candidates.eligibility.value_counts().to_dict()
    stats["synthetic_train_rows"] = len(usable)
    return best, usable, stats
