"""Thực nghiệm trên tập đầu vào hỗn hợp, chia theo bài trước khi cân bằng train."""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..evaluation.metrics import evaluate, positive_scores
from ..preprocessing.prepare_labeled_data import post_identity, prepare_data
from ..preprocessing.split_labeled_data import make_splits
from .experiments import run_training
from .run_all import BASE_DIR, write_json


def export_processed_data(pieces):
    """Lưu bản dữ liệu hiện tại ở đường dẫn cố định để mở và trình bày."""
    destination = BASE_DIR / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [pieces[name].assign(split=name) for name in ("train", "validation", "test")],
        ignore_index=True,
    )
    combined.to_csv(destination / "comments_preprocessed.csv", index=False, encoding="utf-8-sig")
    for name, part in pieces.items():
        part.to_csv(destination / f"{name}.csv", index=False, encoding="utf-8-sig")
    return destination


def run(source):
    output = BASE_DIR / "results" / ("full_input_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for name in ("data", "models", "predictions"):
        (output / name).mkdir(parents=True)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    original = raw.loc[raw.is_synthetic.str.lower().isin(["false", "0"])]
    parent_keys = set(zip(original.post_url.map(post_identity), original.comment_id))
    real, synthetic, audit, rules = prepare_data(source, output / "data")
    if not all((row.post_id, row.source_comment_id) in parent_keys for row in synthetic.itertuples()):
        raise ValueError("Có biến thể thiếu bình luận nguồn trong cùng nhóm bài.")
    merged = pd.concat([real, synthetic], ignore_index=True)
    conflict = merged.groupby("text_key").label.transform("nunique").gt(1)
    duplicate = merged.loc[~conflict].duplicated("text_key")
    excluded = pd.concat([merged.loc[conflict].assign(excluded_reason="conflicting_combined_labels"),
                          merged.loc[duplicate[duplicate].index].assign(excluded_reason="duplicate_combined_text")])
    excluded.to_csv(output / "data" / "excluded_combined.csv", index=False, encoding="utf-8-sig")
    merged = merged.drop(index=excluded.index).reset_index(drop=True)
    pieces, _, stats = make_splits(merged, synthetic.iloc[:0], output / "data")
    export_processed_data(pieces)
    for name, part in pieces.items():
        stats[name]["synthetic"] = int(part.is_synthetic.sum())
        stats[name]["collected_toxic"] = int((~part.is_synthetic & part.label.eq(1)).sum())
    test = pieces["test"]
    write_json(output / "data_audit.json", {
        **audit, "input_counts": raw.label.value_counts().to_dict(),
        "combined_used": len(merged), "combined_counts": merged.label.value_counts().to_dict(),
        "combined_exclusions": excluded.excluded_reason.value_counts().to_dict(),
    })
    write_json(output / "split_stats.json", {k: stats[k] for k in pieces})
    choices, models, selected = run_training(pieces, synthetic.iloc[:0], output, rules,
                                             scenarios=("original_imbalanced", "original_ros"))
    for name in pieces:
        print(name, stats[name], flush=True)
    results = []
    for choice in choices.to_dict("records"):
        name = choice["model"]
        model = models[name]
        predicted = model.predict(test.text_processed)
        scores = positive_scores(model, test.text_processed)
        measures, _, matrix = evaluate(test.label, predicted, scores)
        result = {**choice, **{"test_" + k: v for k, v in measures.items()}}
        collected = ~test.is_synthetic.to_numpy()
        if set(test.label[collected]) == {0, 1}:
            subset, _, _ = evaluate(test.label[collected], predicted[collected], scores[collected])
            result.update({"collected_test_" + k: v for k, v in subset.items()})
        results.append(result)
        write_json(output / "predictions" / f"{name}_confusion.json", matrix.tolist())
        test[["record_id", "label", "is_synthetic", "post_id"]].assign(
            predicted_label=predicted, score_toxic=scores
        ).to_csv(output / "predictions" / f"{name}.csv", index=False)
    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output / "run_status.json", {"status": "complete", "selected_model": selected,
                                           "evaluation": "mixed_collected_and_augmented"})
    write_json(BASE_DIR / "results" / "latest_full_input.json", {"run_dir": str(output)})
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    run(parser.parse_args().input)
