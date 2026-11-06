"""Chạy lại 8 cấu hình NB/SVM của BTL1 trên đúng data/processed để so cùng điều kiện với ViSoBERT."""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..evaluation.metrics import evaluate, positive_scores
from ..preprocessing.normalize_text import load_rules
from ..training.experiments import run_training
from .data import BASE_DIR, load_splits
from .train_visobert import write_json


def run(data_dir, output=None):
    output = Path(output) if output else BASE_DIR / "results" / ("baseline_splits_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for name in ("data", "models", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=False)
    pieces = load_splits(data_dir)
    choices, models, selected = run_training(pieces, pieces["train"].iloc[:0], output, load_rules(),
                                             scenarios=("original_imbalanced", "original_ros"))
    test = pieces["test"]
    collected = ~test.is_synthetic.to_numpy()
    records = []
    for choice in choices.to_dict("records"):
        model = models[choice["model"]]
        predicted = model.predict(test.text_processed)
        scores = positive_scores(model, test.text_processed)
        measures, _, matrix = evaluate(test.label, predicted, scores)
        subset, _, sub_matrix = evaluate(test.label[collected], predicted[collected], scores[collected])
        records.append({**choice, **{"test_" + k: v for k, v in measures.items()},
                        **{"collected_test_" + k: v for k, v in subset.items()}})
        write_json(output / "predictions" / f"{choice['model']}_confusion.json", matrix.tolist())
        write_json(output / "predictions" / f"{choice['model']}_collected_confusion.json", sub_matrix.tolist())
        test[["record_id", "label", "is_synthetic", "post_id"]].assign(
            predicted_label=predicted, score_toxic=scores
        ).to_csv(output / "predictions" / f"{choice['model']}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output / "run_status.json", {"status": "complete", "selected_model": selected})
    write_json(BASE_DIR / "results" / "latest_baseline_splits.json", {"run_dir": str(output.resolve())})
    print(pd.DataFrame(records)[["model", "validation_macro_f1", "test_macro_f1", "test_toxic_f1",
                                 "collected_test_macro_f1"]].to_string(index=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "processed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.data_dir, args.output)
