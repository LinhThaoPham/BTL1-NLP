"""Chạy đủ flow: kiểm tra → tiền xử lý → chia gốc → cân bằng train → 4 cấu hình × 3 kịch bản."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..preprocessing.prepare_labeled_data import prepare_data
from ..preprocessing.split_labeled_data import make_splits
from ..evaluation.metrics import evaluate, positive_scores
from ..evaluation.plots import make_figures
from ..evaluation.report import write_report
from .experiments import run_training

BASE_DIR = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else str(x)), encoding="utf-8")


def run(input_path, output_root=BASE_DIR / "results", seed=42):
    input_path, output_root = Path(input_path).resolve(), Path(output_root)
    original_bytes = input_path.read_bytes()
    source_hash = hashlib.sha256(original_bytes).hexdigest()
    inputs = BASE_DIR / "data" / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    snapshot = inputs / f"labels_{source_hash[:12]}.csv"
    if not snapshot.exists():
        snapshot.write_bytes(original_bytes)
    if snapshot.read_bytes() != original_bytes:
        raise ValueError("Snapshot hiện có khác nguồn; kiểm tra file trước khi chạy.")
    stamp = datetime.now(timezone(timedelta(hours=7))).strftime("%Y%m%d_%H%M%S")
    output = output_root / f"{stamp}_{source_hash[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    for name in ("data", "models", "predictions", "figures", "reports"):
        (output / name).mkdir()
    write_json(output / "run_status.json", {"status": "running"})
    try:
        real, synthetic, audit, rules = prepare_data(snapshot, output / "data")
        write_json(output / "data_audit.json", audit)
        print("Dữ liệu gốc sau làm sạch:", real.label.value_counts().to_dict(), flush=True)
        pieces, synthetic_train, split_stats = make_splits(real, synthetic, output / "data", seed)
        write_json(output / "split_stats.json", split_stats)
        print("Chia tập:", split_stats, flush=True)
        choices, models, best_name = run_training(pieces, synthetic_train, output, rules, seed)
        # Từ đây mới được đánh giá test. Lựa chọn đã ghi vào selected_model_before_test.json.
        test = pieces["test"]
        records, matrices = [], {}
        for _, choice in choices.iterrows():
            name = choice.model
            pipeline = models[name]
            predicted = pipeline.predict(test.text_processed)
            scores = positive_scores(pipeline, test.text_processed)
            measures, report, matrix = evaluate(test.label, predicted, scores)
            records.append({**choice.to_dict(), **{"test_"+k: v for k, v in measures.items()}})
            matrices[name] = matrix
            predictions = test[["record_id", "text_raw", "text_processed", "comment_id", "post_url",
                                "label", "needs_review", "review_status"]].copy()
            predictions["predicted_label"] = predicted
            predictions["score_toxic"] = scores
            predictions.to_csv(output / "predictions" / f"{name}.csv", index=False, encoding="utf-8-sig")
            predictions.loc[predictions.label.ne(predictions.predicted_label)].to_csv(
                output / "predictions" / f"{name}_errors.csv", index=False, encoding="utf-8-sig")
            write_json(output / "predictions" / f"{name}_classification_report.json", report)
            write_json(output / "predictions" / f"{name}_confusion.json", matrix.tolist())
            restored = joblib.load(output / "models" / f"{name}.joblib")
            if not np.array_equal(restored["pipeline"].predict(test.text_processed), predicted):
                raise RuntimeError(f"Model {name} đổi dự đoán sau khi nạp lại.")
        results = pd.DataFrame(records)
        results.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
        baseline_pred = np.zeros(len(test), dtype=int)
        baseline = {"accuracy": accuracy_score(test.label, baseline_pred),
                    "macro_f1": f1_score(test.label, baseline_pred, average="macro", zero_division=0)}
        write_json(output / "majority_baseline.json", baseline)
        make_figures(results, matrices, test.label, positive_scores(models[best_name], test.text_processed), output / "figures")
        write_report(output, audit, split_stats, results, best_name, baseline)
        shutil.copyfile(output / "models" / f"{best_name}.joblib", output / "models" / "selected_model.joblib")
        metadata = {"input_file": str(input_path), "input_snapshot": str(snapshot), "input_sha256": source_hash,
                    "seed": seed, "selected_model": best_name, "selection_metric": "validation_macro_f1",
                    "test_is_original_only": True, "phobert_in_scope": False,
                    "label_mapping": {"0": "Non Toxic", "1": "Toxic"},
                    "label_caveat": audit["label_status"], "rules": rules, "python": platform.python_version(),
                    "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "underthesea", "emoji", "joblib"]}}
        write_json(output / "metadata.json", metadata)
        write_json(output / "run_status.json", {"status": "complete", "models": len(results), "model_reload_verified": True})
        write_json(output_root / "latest.json", {"run_dir": str(output.resolve())})
        print(f"HOÀN THÀNH: {output}", flush=True)
        print(results[["scenario", "feature", "algorithm", "validation_macro_f1", "test_accuracy", "test_macro_f1", "test_toxic_f1"]].to_string(index=False), flush=True)
        return output
    except Exception as error:
        write_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
