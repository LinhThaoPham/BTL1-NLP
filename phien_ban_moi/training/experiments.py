"""Chọn tham số bằng validation gốc; chưa đọc điểm test trong giai đoạn này."""
from itertools import product
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .sampling import random_oversample_indices
from . import train_naive_bayes, train_svm

SCENARIOS = {
    "original_imbalanced": "Gốc — chưa cân bằng",
    "original_ros": "Gốc + Random Oversampling",
    "augmented_ros": "Gốc + tổng hợp hợp lệ + ROS",
}


def run_training(pieces, synthetic_train, output, rules, seed=42, scenarios=None):
    train, validation = pieces["train"], pieces["validation"]
    choices, searches, distribution = [], [], []
    models = {}
    for scenario in (SCENARIOS if scenarios is None else scenarios):
        pool = train if scenario != "augmented_ros" else pd.concat([train, synthetic_train], ignore_index=True)
        y = pool.label.to_numpy(dtype=int)
        sample_idx = np.arange(len(pool)) if scenario == "original_imbalanced" else random_oversample_indices(y, seed)
        # Các dòng lặp nằm trong train; không thao tác validation/test.
        sample_manifest = pool.iloc[sample_idx][["record_id", "label", "is_synthetic"]].copy()
        sample_manifest["sample_position"] = range(len(sample_manifest))
        sample_manifest.to_csv(output / "data" / f"{scenario}_sampling.csv", index=False, encoding="utf-8-sig")
        for label in (0, 1):
            distribution.append({"scenario": scenario, "label": label,
                "before_ros": int((y == label).sum()), "after_ros": int((y[sample_idx] == label).sum()),
                "synthetic_before_ros": int((pool.is_synthetic & pool.label.eq(label)).sum())})
        for feature, vectorizer_class in [("BoW", CountVectorizer), ("TF-IDF", TfidfVectorizer)]:
            vectorized = {}
            for ngram in [(1, 1), (1, 2)]:
                vectorizer = vectorizer_class(lowercase=False, token_pattern=r"(?u)\b\w+\b", ngram_range=ngram)
                # IDF/từ vựng chỉ fit trên train trước ROS, tránh IDF bị đổi do nhân bản mẫu.
                x = vectorizer.fit_transform(pool.text_processed)
                xv = vectorizer.transform(validation.text_processed)
                vectorized[ngram] = (vectorizer, x, xv)
            for algorithm, module in [("NB", train_naive_bayes), ("SVM", train_svm)]:
                name = f"{scenario}__{feature.replace('-', '_')}__{algorithm}"
                best, best_value = None, -1
                started = time.perf_counter()
                for ngram, parameter in product([(1, 1), (1, 2)], module.PARAMETERS):
                    vectorizer, x, xv = vectorized[ngram]
                    classifier = module.make_classifier(parameter, seed)
                    classifier.fit(x[sample_idx], y[sample_idx])
                    predicted = classifier.predict(xv)
                    score = f1_score(validation.label, predicted, average="macro", zero_division=0)
                    searches.append({"model": name, "scenario": scenario, "feature": feature,
                        "algorithm": algorithm, "ngram": str(ngram), "parameter": parameter,
                        "validation_macro_f1": score})
                    if score > best_value:
                        best_value = score
                        best = (Pipeline([("vectorizer", vectorizer), ("classifier", classifier)]), ngram, parameter)
                pipeline, ngram, parameter = best
                bundle = {"pipeline": pipeline, "rules": rules, "label_mapping": {0: "Non Toxic", 1: "Toxic"},
                          "input_column": "text_processed", "scenario": scenario,
                          "note": "Dự đoán văn bản gốc qua training.predict; nhãn thí nghiệm do file đầu vào cung cấp."}
                joblib.dump(bundle, output / "models" / f"{name}.joblib")
                models[name] = pipeline
                choices.append({"model": name, "scenario": scenario, "feature": feature, "algorithm": algorithm,
                    "validation_macro_f1": best_value, "ngram": str(ngram), "parameter": parameter,
                    "fit_seconds": time.perf_counter() - started, "train_before_ros": len(pool),
                    "train_after_ros": len(sample_idx), "vocabulary_size": len(pipeline["vectorizer"].vocabulary_)})
                print(f"{name}: validation Macro-F1={best_value:.4f}", flush=True)
    choices = pd.DataFrame(choices)
    choices.to_csv(output / "validation_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(searches).to_csv(output / "validation_search.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(distribution).to_csv(output / "train_class_balance.csv", index=False, encoding="utf-8-sig")
    best_name = choices.loc[choices.validation_macro_f1.idxmax(), "model"]
    (output / "selected_model_before_test.json").write_text(json.dumps({
        "selected_model": best_name, "criterion": "validation_macro_f1", "test_used_for_selection": False}, indent=2), encoding="utf-8")
    return choices, models, best_name
