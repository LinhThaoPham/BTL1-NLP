"""Ưu tiên Macro-F1 và Recall/F1 Toxic khi dữ liệu mất cân bằng mạnh."""
import math
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)


def evaluate(y, predicted, scores):
    report = classification_report(y, predicted, labels=[0, 1], target_names=["Non Toxic", "Toxic"],
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    measures = {
        "accuracy": accuracy_score(y, predicted),
        "macro_precision": precision_score(y, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(y, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(y, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, predicted, average="weighted", zero_division=0),
        "toxic_precision": report["Toxic"]["precision"], "toxic_recall": report["Toxic"]["recall"],
        "toxic_f1": report["Toxic"]["f1-score"], "toxic_support": int(report["Toxic"]["support"]),
        "average_precision": average_precision_score(y, scores), "roc_auc": roc_auc_score(y, scores),
    }
    # Khoảng Wilson của Recall Toxic chỉ mô tả độ bất định nhị thức; các mẫu cùng bài có thể phụ thuộc.
    n, successes, z = int(matrix[1].sum()), int(matrix[1, 1]), 1.96
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    measures.update(toxic_recall_wilson_low=center-half, toxic_recall_wilson_high=center+half)
    return measures, report, matrix


def positive_scores(pipeline, texts):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts)[:, list(pipeline.classes_).index(1)]
    return np.asarray(pipeline.decision_function(texts))
