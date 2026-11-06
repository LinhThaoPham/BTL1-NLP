"""Biểu đồ có thể tái tạo từ các CSV kết quả; số liệu thật, không tạo điểm mẫu."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from ..training.experiments import SCENARIOS

COLORS = ["#2563EB", "#B7791F", "#B24773"]


def make_figures(scores, matrices, y, selected_scores, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    order = ["BoW_NB", "BoW_SVM", "TF-IDF_NB", "TF-IDF_SVM"]
    scores = scores.copy()
    scores["configuration"] = scores.feature + "_" + scores.algorithm
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, measure, title in zip(axes, ["test_macro_f1", "test_toxic_f1"], ["Macro-F1", "F1 lớp Toxic"]):
        x = np.arange(4)
        for index, (scenario, label) in enumerate(SCENARIOS.items()):
            values = scores.loc[scores.scenario.eq(scenario)].set_index("configuration")[measure].reindex(order)
            ax.bar(x + (index-1)*.24, values, width=.24, label=label, color=COLORS[index])
        ax.set(xticks=x, xticklabels=order, ylim=(0, 1), title=title, ylabel="Điểm trên test gốc")
        ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=10)
    fig.suptitle("4 cấu hình × 3 cách huấn luyện — cùng tập test gốc", fontweight="bold")
    fig.tight_layout(rect=(0, .1, 1, .95))
    fig.savefig(output / "model_comparison.png", dpi=150)
    plt.close(fig)
    for scenario, label in SCENARIOS.items():
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        selected = scores.loc[scores.scenario.eq(scenario)]
        for ax, (_, row) in zip(axes.ravel(), selected.iterrows()):
            ConfusionMatrixDisplay(matrices[row.model], display_labels=["Non Toxic", "Toxic"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(f"{row.feature} + {row.algorithm}")
        fig.suptitle(label + " — test gốc", fontweight="bold")
        fig.tight_layout()
        fig.savefig(output / f"confusion_{scenario}.png", dpi=140)
        plt.close(fig)
    balance = pd.read_csv(output.parent / "train_class_balance.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (scenario, label) in zip(axes, SCENARIOS.items()):
        part = balance.loc[balance.scenario.eq(scenario)].set_index("label").reindex([0, 1])
        ax.bar([-.16, .84], part.before_ros, width=.32, label="Trước ROS", color=COLORS[0])
        ax.bar([.16, 1.16], part.after_ros, width=.32, label="Sau ROS", color=COLORS[1])
        ax.set(xticks=[0, 1], xticklabels=["Non Toxic", "Toxic"], title=label, ylabel="Số lượt mẫu train")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(output / "class_balance.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y, selected_scores, ax=ax, name="Mô hình chọn bằng validation")
    prevalence = np.mean(np.asarray(y) == 1)
    ax.axhline(prevalence, ls="--", color="#64748B", label=f"Tỷ lệ Toxic test = {prevalence:.3f}")
    ax.legend(fontsize=9)
    ax.set_title("Precision–Recall trên test gốc (lớp Toxic)")
    fig.tight_layout()
    fig.savefig(output / "selected_precision_recall.png", dpi=150)
    plt.close(fig)
