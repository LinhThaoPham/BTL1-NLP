"""So sánh ViSoBERT với NB/SVM trên cùng test; sinh bảng, hình, ví dụ cho báo cáo."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import BASE_DIR, load_splits

SERIES = {"full": "#2a78d6", "collected": "#eb6834"}  # categorical slot 1, 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def latest(name):
    return Path(json.loads((BASE_DIR / "results" / name).read_text(encoding="utf-8"))["run_dir"])


def short_name(row):
    tag = "A" if row.scenario == "original_imbalanced" else "B"
    if row.feature == "ViSoBERT":
        return f"ViSoBERT ({tag})"
    return f"{row.feature} + {row.algorithm} ({tag})"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def comparison_figure(table, path):
    table = table.sort_values("test_macro_f1")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    h = 0.36
    ax.barh(y + h / 2, table.test_macro_f1, height=h - 0.04, color=SERIES["full"], label="Toàn bộ test (656 mẫu)")
    ax.barh(y - h / 2, table.collected_test_macro_f1, height=h - 0.04, color=SERIES["collected"],
            label="Chỉ bình luận thu thập (596 mẫu, 8 Toxic)")
    for i, row in enumerate(table.itertuples()):
        ax.text(row.test_macro_f1 + 0.01, i + h / 2, f"{row.test_macro_f1:.3f}", va="center", fontsize=8, color=INK)
        ax.text(row.collected_test_macro_f1 + 0.01, i - h / 2, f"{row.collected_test_macro_f1:.3f}", va="center",
                fontsize=8, color=MUTED)
    ax.set_yticks(y, table.name)
    for label in ax.get_yticklabels():
        if label.get_text().startswith("ViSoBERT"):
            label.set_fontweight("bold")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Macro-F1", color=INK)
    style(ax)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_figure(matrices, titles, path):
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 3.6))
    for ax, matrix, title in zip(np.atleast_1d(axes), matrices, titles):
        matrix = np.asarray(matrix)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max() * 1.15)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=12,
                    color="white" if value > matrix.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["Non Toxic", "Toxic"])
        ax.set_yticks([0, 1], ["Non Toxic", "Toxic"])
        ax.set_xlabel("Dự đoán", fontsize=9)
        ax.set_ylabel("Nhãn", fontsize=9)
        ax.set_title(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def history_figure(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    names = {"original_imbalanced": "A — gốc", "original_ros": "B — ROS"}
    for (scenario, part), color in zip(history.groupby("scenario", sort=False), SERIES.values()):
        axes[0].plot(part.epoch, part.train_loss, marker="o", markersize=5, linewidth=2, color=color, label=names[scenario])
        axes[1].plot(part.epoch, part.validation_macro_f1, marker="o", markersize=5, linewidth=2, color=color,
                     label=names[scenario])
    axes[0].set(title="Loss trên train", xlabel="Epoch")
    axes[1].set(title="Macro-F1 trên validation", xlabel="Epoch")
    for ax in axes:
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xticks(sorted(history.epoch.unique()))
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(output=None):
    visobert_dir, baseline_dir = latest("latest_visobert.json"), latest("latest_baseline_splits.json")
    output = Path(output) if output else visobert_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.concat([pd.read_csv(baseline_dir / "metrics.csv"), pd.read_csv(visobert_dir / "metrics.csv")],
                      ignore_index=True)
    table["name"] = table.apply(short_name, axis=1)
    table.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    comparison_figure(table, output / "fig_macro_f1.png")

    selected_vis = json.loads((visobert_dir / "selected_model_before_test.json").read_text(encoding="utf-8"))["selected_model"]
    selected_base = json.loads((baseline_dir / "run_status.json").read_text(encoding="utf-8"))["selected_model"]
    conf = lambda d, m, kind="": json.loads((d / "predictions" / f"{m}{kind}_confusion.json").read_text())
    confusion_figure([conf(baseline_dir, selected_base), conf(visobert_dir, selected_vis)],
                     [table.set_index("model").loc[selected_base, "name"] + " — toàn bộ test",
                      table.set_index("model").loc[selected_vis, "name"] + " — toàn bộ test"],
                     output / "fig_confusion.png")
    confusion_figure([conf(baseline_dir, selected_base, "_collected"), conf(visobert_dir, selected_vis, "_collected")],
                     [table.set_index("model").loc[selected_base, "name"] + " — thu thập",
                      table.set_index("model").loc[selected_vis, "name"] + " — thu thập"],
                     output / "fig_confusion_collected.png")
    history_figure(pd.read_csv(visobert_dir / "training_history.csv"), output / "fig_history.png")

    test = load_splits()["test"]
    nb = pd.read_csv(baseline_dir / "predictions" / f"{selected_base}.csv")
    vis = pd.read_csv(visobert_dir / "predictions" / f"{selected_vis}.csv")
    merged = test[["record_id", "text_clean", "label", "is_synthetic"]].merge(
        nb[["record_id", "predicted_label"]].rename(columns={"predicted_label": "baseline_pred"}), on="record_id").merge(
        vis[["record_id", "predicted_label", "score_toxic"]].rename(columns={"predicted_label": "visobert_pred"}), on="record_id")
    merged["case"] = np.select(
        [merged.baseline_pred.ne(merged.label) & merged.visobert_pred.eq(merged.label),
         merged.baseline_pred.eq(merged.label) & merged.visobert_pred.ne(merged.label),
         merged.baseline_pred.ne(merged.label) & merged.visobert_pred.ne(merged.label)],
        ["visobert_dung_baseline_sai", "baseline_dung_visobert_sai", "ca_hai_sai"], "ca_hai_dung")
    merged.to_csv(output / "error_comparison.csv", index=False, encoding="utf-8-sig")
    summary = {"selected_visobert": selected_vis, "selected_baseline": selected_base,
               "cases": merged.case.value_counts().to_dict(),
               "cases_collected": merged.loc[~merged.is_synthetic].case.value_counts().to_dict()}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    run(parser.parse_args().output)
