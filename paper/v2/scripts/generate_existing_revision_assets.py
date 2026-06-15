#!/usr/bin/env python3
"""Generate v2 revision assets that can be computed from checked-in CSV files."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / "paper" / "v2"
SOURCE = V2 / "SourceFiles"
OUT = V2 / "results"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, 1.96 * math.sqrt(var) / math.sqrt(len(values))


def write_ragas_aggregate() -> None:
    rows = read_csv(ROOT / "results" / "full_results_df.csv")
    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out_rows: list[dict[str, object]] = []
    for model in sorted(by_model):
        model_rows = by_model[model]
        out: dict[str, object] = {"model": model, "n": len(model_rows)}
        for metric in metrics:
            vals = [float(row[metric]) for row in model_rows]
            mean, ci = mean_ci95(vals)
            out[f"{metric}_mean"] = f"{mean:.4f}"
            out[f"{metric}_ci95"] = f"{ci:.4f}"
        out_rows.append(out)

    out_path = OUT / "ragas_aggregate.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # Heatmap with explicit axes.
    models = ["mpnet-base-ft", "shihab17-ft", "distiluse-ft"]
    data = []
    for model in models:
        model_rows = by_model[model]
        data.append([sum(float(row[m]) for row in model_rows) / len(model_rows) for m in metrics])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Aggregate RAGAS Scores by Retriever Model")
    ax.set_xlabel("RAGAS Metric")
    ax.set_ylabel("Retriever Model")
    ax.set_xticks(range(len(metrics)), [m.replace("_", " ").title() for m in metrics], rotation=20, ha="right")
    ax.set_yticks(range(len(models)), models)
    for y, row in enumerate(data):
        for x, value in enumerate(row):
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Mean Score")
    fig.tight_layout()
    fig.savefig(SOURCE / "ragas_aggregate_heatmap.png", dpi=300)
    plt.close(fig)


def plot_primary_results() -> None:
    rows = read_csv(ROOT / "results" / "base_model_vs_ft_model_resutls.csv")
    metrics = ["ndcg@10", "mrr@10", "map@100", "accuracy@1"]
    models = ["MPNet", "shihab17", "distiluse"]
    colors = {"MPNet": "#2f8f2f", "shihab17": "#2b79a8", "distiluse": "#c43c3c"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    fig.suptitle("Fine-Tuned Dense Retrieval Performance Across Dimensions")
    for ax, metric in zip(axes.flat, metrics):
        for model in models:
            pts = [
                (int(row["Dimension"]), float(row[metric]))
                for row in rows
                if row["Model"] == model and row["Type"] == "finetuned"
            ]
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2, label=model, color=colors[model])
        ax.set_title(metric.upper())
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.25)
        ax.set_xticks([64, 128, 256, 512, 768])
        ax.set_ylim(0, 0.75)
    axes.flat[0].legend(title="Model", loc="lower right")
    fig.tight_layout()
    fig.savefig(SOURCE / "dense_performance_v2.png", dpi=300)
    plt.close(fig)


def plot_base_vs_ft() -> None:
    rows = read_csv(ROOT / "results" / "base_model_vs_ft_model_resutls.csv")
    models = ["MPNet", "shihab17", "distiluse"]
    metrics = [("ndcg@10", "NDCG@10"), ("recall@10", "Recall@10")]
    colors = {"base": "#2b8fd8", "finetuned": "#e84b3c"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=False)
    fig.suptitle("Base vs Fine-Tuned Retrieval Performance")
    for row_idx, (metric, title) in enumerate(metrics):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx][col_idx]
            for typ in ["base", "finetuned"]:
                pts = [
                    (int(row["Dimension"]), float(row[metric]))
                    for row in rows
                    if row["Model"] == model and row["Type"] == typ
                ]
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2, label=typ.title(), color=colors[typ])
            ax.set_title(f"{title}: {model}")
            ax.set_xlabel("Embedding Dimension")
            ax.set_ylabel("Score")
            ax.set_xticks([64, 128, 256, 512, 768])
            ax.grid(alpha=0.25)
    axes[0][0].legend(title="Model Type", loc="lower right")
    fig.tight_layout()
    fig.savefig(SOURCE / "base_vs_finetuned_v2.png", dpi=300)
    plt.close(fig)


def main() -> None:
    write_ragas_aggregate()
    plot_primary_results()
    plot_base_vs_ft()
    print(f"Wrote revision assets to {SOURCE} and {OUT}")


if __name__ == "__main__":
    main()
