#!/usr/bin/env python3
"""Train and evaluate MNRL-only ablations for the v2 revision.

This runner is intentionally separate from src/main.py so reviewer-requested
ablation work does not change the original training entrypoint. It requires the
project ML dependencies from requirements.txt plus sentence-transformers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from data.processor import prepare_datasets, prepare_ir_eval_data  # noqa: E402
from evaluation.evaluator import create_matryoshka_evaluator  # noqa: E402
from models.utils import get_matryoshka_dimensions  # noqa: E402


def train_mnrl_only(model_id: str, train_dataset, evaluator, output_dir: Path, device: str):
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(model_id, device=device)
    train_loss = MultipleNegativesRankingLoss(model)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=4,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        optim="adamw_torch" if device in {"mps", "cpu"} else "adamw_torch_fused",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dim_128_cosine_ndcg@10",
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["anchor", "positive"]),
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()
    trainer.save_model()
    return str(output_dir)


def evaluate_model(model_path: str, dims: list[int], queries, corpus, relevant_docs, output_csv: Path, model_name: str):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.util import cos_sim

    model = SentenceTransformer(model_path)
    max_dim = model.get_sentence_embedding_dimension()
    rows = []
    for dim in dims:
        if dim > max_dim:
            continue
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,
            score_functions={"cosine": cos_sim},
        )
        scores = evaluator(model)
        row = {"Model": model_name, "Loss": "MNRL", "Dimension": dim}
        for metric_key, value in scores.items():
            if metric_key.startswith(f"dim_{dim}_cosine_"):
                row[metric_key.removeprefix(f"dim_{dim}_cosine_")] = f"{value:.6f}"
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--model-map", required=True, type=Path, help='JSON like {"MPNet":"sentence-transformers/..."}')
    parser.add_argument("--output-dir", default=Path("paper/v2/ablation_models"), type=Path)
    parser.add_argument("--results-dir", default=Path("paper/v2/results"), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-training", action="store_true", help="Evaluate existing output-dir/model-name directories.")
    args = parser.parse_args()

    dims = get_matryoshka_dimensions()
    train_dataset, test_dataset = prepare_datasets(args.train, args.validation, args.test)
    corpus, queries, relevant_docs = prepare_ir_eval_data(train_dataset, test_dataset)
    evaluator = create_matryoshka_evaluator(queries, corpus, relevant_docs, dims)
    model_map = json.loads(args.model_map.read_text(encoding="utf-8"))

    for short_name, model_id in model_map.items():
        model_dir = args.output_dir / f"{short_name}-mnrl"
        if not args.skip_training:
            train_mnrl_only(model_id, train_dataset, evaluator, model_dir, args.device)
        evaluate_model(
            str(model_dir),
            dims,
            queries,
            corpus,
            relevant_docs,
            args.results_dir / f"{short_name}_mnrl_ablation.csv",
            short_name,
        )


if __name__ == "__main__":
    main()
