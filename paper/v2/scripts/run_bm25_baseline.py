#!/usr/bin/env python3
"""Evaluate a BM25 lexical baseline on BanglaRQA JSON splits.

This script intentionally uses only the Python standard library. It expects the
original BanglaRQA JSON files with the nested structure used by src/data/loader.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_banglarqa(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for article in data["data"]:
        for qa in article["qas"]:
            rows.append(
                {
                    "passage_id": article["passage_id"],
                    "context": article["context"],
                    "question_id": qa["question_id"],
                    "question_text": qa["question_text"],
                    "is_answerable": str(qa["is_answerable"]),
                }
            )
    return rows


def prepare(train_path: Path, val_path: Path, test_path: Path, include_train_in_pool: bool) -> tuple[list[str], list[str], dict[int, set[int]]]:
    train = [r for r in load_banglarqa(train_path) + load_banglarqa(val_path) if r["is_answerable"] == "1"]
    test = [r for r in load_banglarqa(test_path) if r["is_answerable"] == "1"]
    pool = train + test if include_train_in_pool else test
    corpus = [r["context"] for r in pool]
    queries = [r["question_text"] for r in test]

    passage_to_doc_ids: dict[str, set[int]] = defaultdict(set)
    for idx, row in enumerate(pool):
        passage_to_doc_ids[row["passage_id"]].add(idx)
    relevant = {idx: passage_to_doc_ids[row["passage_id"]] for idx, row in enumerate(test)}
    return corpus, queries, relevant


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in docs]
        self.doc_len = [len(doc) for doc in self.docs]
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        self.tf = [Counter(doc) for doc in self.docs]
        df: Counter[str] = Counter()
        for doc_terms in self.tf:
            df.update(doc_terms.keys())
        n_docs = len(self.docs)
        self.idf = {term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def score(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        scores = [0.0] * len(self.docs)
        for term in q_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, freqs in enumerate(self.tf):
                f = freqs.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf * f * (self.k1 + 1) / denom
        return scores


def metrics_for_query(ranked: list[int], relevant: set[int], k_values: tuple[int, ...] = (1, 3, 5, 10, 100)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in k_values:
        top = ranked[:k]
        hits = [1 if doc_id in relevant else 0 for doc_id in top]
        out[f"accuracy@{k}"] = 1.0 if any(hits) else 0.0
        out[f"precision@{k}"] = sum(hits) / k
        out[f"recall@{k}"] = sum(hits) / len(relevant) if relevant else 0.0
    rr = 0.0
    for rank, doc_id in enumerate(ranked[:10], start=1):
        if doc_id in relevant:
            rr = 1.0 / rank
            break
    out["mrr@10"] = rr
    ap = 0.0
    hits_seen = 0
    for rank, doc_id in enumerate(ranked[:100], start=1):
        if doc_id in relevant:
            hits_seen += 1
            ap += hits_seen / rank
    out["map@100"] = ap / len(relevant) if relevant else 0.0
    dcg = sum((1.0 / math.log2(rank + 2)) for rank, doc_id in enumerate(ranked[:10]) if doc_id in relevant)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", default=Path("paper/v2/results/bm25_results.csv"), type=Path)
    parser.add_argument("--test-only-pool", action="store_true", help="Use only test passages in the retrieval pool.")
    args = parser.parse_args()

    corpus, queries, relevant = prepare(args.train, args.validation, args.test, include_train_in_pool=not args.test_only_pool)
    bm25 = BM25(corpus)
    per_query = []
    for q_idx, query in enumerate(queries):
        scores = bm25.score(query)
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
        per_query.append(metrics_for_query(ranked, relevant[q_idx]))

    keys = sorted(per_query[0].keys())
    summary = {key: sum(row[key] for row in per_query) / len(per_query) for key in keys}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Pool", "Queries", "Corpus"] + keys)
        writer.writeheader()
        writer.writerow(
            {
                "Model": "BM25",
                "Pool": "test-only" if args.test_only_pool else "train+validation+test",
                "Queries": len(queries),
                "Corpus": len(corpus),
                **{key: f"{summary[key]:.6f}" for key in keys},
            }
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
