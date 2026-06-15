# TALLIP Revision v2

This folder contains the reviewer-response revision artifacts.

## Generated data and results

- `data/BanglaRQA/`: public BanglaRQA train, validation, and test JSON files.
- `results/bm25_results.csv`: BM25 lexical baseline on the same answerable BanglaRQA test set.
- `results/ragas_aggregate.csv`: aggregate RAGAS metrics used in the revised RAG section.
- `SourceFiles/*.png`: regenerated reviewer-facing figures with explicit axis labels.

## Reproduction commands

From the repository root:

```bash
python3 paper/v2/scripts/run_bm25_baseline.py \
  --train paper/v2/data/BanglaRQA/Train.json \
  --validation paper/v2/data/BanglaRQA/Validation.json \
  --test paper/v2/data/BanglaRQA/Test.json \
  --output paper/v2/results/bm25_results.csv
```

```bash
paper/v2/.venv/bin/python paper/v2/scripts/run_mnrl_ablation.py \
  --train paper/v2/data/BanglaRQA/Train.json \
  --validation paper/v2/data/BanglaRQA/Validation.json \
  --test paper/v2/data/BanglaRQA/Test.json \
  --model-map paper/v2/config/model_map.json \
  --output-dir paper/v2/ablation_models \
  --results-dir paper/v2/results \
  --device mps
```

Compile from `paper/v2/SourceFiles`:

```bash
tectonic manuscript.tex --keep-logs --keep-intermediates
```
