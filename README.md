# Bangla Embedding Model Fine-Tuning and Evaluation

This project provides a modular pipeline for fine-tuning, evaluating, and analyzing Bengali Sentence Transformer models using Matryoshka Representation Learning (MRL).

## Project Structure

```text
src/
├── analysis/
│   ├── efficiency.py       # Storage vs Performance analysis
│   ├── stats.py            # Wilcoxon Signed-Rank Test
│   └── visualization.py    # Plotting functions (Radar charts, Bar plots)
├── data/
│   ├── loader.py           # Loading raw JSON data
│   └── processor.py        # Dataset splitting and formatting
├── evaluation/
│   ├── evaluator.py        # Custom Information Retrieval Evaluator
│   ├── mteb_eval.py        # MTEB Task execution
├── models/
│   ├── trainer.py          # Fine-tuning logic (Matryoshka loss)
│   └── utils.py            # Model loading utilities
└── main.py                 # CLI entry point
```

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables:
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Add your Hugging Face token to `.env` if you plan to push models or access private datasets.

## Usage

The `src/main.py` script is the entry point for all operations.

### 1. Train the Model

Fine-tune a base model on the BanglaRQA dataset.

```bash
python src/main.py --mode train \
    --model_id "shihab17/bangla-sentence-transformer" \
    --output_dir "output/bangla-bert-ft" \
    --hub_model_id "your-username/bangla-embedding-ft"
```

*   `--hub_model_id`: Optional. If provided along with `HF_TOKEN` in `.env`, the model will be pushed to the Hugging Face Hub after training.

### 2. Run MTEB Evaluation

Evaluate the model on Massive Text Embedding Benchmark (MTEB) tasks for Bengali.

```bash
python src/main.py --mode mteb \
    --model_id "output/bangla-bert-ft" \
    --output_dir "results/mteb"
```

Results will be saved as JSON files in the output directory, and an aggregated CSV `mteb_aggregated_results.csv` will be generated.

### 3. Statistical Significance Testing

Run Wilcoxon Signed-Rank Test to compare your model against others.

```bash
python src/main.py --mode stats \
    --model_id "output/bangla-bert-ft" \
    --output_dir "results/stats"
```

Results will be saved to `results/stats/significance_test_results.csv`.

### 4. Efficiency Analysis

Analyze the trade-off between storage size (embedding dimension) and performance.

```bash
python src/main.py --mode efficiency \
    --model_id "output/bangla-bert-ft" \
    --output_dir "results/efficiency"
```

Results will be saved to `results/efficiency/efficiency_results.csv` and a plot `efficiency.png` will be generated.

## Outputs

*   **Training**: Saved model in `output_dir`.
*   **MTEB**: JSON results and `mteb_aggregated_results.csv`.
*   **Stats**: `significance_test_results.csv`.
*   **Efficiency**: `efficiency_results.csv` and `efficiency.png`.
