import torch
import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator

def analyze_efficiency(model_map, dims, queries, corpus, relevant_docs, device="cuda"):
    """
    Analyzes storage vs accuracy trade-off.
    """
    corpus_size = len(corpus)
    results_data = []

    print(f"Calculating efficiency for Corpus Size: {corpus_size} documents")

    for model_name, model_id in model_map.items():
        print(f"\nProcessing: {model_name}...")
        try:
            model = SentenceTransformer(model_id, device=device)
            model_max_dim = model.get_sentence_embedding_dimension()

            # Run evaluation for each dimension
            for dim in dims:
                # Skip dimensions larger than model capacity
                if dim > model_max_dim:
                    continue

                # 1. Calculate Storage Size (Float32 = 4 bytes)
                # Size in MB = (Num_Docs * Dim * 4 bytes) / (1024 * 1024)
                storage_mb = (corpus_size * dim * 4) / (1024 ** 2)

                # 2. Evaluate Accuracy (NDCG@10)
                ir_eval = InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=f"eval_{dim}",
                    truncate_dim=dim,
                    score_functions={"cosine": util.cos_sim}
                )

                scores = ir_eval(model)
                ndcg_score = scores[f"eval_{dim}_cosine_ndcg@10"]

                # Store Result
                results_data.append({
                    "Model": model_name,
                    "Type": "Fine-Tuned" if "-FT" in model_name else "Base",
                    "Architecture": model_name.split("-")[0],
                    "Dimension": dim,
                    "Storage_MB": storage_mb,
                    "NDCG@10": ndcg_score
                })

                print(f"  Dim: {dim}d | Size: {storage_mb:.2f}MB | NDCG: {ndcg_score:.4f}")

            # Cleanup to save RAM
            del model
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    df_eff = pd.DataFrame(results_data)
    return df_eff

def plot_efficiency(df_eff, output_path="efficiency_tradeoff.png"):
    """
    Plots the efficiency frontier.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(
        data=df_eff,
        x="Storage_MB",
        y="NDCG@10",
        hue="Model",
        style="Type",
        markers=True,
        dashes=False,
        linewidth=2.5,
        palette="tab10",
        ax=ax
    )

    corpus_size = int(df_eff['Storage_MB'].iloc[0] * (1024**2) / (df_eff['Dimension'].iloc[0] * 4)) # Reverse calc approx

    ax.set_title("Efficiency Frontier: Storage Cost vs. Retrieval Performance", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Vector Index Size (MB) for ~{corpus_size} Documents", fontsize=12)
    ax.set_ylabel("NDCG@10 Score", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
