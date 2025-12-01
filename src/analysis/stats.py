import pandas as pd
import numpy as np
from scipy import stats
from ..evaluation.evaluator import get_per_query_ndcg

def run_significance_test(models_to_test, queries, corpus, relevant_docs):
    """
    Runs Wilcoxon Signed-Rank Test for statistical significance.
    Returns a DataFrame with the results.
    """
    all_scores = {}

    # Compute scores
    for name, hf_id in models_to_test.items():
        try:
            scores = get_per_query_ndcg(hf_id, queries, corpus, relevant_docs)
            all_scores[name] = scores
            print(f"--> Average NDCG@10 for {name}: {np.mean(scores):.4f}")
        except Exception as e:
            print(f"Error loading {name} ({hf_id}): {e}")

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST (Wilcoxon Signed-Rank)")
    print("="*80)
    print(f"{'Comparison (Model A vs Model B)':<40} | {'Statistic':<10} | {'P-Value':<12} | {'Result'}")
    print("-" * 80)

    # Define comparisons (assuming keys exist)
    model_names = list(models_to_test.keys())
    comparisons = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            comparisons.append((model_names[i], model_names[j]))

    results_data = []
    for model_a, model_b in comparisons:
        if model_a in all_scores and model_b in all_scores:
            stat, p_val = stats.wilcoxon(all_scores[model_a], all_scores[model_b])
            significance = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "Not Significant"
            print(f"{model_a} vs {model_b:<19} | {stat:.2e}   | {p_val:.4e}   | {significance}")
            
            results_data.append({
                "Model A": model_a,
                "Model B": model_b,
                "Statistic": stat,
                "P-Value": p_val,
                "Significance": significance
            })
        else:
            print(f"Skipping {model_a} vs {model_b} (Model data missing)")

    print("-" * 80)
    return pd.DataFrame(results_data)
