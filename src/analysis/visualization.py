import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def plot_model_comparison(results_df, output_path="model_comparison_results.png"):
    """
    Plots bar charts comparing models across dimensions for multiple metrics.
    """
    sns.set_theme(style="whitegrid")
    
    custom_palette = {
        "Shihab17": "#0072B2",  # Vivid Blue
        "Distiluse": "#D55E00", # Vivid Orange
        "MPNet": "#009E73"      # Teal/Greenish-Blue
    }

    metrics_to_plot = ["ndcg@10", "mrr@10", "map@100", "accuracy@1"]
    matryoshka_dimensions = [64, 128, 256, 512, 768]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fine-Tuned Model Performance Comparison Across Dimensions', fontsize=18, fontweight='bold')

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        metric_df = results_df[results_df['Metric'] == metric]
        
        sns.barplot(
            data=metric_df, 
            x='Dimension', 
            y='Score', 
            hue='Model', 
            palette=custom_palette,
            ax=ax, 
            order=matryoshka_dimensions,
            hue_order=["MPNet", "Shihab17", "Distiluse"]
        )
        
        ax.set_title(f'Comparison for {metric}', fontsize=14)
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Score')
        ax.legend(title='Model', loc='lower right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

def plot_radar_chart(mpnet_data, shihab_data, distiluse_data, dims_to_compare=[64, 512], output_path="paper_plot_radar_chart.png"):
    """
    Plots radar charts for low vs high dimensions.
    """
    metrics = ['ndcg@10', 'mrr@10', 'map@100', 'accuracy@1']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), subplot_kw=dict(polar=True))
    fig.suptitle('Model Robustness: Low (64d) vs High (512d) Dimensions', fontsize=18, fontweight='bold')

    for idx, dim in enumerate(dims_to_compare):
        ax = axes[idx]
        
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1] 
        
        def plot_radar(model_name, data_source, color):
            if dim not in data_source: return
            values = [data_source[dim][m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=color)
            ax.fill(angles, values, color=color, alpha=0.15)

        plot_radar("MPNet", mpnet_data, "#2ca02c")
        plot_radar("Shihab17", shihab_data, "#1f77b4")
        plot_radar("Distiluse", distiluse_data, "#d62728")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
        ax.set_title(f'Dimension: {dim}d', size=15, pad=20)
        ax.set_ylim(0, 0.7) 

    axes[0].legend(loc='upper right', bbox_to_anchor=(0.1, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved as {output_path}")

def plot_retention(mpnet_data, shihab_data, distiluse_data, output_path="paper_plot_retention.png"):
    """
    Plots performance retention relative to MPNet.
    """
    retention_data = []
    target_dims = [64, 128, 256, 512]

    for dim in target_dims:
        base_score = mpnet_data[dim]['ndcg@10']
        
        retention_data.append({
            "Dimension": dim, "Model": "Shihab17", 
            "Retention": (shihab_data[dim]['ndcg@10'] / base_score) * 100
        })
        retention_data.append({
            "Dimension": dim, "Model": "Distiluse", 
            "Retention": (distiluse_data[dim]['ndcg@10'] / base_score) * 100
        })
        retention_data.append({
            "Dimension": dim, "Model": "MPNet", 
            "Retention": 100.0
        })

    df_retention = pd.DataFrame(retention_data)

    sns.set_theme(style="whitegrid")
    custom_palette = {"Shihab17": "#1f77b4", "Distiluse": "#d62728", "MPNet": "#2ca02c"}

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        x="Dimension", 
        y="Retention", 
        hue="Model", 
        data=df_retention, 
        palette=custom_palette, 
        hue_order=["MPNet", "Shihab17", "Distiluse"]
    )

    ax.set_title("Performance Retention Relative to MPNet (NDCG@10)", fontsize=18, fontweight='bold')
    ax.set_ylabel("Performance Retention (%)", fontsize=14)
    ax.set_xlabel("Embedding Dimension", fontsize=14)
    ax.set_ylim(60, 105)

    ax.axhline(95, color='gray', linestyle='--', alpha=0.5)
    ax.text(3.6, 95.5, '95% Threshold', color='gray', fontsize=10, fontweight='bold')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=11)

    plt.legend(title="Model", loc="lower left", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved as {output_path}")
