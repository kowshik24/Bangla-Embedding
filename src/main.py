import argparse
import os
import torch
from dotenv import load_dotenv
from data.processor import prepare_datasets, prepare_ir_eval_data
from models.trainer import train_model
from models.utils import get_matryoshka_dimensions
from evaluation.evaluator import create_matryoshka_evaluator
from evaluation.mteb_eval import run_mteb_evaluation
from analysis.stats import run_significance_test
from analysis.efficiency import analyze_efficiency, plot_efficiency

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Bangla Embedding Model Fine-Tuning and Evaluation")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "mteb", "stats", "efficiency"], help="Mode to run")
    parser.add_argument("--model_id", type=str, default="shihab17/bangla-sentence-transformer", help="Base model ID")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--train_path", type=str, default="BanglaRQA/Train.json", help="Path to training data")
    parser.add_argument("--val_path", type=str, default="BanglaRQA/Validation.json", help="Path to validation data")
    parser.add_argument("--test_path", type=str, default="BanglaRQA/Test.json", help="Path to test data")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face token")
    parser.add_argument("--hub_model_id", type=str, help="Hugging Face Hub Model ID (e.g. username/repo_name) for pushing")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matryoshka_dims = get_matryoshka_dimensions()

    if args.mode == "train":
        print("Preparing datasets...")
        train_dataset, test_dataset = prepare_datasets(args.train_path, args.val_path, args.test_path)
        
        print("Preparing evaluator...")
        corpus, queries, relevant_docs = prepare_ir_eval_data(train_dataset, test_dataset)
        evaluator = create_matryoshka_evaluator(queries, corpus, relevant_docs, matryoshka_dims)
        
        print("Starting training...")
        train_model(args.model_id, train_dataset, evaluator, args.output_dir, matryoshka_dims, device, args.hf_token, args.hub_model_id)

    elif args.mode == "mteb":
        print("Running MTEB evaluation...")
        run_mteb_evaluation(args.model_id, args.output_dir, matryoshka_dims, args.hf_token)

    elif args.mode == "stats":
        print("Running statistical significance testing...")
        # Note: This requires the test dataset to be available
        _, test_dataset = prepare_datasets(args.train_path, args.val_path, args.test_path)
        corpus, queries, relevant_docs = prepare_ir_eval_data([], test_dataset) # Empty train dataset as we only need test for eval
        
        # Define models to test (you might want to make this configurable)
        models_to_test = {
            "MyModel": args.model_id,
            # Add other models here
        }
        df_stats = run_significance_test(models_to_test, queries, corpus, relevant_docs)
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_csv = os.path.join(args.output_dir, "significance_test_results.csv")
        df_stats.to_csv(output_csv, index=False)
        print(f"Significance test results saved to {output_csv}")

    elif args.mode == "efficiency":
        print("Running efficiency analysis...")
        _, test_dataset = prepare_datasets(args.train_path, args.val_path, args.test_path)
        corpus, queries, relevant_docs = prepare_ir_eval_data([], test_dataset)
        
        model_map = {
            "MyModel": args.model_id,
            # Add other models here
        }
        df_eff = analyze_efficiency(model_map, matryoshka_dims, queries, corpus, relevant_docs, device)
        
        os.makedirs(args.output_dir, exist_ok=True)
        df_eff.to_csv(os.path.join(args.output_dir, "efficiency_results.csv"), index=False)
        print(f"Efficiency results saved to {os.path.join(args.output_dir, 'efficiency_results.csv')}")
        
        plot_efficiency(df_eff, os.path.join(args.output_dir, "efficiency.png"))

if __name__ == "__main__":
    main()
