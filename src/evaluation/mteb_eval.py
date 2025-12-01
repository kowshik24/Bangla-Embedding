import logging
import os
import torch
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from mteb import evaluate, get_tasks
from mteb.cache import ResultCache

def run_mteb_evaluation(model_name, output_dir, matryoshka_dims, hf_token=None, batch_size=32):
    """
    Runs MTEB evaluation for Bengali tasks.
    """
    # Configure logging to see progress
    logging.basicConfig(level=logging.INFO)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Running on CPU (will be very slow).")

    # Define the specific task names you are interested in running (Pure Bengali Focus)
    desired_task_names = [
        "BanglaSentimentAnalysis.v2",  # Likely pure Bengali sentiment
        "BengaliDocumentClassification.v2",
        "BengaliHateSpeechClassification.v2",
        "BanglaParaphrase",            # Paraphrase identification
        "Tatoeba",                     # Cross-lingual sentence retrieval (uses bn subset)
    ]

    # Use mteb.get_tasks(languages=["ben"]) to get the list of ALL available *task objects*
    available_bn_tasks_objects = get_tasks(languages=["ben"])

    # Filter the *objects* themselves to match the desired names
    filtered_tasks_to_run_objects = []
    for task_obj in available_bn_tasks_objects:
        if task_obj.metadata.name in desired_task_names:
            filtered_tasks_to_run_objects.append(task_obj)

    print(f"\nTasks selected for evaluation: {[task.metadata.name for task in filtered_tasks_to_run_objects]}")
    print(f"Total tasks selected: {len(filtered_tasks_to_run_objects)}")

    all_results = []

    # Loop through each dimension and run the evaluation
    for dim in matryoshka_dims:
        print(f"\n--- Running evaluation for dimension: {dim} ---")

        # Load the model with potentially a token
        model = SentenceTransformer(model_name, token=hf_token)

        # Define the specific output path for this dimension
        output_folder_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}_dim_{dim}")
        os.makedirs(output_folder_path, exist_ok=True)

        # Initialize the cache: this is where results are saved automatically as JSON files
        cache = ResultCache(output_folder_path)
        print(f"Results will be cached in: {output_folder_path}")

        # Use the evaluate function with the LIST OF TASK OBJECTS
        try:
            results = evaluate(
                model,
                tasks=filtered_tasks_to_run_objects, # Pass the list of objects here
                cache=cache,
                encode_kwargs={
                    'truncate_dim': dim,
                    'batch_size': batch_size
                },
            )
            print(f"Evaluation results for dim {dim} returned by evaluate(): {results}")
            
            # Aggregate results for CSV
            for task_res in results:
                # MTEB results structure can be complex, simplifying for CSV
                # Assuming task_res is an MTEBResult object or similar
                # We'll try to extract the main score
                task_name = task_res.task_name
                main_score = task_res.get_score() # This usually gets the main metric
                
                all_results.append({
                    "Model": model_name,
                    "Dimension": dim,
                    "Task": task_name,
                    "Main Score": main_score
                })

        except RuntimeError as e:
            print(f"\nCaught a runtime error for dimension {dim}. This might be an OOM error.")
            print(f"Error details: {e}")
            print(f"Try reducing the BATCH_SIZE (currently {batch_size}) and restart the runtime.")

    # Save aggregated results to CSV
    if all_results:
        df_results = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, "mteb_aggregated_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nAggregated MTEB results saved to {csv_path}")

    print("\n--- All evaluations complete! ---")
