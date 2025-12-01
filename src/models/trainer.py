import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData, SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

def train_model(model_id, train_dataset, evaluator, output_dir, matryoshka_dimensions, device="cuda", hf_token=None, hub_model_id=None):
    """
    Fine-tunes a SentenceTransformer model using Matryoshka Representation Learning.
    """
    # Reload the base model for training, enabling Scaled Dot Product Attention (SDPA) for efficiency
    model = SentenceTransformer(
        model_id,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="bn",
            license="apache-2.0",
            model_name="Bangla Sentence Transformer FT Matryoshka",
        ),
        device=device,
        token=hf_token
    )

    # Define the base loss function
    base_loss = MultipleNegativesRankingLoss(model)

    # Wrap it with MatryoshkaLoss
    train_loss = MatryoshkaLoss(
        model, base_loss, matryoshka_dims=matryoshka_dimensions
    )

    training_args_kwargs = {
        "output_dir": output_dir, 
        "num_train_epochs": 4,                                        
        "per_device_train_batch_size": 32,                            
        "gradient_accumulation_steps": 4,                             
        "per_device_eval_batch_size": 32,                             
        "warmup_ratio": 0.1,                                          
        "learning_rate": 2e-5,                                        
        "lr_scheduler_type": "cosine",                              
        "optim": "adamw_torch_fused",                                 
        "batch_sampler": BatchSamplers.NO_DUPLICATES,                 
        "eval_strategy": "epoch",                                     
        "save_strategy": "epoch",                                     
        "logging_steps": 50,                                          
        "save_total_limit": 2,                                        
        "load_best_model_at_end": True,                               
        "metric_for_best_model": "eval_dim_128_cosine_ndcg@10",       
        "report_to": ["tensorboard"]
    }

    if hf_token and hub_model_id:
        training_args_kwargs["push_to_hub"] = True
        training_args_kwargs["hub_token"] = hf_token
        training_args_kwargs["hub_model_id"] = hub_model_id

    args = SentenceTransformerTrainingArguments(**training_args_kwargs)

    # Initialize the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset.select_columns(["positive", "anchor"]),
        loss=train_loss,
        evaluator=evaluator,
    )

    # Start the training process
    trainer.train()

    # Save the final, best-performing model
    trainer.save_model()
    
    if hf_token and hub_model_id:
        trainer.push_to_hub()
    
    return trainer
