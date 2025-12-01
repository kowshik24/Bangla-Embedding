from datasets import Dataset, concatenate_datasets
from .loader import load_and_parse_banglarqa

def prepare_datasets(train_path, val_path, test_path):
    """
    Loads and processes the BanglaRQA datasets.
    """
    # Load data from local files
    train_data = load_and_parse_banglarqa(train_path)
    validation_data = load_and_parse_banglarqa(val_path)
    test_data = load_and_parse_banglarqa(test_path)

    # Combine train and validation sets for a larger training pool
    full_train_data = train_data + validation_data

    # Convert to datasets.Dataset objects
    train_ds = Dataset.from_list(full_train_data)
    test_ds = Dataset.from_list(test_data)

    # Process Training Set
    train_dataset = train_ds.filter(lambda example: example["is_answerable"] == "1")
    train_dataset = train_dataset.rename_column("question_text", "anchor")
    train_dataset = train_dataset.rename_column("context", "positive")
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ['anchor', 'positive', 'passage_id']])
    train_dataset = train_dataset.add_column("id", range(len(train_dataset)))

    # Process Test Set
    test_dataset = test_ds.filter(lambda example: example["is_answerable"] == "1")
    test_dataset = test_dataset.rename_column("question_text", "anchor")
    test_dataset = test_dataset.rename_column("context", "positive")
    test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in ['anchor', 'positive', 'passage_id']])
    test_dataset = test_dataset.add_column("id", range(len(train_dataset), len(train_dataset) + len(test_dataset)))

    return train_dataset, test_dataset

def prepare_ir_eval_data(train_dataset, test_dataset):
    """
    Prepares data for Information Retrieval Evaluator.
    """
    # Create the corpus by combining all unique passages from both train and test sets
    corpus_data = concatenate_datasets([train_dataset, test_dataset])
    corpus = dict(zip(corpus_data["id"], corpus_data["positive"])) # Using the unique row ID as doc_id

    # Create the queries from the test set
    queries = dict(zip(test_dataset["id"], test_dataset["anchor"]))

    # Create the mapping from queries to relevant documents
    relevant_docs = {}
    for row in test_dataset:
        q_id = row['id']
        passage_id = row['passage_id']
        if q_id not in relevant_docs:
            relevant_docs[q_id] = set()
        # Find all corpus entries (docs) that share the same passage_id
        # This is our definition of relevance
        matching_corpus_ids = [cid for cid, pid in zip(corpus_data['id'], corpus_data['passage_id']) if pid == passage_id]
        relevant_docs[q_id].update(matching_corpus_ids)
    
    return corpus, queries, relevant_docs
