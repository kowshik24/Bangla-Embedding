import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim

def create_matryoshka_evaluator(queries, corpus, relevant_docs, matryoshka_dimensions):
    """
    Creates a SequentialEvaluator for multiple Matryoshka dimensions.
    """
    matryoshka_evaluators = []
    for dim in matryoshka_dimensions:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{dim}",
            truncate_dim=dim,  # This is the key MRL parameter
            score_functions={"cosine": cos_sim},
        )
        matryoshka_evaluators.append(ir_evaluator)

    # The SequentialEvaluator runs all evaluators in the list
    evaluator = SequentialEvaluator(matryoshka_evaluators)
    return evaluator

def get_per_query_ndcg(model_id, queries, corpus, relevant_docs, k=10, batch_size=32, device="cuda"):
    """
    Calculates NDCG@k for each query individually.
    """
    print(f"\nLoading Model: {model_id}...")
    model = SentenceTransformer(model_id, device=device)

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]

    print("Encoding Queries and Corpus...")
    q_embs = model.encode(query_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    c_embs = model.encode(corpus_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)

    print("Computing Similarities...")
    cos_scores = util.cos_sim(q_embs, c_embs).cpu() # Move to CPU

    scores_list = []

    for i, qid in enumerate(query_ids):
        true_doc_ids = relevant_docs.get(qid, set())

        if not true_doc_ids:
            scores_list.append(0.0)
            continue

        # Get Top K
        top_k_vals, top_k_ind = torch.topk(cos_scores[i], k=k)
        top_k_ind = top_k_ind.tolist()

        # DCG
        dcg = 0.0
        for rank, idx in enumerate(top_k_ind):
            retrieved_id = corpus_ids[idx]
            if retrieved_id in true_doc_ids:
                dcg += 1.0 / np.log2(rank + 2)

        # IDCG
        idcg = 0.0
        for rank in range(min(len(true_doc_ids), k)):
            idcg += 1.0 / np.log2(rank + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        scores_list.append(ndcg)

    # Clear GPU memory
    del model
    del q_embs
    del c_embs
    torch.cuda.empty_cache()

    return scores_list
