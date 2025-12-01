from sentence_transformers import SentenceTransformer

def load_model(model_id, device="cuda"):
    """
    Loads a SentenceTransformer model.
    """
    return SentenceTransformer(model_id, device=device)

def get_matryoshka_dimensions():
    """
    Returns the standard Matryoshka dimensions used in this project.
    """
    return [768, 512, 256, 128, 64]
