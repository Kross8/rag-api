from fastembed import TextEmbedding

_model = None

def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model

def get_embedding(text: str):
    model = get_model()
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()