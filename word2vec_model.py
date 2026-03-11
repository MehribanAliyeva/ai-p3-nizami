from gensim.models import Word2Vec
import numpy as np


def train_word2vec(
    tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    epochs=20,
    workers=4
):
    """
    sg=1 -> Skip-gram
    sg=0 -> CBOW
    """
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=workers
    )
    return model


def get_similar_words(model, word, topn=10):
    if word not in model.wv:
        return []
    return model.wv.most_similar(word, topn=topn)


def vector_arithmetic(model, positive=None, negative=None, topn=10):
    positive = positive or []
    negative = negative or []
    try:
        return model.wv.most_similar(positive=positive, negative=negative, topn=topn)
    except KeyError:
        return []


def cosine_similarity_between_words(model, word1, word2):
    if word1 not in model.wv or word2 not in model.wv:
        return None
    v1 = model.wv[word1]
    v2 = model.wv[word2]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))