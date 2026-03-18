"""
GloVe-style model training and analysis utilities using Mittens.
This version avoids the old glove-python package and replaces the
dense co-occurrence matrix with a sparse SciPy matrix.

Install:
    pip install mittens scipy numpy
"""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from mittens import Mittens


@dataclass
class Corpus:
    """
    Replacement for glove.Corpus using a sparse co-occurrence matrix.
    """
    dictionary: Dict[str, int]
    inverse_dictionary: Dict[int, str]
    matrix: csr_matrix
    word_counts: Counter


class GloveModel:
    """
    Lightweight wrapper around trained word vectors.
    Keeps an interface similar to your previous code.
    """

    def __init__(self, word_vectors: np.ndarray, dictionary: Dict[str, int]):
        self.word_vectors = np.asarray(word_vectors, dtype=np.float32)
        self.dictionary = dictionary
        self.inverse_dictionary = {i: w for w, i in dictionary.items()}

    def most_similar(self, word_id: int, number: int = 10) -> List[Tuple[int, float]]:
        if self.word_vectors is None or len(self.word_vectors) == 0:
            return []

        target = self.word_vectors[word_id]
        target_norm = np.linalg.norm(target) + 1e-10

        norms = np.linalg.norm(self.word_vectors, axis=1) + 1e-10
        sims = np.dot(self.word_vectors, target) / (norms * target_norm)

        sims[word_id] = -np.inf
        top_ids = np.argsort(-sims)[:number]
        return [(int(idx), float(sims[idx])) for idx in top_ids]

    def save(self, path: str) -> None:
        payload = {
            "word_vectors": self.word_vectors,
            "dictionary": self.dictionary,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "GloveModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(
            word_vectors=payload["word_vectors"],
            dictionary=payload["dictionary"],
        )


def _normalize_doc(doc: Sequence[object]) -> List[str]:
    return [str(w).lower() for w in doc if str(w).strip()]


def build_cooccurrence_corpus(
    tokenized_docs: Sequence[Sequence[object]],
    window_size: int = 10,
    min_count: int = 1,
    symmetric: bool = True,
) -> Corpus:
    """
    Build a sparse co-occurrence matrix.

    Args:
        tokenized_docs: list of tokenized documents
        window_size: context window size
        min_count: minimum token frequency to keep
        symmetric: if True, add both (i, j) and (j, i)

    Returns:
        Corpus with sparse CSR matrix
    """
    word_counts: Counter = Counter()

    normalized_docs: List[List[str]] = []
    for doc in tokenized_docs:
        tokens = _normalize_doc(doc)
        normalized_docs.append(tokens)
        word_counts.update(tokens)

    vocab = sorted([w for w, c in word_counts.items() if c >= min_count])
    dictionary = {w: i for i, w in enumerate(vocab)}
    inverse_dictionary = {i: w for w, i in dictionary.items()}

    cooc_dict = defaultdict(float)

    for tokens in normalized_docs:
        ids = [dictionary[w] for w in tokens if w in dictionary]

        for center_pos, center_id in enumerate(ids):
            start = max(0, center_pos - window_size)
            end = min(len(ids), center_pos + window_size + 1)

            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue

                context_id = ids[context_pos]
                distance = abs(center_pos - context_pos)
                weight = 1.0 / distance

                cooc_dict[(center_id, context_id)] += weight
                if symmetric:
                    cooc_dict[(context_id, center_id)] += weight

    if not cooc_dict:
        matrix = csr_matrix((len(vocab), len(vocab)), dtype=np.float32)
    else:
        rows, cols, data = zip(*[(i, j, v) for (i, j), v in cooc_dict.items()])
        matrix = coo_matrix(
            (np.array(data, dtype=np.float32), (rows, cols)),
            shape=(len(vocab), len(vocab)),
            dtype=np.float32,
        ).tocsr()

    return Corpus(
        dictionary=dictionary,
        inverse_dictionary=inverse_dictionary,
        matrix=matrix,
        word_counts=word_counts,
    )


def train_glove(
    corpus,
    vector_size=100,
    learning_rate=0.05,
    epochs=30,
    no_components=None,
    alpha=0.75,
    x_max=100.0,
    random_state=42,
):
    """
    Train GloVe model on the corpus.

    Parameters:
    - vector_size: embedding size
    - no_components: optional alias for vector_size, kept for compatibility
    - learning_rate: learning rate
    - epochs: number of iterations
    - alpha: weighting exponent
    - x_max: GloVe x_max parameter
    """
    actual_dim = no_components if no_components is not None else vector_size

    glove = Glove(
        no_components=actual_dim,
        learning_rate=learning_rate,
        alpha=alpha,
        x_max=x_max,
        random_state=random_state,
    )
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    return glove

def get_similar_words_glove(
    model: GloveModel,
    word: str,
    topn: int = 10,
) -> List[Tuple[str, float]]:
    word_id = model.dictionary.get(word.lower())
    if word_id is None:
        return []

    similar = model.most_similar(word_id, number=topn)
    results = []

    for sim_id, score in similar:
        sim_word = model.inverse_dictionary.get(sim_id)
        if sim_word is not None:
            results.append((sim_word, float(score)))

    return results


def vector_arithmetic_glove(
    model: GloveModel,
    positive: Optional[List[str]] = None,
    negative: Optional[List[str]] = None,
    topn: int = 10,
) -> List[Tuple[str, float]]:
    positive = positive or []
    negative = negative or []

    result_vector = np.zeros(model.word_vectors.shape[1], dtype=np.float32)

    for word in positive:
        word_id = model.dictionary.get(word.lower())
        if word_id is not None:
            result_vector += model.word_vectors[word_id]

    for word in negative:
        word_id = model.dictionary.get(word.lower())
        if word_id is not None:
            result_vector -= model.word_vectors[word_id]

    result_norm = np.linalg.norm(result_vector)
    if result_norm < 1e-10:
        return []

    similarities = []
    excluded = {w.lower() for w in positive + negative}

    for word, word_id in model.dictionary.items():
        if word in excluded:
            continue

        word_vec = model.word_vectors[word_id]
        sim = np.dot(result_vector, word_vec) / (
            result_norm * (np.linalg.norm(word_vec) + 1e-10)
        )
        similarities.append((word, float(sim)))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:topn]


def cosine_similarity_glove(
    model: GloveModel,
    word1: str,
    word2: str,
) -> Optional[float]:
    word1_id = model.dictionary.get(word1.lower())
    word2_id = model.dictionary.get(word2.lower())

    if word1_id is None or word2_id is None:
        return None

    v1 = model.word_vectors[word1_id]
    v2 = model.word_vectors[word2_id]

    similarity = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    )
    return float(similarity)


def get_vocab_size(model: GloveModel) -> int:
    return len(model.dictionary)


def get_vector_size(model: GloveModel) -> int:
    return int(model.word_vectors.shape[1])


def save_corpus(corpus: Corpus, path: str) -> None:
    payload = {
        "dictionary": corpus.dictionary,
        "inverse_dictionary": corpus.inverse_dictionary,
        "matrix": corpus.matrix,
        "word_counts": corpus.word_counts,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_corpus(path: str) -> Corpus:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    return Corpus(
        dictionary=payload["dictionary"],
        inverse_dictionary=payload["inverse_dictionary"],
        matrix=payload["matrix"],
        word_counts=payload["word_counts"],
    )