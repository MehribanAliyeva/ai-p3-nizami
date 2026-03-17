"""
GloVe (Global Vectors) model training and analysis utilities.
Pure NumPy implementation without glove-python library.
"""

import numpy as np
from collections import Counter, defaultdict


class Corpus:
    """
    Minimal replacement for glove.Corpus.
    Builds vocabulary and word-word co-occurrence matrix.
    """

    def __init__(self):
        self.dictionary = {}
        self.inverse_dictionary = {}
        self.matrix = None
        self.word_counts = Counter()

    def fit(self, tokenized_docs, window=10, min_count=1):
        """
        Build co-occurrence matrix from tokenized documents.

        Args:
            tokenized_docs: list of tokenized documents
            window: context window size
            min_count: minimum word frequency to keep in vocabulary
        """
        # Count words
        for doc in tokenized_docs:
            self.word_counts.update([str(w).lower() for w in doc if str(w).strip()])

        vocab = sorted([w for w, c in self.word_counts.items() if c >= min_count])
        self.dictionary = {w: i for i, w in enumerate(vocab)}
        self.inverse_dictionary = {i: w for w, i in self.dictionary.items()}

        vocab_size = len(self.dictionary)
        cooc = np.zeros((vocab_size, vocab_size), dtype=np.float64)

        for doc in tokenized_docs:
            tokens = [str(w).lower() for w in doc if str(w).strip()]
            ids = [self.dictionary[w] for w in tokens if w in self.dictionary]

            for center_pos, center_id in enumerate(ids):
                start = max(0, center_pos - window)
                end = min(len(ids), center_pos + window + 1)

                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue

                    context_id = ids[context_pos]
                    distance = abs(center_pos - context_pos)

                    # inverse-distance weighting
                    cooc[center_id, context_id] += 1.0 / distance

        self.matrix = cooc
        return self


class Glove:
    """
    Minimal replacement for glove.Glove using NumPy.
    """

    def __init__(
        self,
        no_components=100,
        learning_rate=0.05,
        alpha=0.75,
        x_max=100.0,
        random_state=42,
    ):
        self.no_components = no_components
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.x_max = x_max
        self.random_state = random_state

        self.word_vectors = None
        self.context_vectors = None
        self.word_biases = None
        self.context_biases = None
        self.dictionary = {}
        self.inverse_dictionary = {}

    def fit(self, matrix, epochs=30, no_threads=1, verbose=False):
        """
        Train GloVe embeddings on co-occurrence matrix.

        Args:
            matrix: co-occurrence matrix
            epochs: number of passes
            no_threads: kept for compatibility, not used
            verbose: print training progress
        """
        rng = np.random.default_rng(self.random_state)
        vocab_size = matrix.shape[0]
        dim = self.no_components

        # Initialize parameters
        self.word_vectors = rng.normal(scale=0.1, size=(vocab_size, dim))
        self.context_vectors = rng.normal(scale=0.1, size=(vocab_size, dim))
        self.word_biases = np.zeros(vocab_size, dtype=np.float64)
        self.context_biases = np.zeros(vocab_size, dtype=np.float64)

        # AdaGrad accumulators
        grad_sq_w = np.ones((vocab_size, dim), dtype=np.float64)
        grad_sq_c = np.ones((vocab_size, dim), dtype=np.float64)
        grad_sq_bw = np.ones(vocab_size, dtype=np.float64)
        grad_sq_bc = np.ones(vocab_size, dtype=np.float64)

        # Only train on nonzero co-occurrences
        nz_i, nz_j = np.nonzero(matrix)
        coocs = [(i, j, matrix[i, j]) for i, j in zip(nz_i, nz_j)]

        for epoch in range(epochs):
            rng.shuffle(coocs)
            total_loss = 0.0

            for i, j, x_ij in coocs:
                if x_ij <= 0:
                    continue

                if x_ij < self.x_max:
                    weight = (x_ij / self.x_max) ** self.alpha
                else:
                    weight = 1.0

                wi = self.word_vectors[i]
                wj = self.context_vectors[j]
                bi = self.word_biases[i]
                bj = self.context_biases[j]

                inner = np.dot(wi, wj) + bi + bj - np.log(x_ij)
                loss_contrib = weight * (inner ** 2)
                total_loss += 0.5 * loss_contrib

                grad_common = weight * inner

                grad_wi = grad_common * wj
                grad_wj = grad_common * wi
                grad_bi = grad_common
                grad_bj = grad_common

                # AdaGrad updates
                self.word_vectors[i] -= (
                    self.learning_rate * grad_wi / np.sqrt(grad_sq_w[i])
                )
                self.context_vectors[j] -= (
                    self.learning_rate * grad_wj / np.sqrt(grad_sq_c[j])
                )
                self.word_biases[i] -= (
                    self.learning_rate * grad_bi / np.sqrt(grad_sq_bw[i])
                )
                self.context_biases[j] -= (
                    self.learning_rate * grad_bj / np.sqrt(grad_sq_bc[j])
                )

                grad_sq_w[i] += grad_wi ** 2
                grad_sq_c[j] += grad_wj ** 2
                grad_sq_bw[i] += grad_bi ** 2
                grad_sq_bc[j] += grad_bj ** 2

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, loss={total_loss:.4f}")

        # Final word vectors: standard practice is word + context
        self.word_vectors = self.word_vectors + self.context_vectors
        return self

    def add_dictionary(self, dictionary):
        self.dictionary = dictionary
        self.inverse_dictionary = {i: w for w, i in dictionary.items()}

    def most_similar(self, word_id, number=10):
        """
        Return list of (word_id, similarity) like glove-python.
        """
        if self.word_vectors is None:
            return []

        target = self.word_vectors[word_id]
        target_norm = np.linalg.norm(target) + 1e-10

        norms = np.linalg.norm(self.word_vectors, axis=1) + 1e-10
        sims = np.dot(self.word_vectors, target) / (norms * target_norm)

        # exclude self
        sims[word_id] = -np.inf
        top_ids = np.argsort(-sims)[:number]

        return [(int(idx), float(sims[idx])) for idx in top_ids]


def build_cooccurrence_corpus(tokenized_docs, window_size=10):
    """
    Build GloVe corpus with co-occurrence statistics.

    Args:
        tokenized_docs: List of tokenized documents
        window_size: Context window size for co-occurrences

    Returns:
        Corpus object for GloVe training
    """
    corpus = Corpus()
    corpus.fit(tokenized_docs, window=window_size)
    return corpus


def train_glove(
    corpus,
    vector_size=100,
    learning_rate=0.05,
    epochs=30,
    no_components=100,
    alpha=0.75
):
    """
    Train GloVe model on the corpus.

    Parameters:
    - vector_size (no_components): Size of word embeddings
    - learning_rate: Learning rate for training
    - epochs: Number of training iterations
    - alpha: Weighting function exponent

    Returns:
        Trained Glove model
    """
    actual_dim = no_components if no_components is not None else vector_size

    glove = Glove(
        no_components=actual_dim,
        learning_rate=learning_rate,
        alpha=alpha
    )
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    return glove


def get_similar_words_glove(model, word, topn=10):
    """
    Find most similar words using GloVe model.
    """
    try:
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
    except (KeyError, AttributeError):
        return []


def vector_arithmetic_glove(model, positive=None, negative=None, topn=10):
    """
    Perform vector arithmetic: positive - negative.
    """
    positive = positive or []
    negative = negative or []

    try:
        result_vector = np.zeros(model.word_vectors.shape[1], dtype=np.float64)

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
        for word, word_id in model.dictionary.items():
            if word not in positive and word not in negative:
                word_vec = model.word_vectors[word_id]
                sim = np.dot(result_vector, word_vec) / (
                    result_norm * (np.linalg.norm(word_vec) + 1e-10)
                )
                similarities.append((word, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    except (KeyError, AttributeError):
        return []


def cosine_similarity_glove(model, word1, word2):
    """
    Calculate cosine similarity between two words.
    """
    try:
        word1_id = model.dictionary.get(word1.lower())
        word2_id = model.dictionary.get(word2.lower())

        if word1_id is None or word2_id is None:
            return None

        v1 = model.word_vectors[word1_id]
        v2 = model.word_vectors[word2_id]

        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        return float(similarity)

    except (KeyError, AttributeError):
        return None


def get_vocab_size(model):
    """Get vocabulary size of trained model."""
    return len(model.dictionary)


def get_vector_size(model):
    """Get embedding dimension size."""
    return model.word_vectors.shape[1]