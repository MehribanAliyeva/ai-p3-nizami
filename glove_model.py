"""
GloVe (Global Vectors) model training and analysis utilities.
Using glove-python library for training on Azerbaijani poetry corpus.
"""

import numpy as np
from collections import Counter
from glove import Corpus, Glove


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
    - alpha: Weighting function exponent (default 0.75 from paper)
    
    Returns:
        Trained Glove model
    """
    glove = Glove(no_components=no_components, learning_rate=learning_rate, alpha=alpha)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    return glove


def get_similar_words_glove(model, word, topn=10):
    """
    Find most similar words using GloVe model.
    
    Args:
        model: Trained GloVe model
        word: Query word
        topn: Number of similar words to return
    
    Returns:
        List of (word, similarity_score) tuples
    """
    try:
        word_id = model.dictionary.get(word.lower())
        if word_id is None:
            return []
        
        # Get most similar using cosine similarity
        similar = model.most_similar(word_id, number=topn)
        
        # Convert word IDs back to words
        results = []
        for sim_id, score in similar:
            sim_word = [w for w, idx in model.dictionary.items() if idx == sim_id]
            if sim_word:
                results.append((sim_word[0], float(score)))
        
        return results
    except (KeyError, AttributeError):
        return []


def vector_arithmetic_glove(model, positive=None, negative=None, topn=10):
    """
    Perform vector arithmetic: positive - negative.
    
    Example: sevgi - pis = ?
    
    Args:
        model: Trained GloVe model
        positive: List of positive words to add
        negative: List of negative words to subtract
        topn: Number of results to return
    
    Returns:
        List of (word, score) tuples
    """
    positive = positive or []
    negative = negative or []
    
    try:
        # Get word vectors
        result_vector = np.zeros(model.word_vectors.shape[1])
        
        # Add positive words
        for word in positive:
            word_id = model.dictionary.get(word.lower())
            if word_id is not None:
                result_vector += model.word_vectors[word_id]
        
        # Subtract negative words
        for word in negative:
            word_id = model.dictionary.get(word.lower())
            if word_id is not None:
                result_vector -= model.word_vectors[word_id]
        
        # Find most similar to result vector
        similarities = []
        for word, word_id in model.dictionary.items():
            if word not in positive and word not in negative:
                word_vec = model.word_vectors[word_id]
                # Cosine similarity
                sim = np.dot(result_vector, word_vec) / (
                    np.linalg.norm(result_vector) * np.linalg.norm(word_vec) + 1e-10
                )
                similarities.append((word, float(sim)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]
        
    except (KeyError, AttributeError):
        return []


def cosine_similarity_glove(model, word1, word2):
    """
    Calculate cosine similarity between two words.
    
    Args:
        model: Trained GloVe model
        word1: First word
        word2: Second word
    
    Returns:
        Similarity score (0-1) or None if words not in vocabulary
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
