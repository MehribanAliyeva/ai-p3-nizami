"""
Feature Extraction Module for Task 5
Supports: CountVectorizer, TF-IDF, PMI, Word2Vec, GloVe
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict, Counter
import re
import string


def extract_count_vectorizer_features(texts, max_features=100):
    """
    Extract Count Vectorizer features.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features
    
    Returns:
        Feature matrix (n_samples, max_features)
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray(), vectorizer


def extract_tfidf_features(texts, max_features=100):
    """
    Extract TF-IDF features.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features
    
    Returns:
        Feature matrix (n_samples, max_features)
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X.toarray(), vectorizer


def extract_pmi_features(texts, tokenized_docs, max_features=100, window_size=5):
    """
    Extract PMI (Pointwise Mutual Information) features.
    
    PMI measures association between words and documents.
    
    Args:
        texts: List of text documents
        tokenized_docs: List of tokenized documents
        max_features: Number of top PMI words to use
        window_size: Context window for co-occurrence
    
    Returns:
        Feature matrix (n_samples, max_features)
    """
    # Get vocabulary
    all_tokens = [token for doc in tokenized_docs for token in doc]
    word_freq = Counter(all_tokens)
    vocab = [w for w, _ in word_freq.most_common(max_features)]
    vocab_set = set(vocab)
    
    # Build co-occurrence matrix
    cooccur = defaultdict(lambda: defaultdict(int))
    
    for doc in tokenized_docs:
        filtered = [w for w in doc if w in vocab_set]
        for i, word in enumerate(filtered):
            start = max(0, i - window_size)
            end = min(len(filtered), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    cooccur[word][filtered[j]] += 1
    
    # Calculate PMI scores
    total_pairs = sum(sum(cooccur[w].values()) for w in cooccur)
    pmi_scores = {}
    
    for word in vocab:
        word_count = sum(cooccur[word].values())
        if word_count == 0:
            pmi_scores[word] = 0
            continue
        
        pmi_sum = 0
        for context_word in cooccur[word]:
            context_count = sum(cooccur[w][context_word] for w in vocab)
            if context_count == 0:
                continue
            
            p_word = word_count / total_pairs
            p_context = context_count / total_pairs
            p_together = cooccur[word][context_word] / total_pairs
            
            if p_together > 0:
                pmi = np.log2(p_together / (p_word * p_context + 1e-10))
                pmi_sum += max(0, pmi)  # Positive PMI
        
        pmi_scores[word] = pmi_sum
    
    # Create feature vectors
    X = np.zeros((len(texts), len(vocab)))
    
    for i, doc in enumerate(tokenized_docs):
        for word in doc:
            if word in vocab:
                word_idx = vocab.index(word)
                X[i, word_idx] = pmi_scores.get(word, 0)
    
    return X, vocab



def extract_word2vec_features(docs, tokenized_docs, model, vector_size=None):
    if model is None:
        return np.zeros((len(tokenized_docs), 0), dtype=np.float32)

    actual_dim = model.wv.vector_size
    X = np.zeros((len(tokenized_docs), actual_dim), dtype=np.float32)

    for i, doc in enumerate(tokenized_docs):
        vectors = [model.wv[word] for word in doc if word in model.wv]
        if vectors:
            X[i] = np.mean(vectors, axis=0)

    return X

import numpy as np

def extract_glove_features(docs, tokenized_docs, model, vector_size=None):
    if model is None or model.word_vectors is None or model.word_vectors.size == 0:
        return np.zeros((len(tokenized_docs), 0), dtype=np.float32)

    actual_dim = model.word_vectors.shape[1]
    X = np.zeros((len(tokenized_docs), actual_dim), dtype=np.float32)

    for i, doc in enumerate(tokenized_docs):
        vectors = [
            model.word_vectors[model.dictionary[word]]
            for word in doc
            if word in model.dictionary
        ]
        if vectors:
            X[i] = np.mean(vectors, axis=0)

    return X


def prepare_features_for_rnn(X, max_len=50):
    """
    Prepare features for RNN input (pad sequences).
    
    Args:
        X: Feature matrix
        max_len: Maximum sequence length
    
    Returns:
        Padded feature matrix
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # If X is 2D, reshape for sequence input
    if len(X.shape) == 2:
        # Assume each feature is a timestep
        n_samples, n_features = X.shape
        if n_features > max_len:
            X = X[:, :max_len]
        elif n_features < max_len:
            # Pad with zeros
            padding = np.zeros((n_samples, max_len - n_features))
            X = np.concatenate([X, padding], axis=1)
    
    # Reshape to (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X
