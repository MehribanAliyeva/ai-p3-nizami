import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


def build_term_document_matrix(documents, max_features=50):
    """
    Term-document matrix:
    rows = documents
    columns = words
    """
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(documents)
    terms = vectorizer.get_feature_names_out()
    df = pd.DataFrame(X.toarray(), columns=terms)
    df.index = [f"doc_{i+1}" for i in range(len(documents))]
    return df, vectorizer


def build_word_word_matrix(tokenized_docs, vocab_size=50, window_size=2):
    from collections import Counter
    all_tokens = [t for doc in tokenized_docs for t in doc]
    vocab = [w for w, _ in Counter(all_tokens).most_common(vocab_size)]
    vocab_set = set(vocab)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    matrix = np.zeros((len(vocab), len(vocab)), dtype=int)

    for doc in tokenized_docs:
        filtered_doc = [w for w in doc if w in vocab_set]
        for i, center_word in enumerate(filtered_doc):
            center_idx = word_to_idx[center_word]
            left = max(0, i - window_size)
            right = min(len(filtered_doc), i + window_size + 1)

            for j in range(left, right):
                if i == j:
                    continue
                context_word = filtered_doc[j]
                context_idx = word_to_idx[context_word]
                matrix[center_idx, context_idx] += 1

    df = pd.DataFrame(matrix, index=vocab, columns=vocab)
    return df


def get_top_word_frequencies(counter, n=20):
    df = pd.DataFrame(counter.most_common(n), columns=["word", "frequency"])
    return df