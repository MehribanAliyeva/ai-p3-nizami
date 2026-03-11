import re
import string
from collections import Counter
from typing import List, Tuple


AZ_PUNCT_EXTRA = "“”‘’«»…–—"
PUNCT_TO_REMOVE = string.punctuation + AZ_PUNCT_EXTRA


def read_corpus(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_into_documents(text: str) -> List[str]:
    docs = [doc.strip() for doc in re.split(r"\n\s*\n+", text) if doc.strip()]
    return docs


def tokenize(text: str) -> List[str]:
    text = text.lower()

    trans_table = str.maketrans("", "", PUNCT_TO_REMOVE)
    text = text.translate(trans_table)

    tokens = text.split()
    return tokens

def docs_to_tokens(docs: List[str]) -> List[List[str]]:
    return [tokenize(doc) for doc in docs]


def flatten(list_of_lists: List[List[str]]) -> List[str]:
    return [item for sublist in list_of_lists for item in sublist]


def get_frequency_info(tokenized_docs: List[List[str]]) -> Tuple[Counter, int, int]:
    all_tokens = flatten(tokenized_docs)
    counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    distinct_words = len(counter)
    return counter, total_tokens, distinct_words


def get_frequent_and_rare_words(counter: Counter, frequent_threshold: int = 5, rare_threshold: int = 1):
    frequent_words = {w: c for w, c in counter.items() if c >= frequent_threshold}
    rare_words = {w: c for w, c in counter.items() if c <= rare_threshold}
    return frequent_words, rare_words


def corpus_summary(tokenized_docs: List[List[str]], frequent_threshold: int = 5, rare_threshold: int = 1):
    counter, total_tokens, distinct_words = get_frequency_info(tokenized_docs)
    frequent_words, rare_words = get_frequent_and_rare_words(
        counter,
        frequent_threshold=frequent_threshold,
        rare_threshold=rare_threshold
    )

    return {
        "num_documents": len(tokenized_docs),
        "total_tokens": total_tokens,
        "distinct_words": distinct_words,
        "frequent_words_count": len(frequent_words),
        "rare_words_count": len(rare_words),
        "top_20_words": counter.most_common(20),
        "counter": counter,
        "frequent_words": frequent_words,
        "rare_words": rare_words,
    }