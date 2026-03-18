from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Corpus:
    dictionary: Dict[str, int]
    inverse_dictionary: Dict[int, str]
    word_counts: Counter
    tokenized_docs: List[List[str]]


class GloveModel:
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
    min_count: int = 1,
) -> Corpus:
    word_counts: Counter = Counter()
    normalized_docs: List[List[str]] = []

    for doc in tokenized_docs:
        tokens = _normalize_doc(doc)
        normalized_docs.append(tokens)
        word_counts.update(tokens)

    vocab = sorted([w for w, c in word_counts.items() if c >= min_count])
    dictionary = {w: i for i, w in enumerate(vocab)}
    inverse_dictionary = {i: w for w, i in dictionary.items()}

    filtered_docs = [
        [w for w in doc if w in dictionary]
        for doc in normalized_docs
    ]

    return Corpus(
        dictionary=dictionary,
        inverse_dictionary=inverse_dictionary,
        word_counts=word_counts,
        tokenized_docs=filtered_docs,
    )


def _find_glove_binary(glove_bin_dir: str, binary_name: str) -> str:
    candidates = [
        os.path.join(glove_bin_dir, binary_name),
        os.path.join(glove_bin_dir, "build", binary_name),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        f"Could not find executable '{binary_name}' in: {candidates}"
    )


def _write_corpus_text(corpus: Corpus, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for doc in corpus.tokenized_docs:
            if doc:
                f.write(" ".join(doc) + "\n")


def _run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def _load_glove_vectors(vectors_txt_path: str) -> GloveModel:
    dictionary: Dict[str, int] = {}
    vectors: List[np.ndarray] = []

    with open(vectors_txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.rstrip("\n").split(" ")
            if len(parts) < 2:
                continue
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            dictionary[word] = idx
            vectors.append(vec)

    if not vectors:
        return GloveModel(
            word_vectors=np.zeros((0, 0), dtype=np.float32),
            dictionary={},
        )

    matrix = np.vstack(vectors).astype(np.float32, copy=False)
    return GloveModel(word_vectors=matrix, dictionary=dictionary)


def train_glove(
    corpus: Corpus,
    vector_size: int = 100,
    learning_rate: float = 0.05,
    epochs: int = 30,
    x_max: float = 100.0,
    window_size: int = 10,
    min_count: int = 1,
    symmetric: bool = True,
    num_threads: int = 4,
    memory_gb: float = 4.0,
    verbose: int = 2,
    glove_bin_dir: str = "GloVe",
    cleanup_temp: bool = True,
) -> GloveModel:
    """
    Train exact GloVe using Stanford's original C implementation.

    Requirements:
        git clone https://github.com/stanfordnlp/GloVe.git
        cd GloVe && make
    """
    vocab_size = len(corpus.dictionary)
    if vocab_size == 0:
        return GloveModel(
            word_vectors=np.zeros((0, vector_size), dtype=np.float32),
            dictionary={},
        )

    vocab_count_bin = _find_glove_binary(glove_bin_dir, "vocab_count")
    cooccur_bin = _find_glove_binary(glove_bin_dir, "cooccur")
    shuffle_bin = _find_glove_binary(glove_bin_dir, "shuffle")
    glove_bin = _find_glove_binary(glove_bin_dir, "glove")

    tmpdir = tempfile.mkdtemp(prefix="glove_train_")
    try:
        corpus_txt = os.path.join(tmpdir, "corpus.txt")
        vocab_file = os.path.join(tmpdir, "vocab.txt")
        coocc_file = os.path.join(tmpdir, "cooccurrence.bin")
        coocc_shuf_file = os.path.join(tmpdir, "cooccurrence.shuf.bin")
        save_prefix = os.path.join(tmpdir, "vectors")

        _write_corpus_text(corpus, corpus_txt)

        # vocab_count < corpus.txt > vocab.txt
        with open(corpus_txt, "r", encoding="utf-8") as fin, open(vocab_file, "w", encoding="utf-8") as fout:
            result = subprocess.run(
                [vocab_count_bin, "-min-count", str(min_count), "-verbose", str(verbose)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"vocab_count failed:\n{result.stderr}")

        # cooccur < corpus.txt > cooccurrence.bin
        with open(corpus_txt, "r", encoding="utf-8") as fin, open(coocc_file, "wb") as fout:
            cmd = [
                cooccur_bin,
                "-memory", str(memory_gb),
                "-vocab-file", vocab_file,
                "-verbose", str(verbose),
                "-window-size", str(window_size),
            ]
            if symmetric:
                cmd.extend(["-symmetric", "1"])
            else:
                cmd.extend(["-symmetric", "0"])

            result = subprocess.run(
                cmd,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,

                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"cooccur failed:\n{result.stderr}")

        # shuffle < cooccurrence.bin > cooccurrence.shuf.bin
        with open(coocc_file, "rb") as fin, open(coocc_shuf_file, "wb") as fout:
            result = subprocess.run(
                [shuffle_bin, "-memory", str(memory_gb), "-verbose", str(verbose)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"shuffle failed:\n{result.stderr}")

        # glove
        cmd = [
            glove_bin,
            "-save-file", save_prefix,
            "-threads", str(num_threads),
            "-input-file", coocc_shuf_file,
            "-x-max", str(x_max),
            "-iter", str(epochs),
            "-vector-size", str(vector_size),
            "-binary", "0",
            "-vocab-file", vocab_file,
            "-verbose", str(verbose),
            "-eta", str(learning_rate),
        ]
        _run_command(cmd)

        vectors_txt = save_prefix + ".txt"
        return _load_glove_vectors(vectors_txt)

    finally:
        if cleanup_temp:
            shutil.rmtree(tmpdir, ignore_errors=True)


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

    if model.word_vectors is None or model.word_vectors.size == 0:
        return []

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
    return int(model.word_vectors.shape[1]) if model.word_vectors.size else 0


def save_corpus(corpus: Corpus, path: str) -> None:
    payload = {
        "dictionary": corpus.dictionary,
        "inverse_dictionary": corpus.inverse_dictionary,
        "word_counts": corpus.word_counts,
        "tokenized_docs": corpus.tokenized_docs,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_corpus(path: str) -> Corpus:
    with open(path, "rb") as f:
        payload = pickle.load(f)

    return Corpus(
        dictionary=payload["dictionary"],
        inverse_dictionary=payload["inverse_dictionary"],
        word_counts=payload["word_counts"],
        tokenized_docs=payload["tokenized_docs"],
    )