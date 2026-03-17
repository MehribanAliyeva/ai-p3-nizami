import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)

WORD2VEC_PATH = MODEL_DIR / "word2vec.model"
GLOVE_PATH = MODEL_DIR / "glove.pkl"
from gensim.models import Word2Vec

def save_word2vec_model(model, path=WORD2VEC_PATH):
    model.save(str(path))


def load_word2vec_model(path=WORD2VEC_PATH):
    if path.exists():
        return Word2Vec.load(str(path))
    return None
from corpus_utils import (
    read_corpus,
    split_into_documents,
    docs_to_tokens,
    corpus_summary,
    flatten,
)
from analysis import (
    build_term_document_matrix,
    build_word_word_matrix,
    get_top_word_frequencies,
)
from word2vec_model import (
    train_word2vec,
    get_similar_words,
    vector_arithmetic,
    cosine_similarity_between_words,
)
from glove_model import (
    build_cooccurrence_corpus,
    train_glove,
    get_similar_words_glove,
    vector_arithmetic_glove,
    cosine_similarity_glove,
    get_vocab_size,
    get_vector_size,
)
from feature_extraction import (
    extract_count_vectorizer_features,
    extract_tfidf_features,
    extract_pmi_features,
    extract_word2vec_features,
    extract_glove_features,
    prepare_features_for_rnn,
)
from rnn_classifier import (
    create_labels_from_corpus,
    run_full_comparison,
)

st.set_page_config(page_title="Poetry NLP Analyzer", layout="wide")
def save_glove_model(model, path=GLOVE_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_glove_model(path=GLOVE_PATH):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data
def load_and_process_corpus(path):
    raw_text = read_corpus(path)
    docs = split_into_documents(raw_text)
    tokenized_docs = docs_to_tokens(docs)
    return raw_text, docs, tokenized_docs


def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make dataframe safe for Streamlit Arrow serialization.
    Converts object/mixed columns to string.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str)
    return df


def plot_bar(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


def plot_heatmap(df, title, max_rows=25, max_cols=25):
    sub_df = df.iloc[:max_rows, :max_cols]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(sub_df, cmap="YlGnBu", ax=ax)
    ax.set_title(title)
    st.pyplot(fig)


def train_word2vec_once(tokenized_docs, vector_size, window, min_count, sg, epochs):
    with st.spinner("Training Word2Vec..."):
        model = train_word2vec(
            tokenized_docs=tokenized_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            epochs=epochs,
            workers=4,
        )

    save_word2vec_model(model)

    st.session_state["w2v_model"] = model
    st.session_state["w2v_params"] = {
        "vector_size": vector_size,
        "window": window,
        "min_count": min_count,
        "sg": sg,
        "epochs": epochs,
    }


def train_glove_once(tokenized_docs, vector_size, window_size, learning_rate, epochs):
    with st.spinner("Building co-occurrence corpus and training GloVe..."):
        corpus = build_cooccurrence_corpus(tokenized_docs, window_size=window_size)
        model = train_glove(
            corpus=corpus,
            no_components=vector_size,
            learning_rate=learning_rate,
            epochs=epochs,
        )

    save_glove_model(model)

    st.session_state["glove_model"] = model
    st.session_state["glove_params"] = {
        "vector_size": vector_size,
        "window_size": window_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }

def get_trained_word2vec():
    if "w2v_model" not in st.session_state:
        loaded = load_word2vec_model()
        if loaded is not None:
            st.session_state["w2v_model"] = loaded
    return st.session_state.get("w2v_model")


def get_trained_glove():
    if "glove_model" not in st.session_state:
        loaded = load_glove_model()
        if loaded is not None:
            st.session_state["glove_model"] = loaded
    return st.session_state.get("glove_model")


def get_trained_word2vec():
    return st.session_state.get("w2v_model")


def get_trained_glove():
    return st.session_state.get("glove_model")


st.title("Poetry Corpus NLP Dashboard")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")
corpus_path = st.sidebar.text_input("Corpus file path", "_ALL_CLEAN_CORPUS.txt")

freq_threshold = st.sidebar.number_input("Frequent word threshold", min_value=1, value=5)
rare_threshold = st.sidebar.number_input("Rare word threshold", min_value=1, value=1)

tdm_features = st.sidebar.slider("Term-document matrix vocab size", 10, 200, 50)
wwm_vocab_size = st.sidebar.slider("Word-word matrix vocab size", 10, 200, 50)
wwm_window = st.sidebar.slider("Word-word context window", 1, 10, 2)

st.sidebar.header("Word2Vec Parameters")
vector_size = st.sidebar.slider("Vector size", 50, 300, 100, step=50)
window = st.sidebar.slider("Word2Vec window", 2, 10, 5)
min_count = st.sidebar.slider("Min count", 1, 10, 2)
epochs = st.sidebar.slider("Epochs", 5, 100, 20)
architecture = st.sidebar.selectbox("Architecture", ["Skip-gram", "CBOW"])
sg = 1 if architecture == "Skip-gram" else 0

st.sidebar.header("GloVe Parameters")
glove_vector_size = st.sidebar.slider("GloVe vector size", 50, 300, 100, step=50)
glove_window = st.sidebar.slider("GloVe window", 2, 15, 5)
glove_learning_rate = st.sidebar.slider("GloVe learning rate", 0.01, 0.1, 0.05, step=0.01)
glove_epochs = st.sidebar.slider("GloVe epochs", 5, 30, 10)

# -----------------------------
# Load corpus first
# -----------------------------
try:
    raw_text, docs, tokenized_docs = load_and_process_corpus(corpus_path)
except FileNotFoundError:
    st.error(f"File not found: {corpus_path}")
    st.stop()

# -----------------------------
# Training buttons
# -----------------------------
st.sidebar.header("Train Models")

if st.sidebar.button("Train / Refresh Word2Vec"):
    train_word2vec_once(tokenized_docs, vector_size, window, min_count, sg, epochs)
    st.sidebar.success("Word2Vec trained.")

if st.sidebar.button("Train / Refresh GloVe"):
    train_glove_once(tokenized_docs, glove_vector_size, glove_window, glove_learning_rate, glove_epochs)
    st.sidebar.success("GloVe trained.")

summary = corpus_summary(
    tokenized_docs,
    frequent_threshold=freq_threshold,
    rare_threshold=rare_threshold
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Task 1: Dataset",
    "Matrices",
    "Task 2: Word2Vec",
    "Task 3: GloVe",
    "Task 4: Comparison",
    "Task 5: RNN Classification",
    "Interactive Model"
])

# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.header("Dataset Description")

    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", summary["num_documents"])
    col2.metric("Total Tokens", summary["total_tokens"])
    col3.metric("Distinct Words", summary["distinct_words"])

    col4, col5 = st.columns(2)
    col4.metric("Frequent Words", summary["frequent_words_count"])
    col5.metric("Rare Words", summary["rare_words_count"])

    st.subheader("Top 20 Most Frequent Words")
    freq_df = get_top_word_frequencies(summary["counter"], n=20)
    st.dataframe(safe_dataframe(freq_df), width="stretch")
    plot_bar(freq_df, "word", "frequency", "Top 20 Word Frequencies")

    st.subheader("Sample Documents")
    preview_df = pd.DataFrame({
        "document_id": [f"doc_{i+1}" for i in range(min(5, len(docs)))],
        "text": docs[:5]
    })
    st.dataframe(safe_dataframe(preview_df), width="stretch")

# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.header("Matrix Visualizations")

    st.subheader("Term-Document Matrix")
    tdm_df, _ = build_term_document_matrix(docs, max_features=tdm_features)
    st.dataframe(safe_dataframe(tdm_df.head(20)), width="stretch")
    plot_heatmap(
        tdm_df,
        "Term-Document Matrix Heatmap",
        max_rows=min(25, len(tdm_df)),
        max_cols=min(25, len(tdm_df.columns))
    )

    st.subheader("Word-Word Co-occurrence Matrix")
    wwm_df = build_word_word_matrix(
        tokenized_docs,
        vocab_size=wwm_vocab_size,
        window_size=wwm_window
    )
    st.dataframe(safe_dataframe(wwm_df.iloc[:20, :20]), width="stretch")
    plot_heatmap(wwm_df, "Word-Word Co-occurrence Matrix Heatmap", max_rows=25, max_cols=25)

# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.header("Word2Vec Results")

    model = get_trained_word2vec()
    if model is None:
        st.info("Train Word2Vec from the sidebar first.")
    else:
        st.markdown("### Chosen Parameters")
        param_df = pd.DataFrame({
            "Parameter": ["vector_size", "window", "min_count", "architecture", "epochs"],
            "Value": [str(vector_size), str(window), str(min_count), architecture, str(epochs)]
        })
        st.dataframe(safe_dataframe(param_df), width="stretch")

        st.markdown("""
**Parameter meaning**
- **vector_size**: size of each word embedding
- **window**: how many neighboring words are used as context
- **min_count**: ignores very rare words below this frequency
- **architecture**: Skip-gram learns well on smaller corpora and rare words, CBOW is usually faster
- **epochs**: number of training passes over the corpus
""")

        st.subheader("Similar Words for 10 Query Words")
        default_words = ["sevgi", "gözəl", "şah", "qəlb", "dünya", "gecə", "işıq", "qara", "yol", "gül"]
        query_words = st.text_area(
            "Enter 10 words separated by commas",
            value=", ".join(default_words)
        )
        query_words = [w.strip().lower() for w in query_words.split(",") if w.strip()]

        results = []
        for word in query_words[:10]:
            sims = get_similar_words(model, word, topn=5)
            if sims:
                for sim_word, score in sims:
                    results.append({
                        "query_word": word,
                        "similar_word": sim_word,
                        "similarity": round(score, 4)
                    })
            else:
                results.append({
                    "query_word": word,
                    "similar_word": "OOV / not in vocabulary",
                    "similarity": ""
                })

        sim_df = pd.DataFrame(results)
        st.dataframe(safe_dataframe(sim_df), width="stretch")

        st.subheader("Vector Arithmetic")
        col1, col2 = st.columns(2)
        with col1:
            positive_words = st.text_input("Positive words (+)", "sevgi")
        with col2:
            negative_words = st.text_input("Negative words (-)", "pis")

        pos = [w.strip().lower() for w in positive_words.split(",") if w.strip()]
        neg = [w.strip().lower() for w in negative_words.split(",") if w.strip()]

        arithmetic_results = vector_arithmetic(model, positive=pos, negative=neg, topn=10)
        arithmetic_df = pd.DataFrame(arithmetic_results, columns=["word", "score"]) if arithmetic_results else pd.DataFrame(columns=["word", "score"])
        st.dataframe(safe_dataframe(arithmetic_df), width="stretch")

        st.subheader("Pairwise Similarity")
        c1, c2 = st.columns(2)
        with c1:
            w1 = st.text_input("Word 1", "sevgi")
        with c2:
            w2 = st.text_input("Word 2", "məhəbbət")

        sim_score = cosine_similarity_between_words(model, w1.lower(), w2.lower())
        if sim_score is not None:
            st.success(f"Cosine similarity between '{w1}' and '{w2}': {sim_score:.4f}")
        else:
            st.warning("One or both words are not in the model vocabulary.")

# -----------------------------
# Tab 4
# -----------------------------
with tab4:
    st.header("GloVe Results")

    glove_model = get_trained_glove()
    if glove_model is None:
        st.info("Train GloVe from the sidebar first.")
    else:
        st.markdown("### Chosen Parameters")
        glove_param_df = pd.DataFrame({
            "Parameter": ["vector_size", "window_size", "learning_rate", "epochs", "alpha"],
            "Value": [
                str(glove_vector_size),
                str(glove_window),
                str(glove_learning_rate),
                str(glove_epochs),
                "0.75 (default)"
            ]
        })
        st.dataframe(safe_dataframe(glove_param_df), width="stretch")

        st.markdown("""
**GloVe Parameter Meanings:**
- **vector_size (no_components)**: Dimensionality of word embeddings
- **window_size**: Co-occurrence window size
- **learning_rate**: Step size for optimization
- **epochs**: Number of training iterations
- **alpha**: Weighting function exponent
""")

        st.markdown(
            f"**Model Info:** Vocabulary size = {get_vocab_size(glove_model):,}, "
            f"Vector dimension = {get_vector_size(glove_model)}"
        )

        st.subheader("Similar Words for 10 Query Words")
        default_words_glove = ["sevgi", "gözəl", "şah", "qəlb", "dünya", "gecə", "işıq", "qara", "yol", "gül"]
        query_words_glove = st.text_area(
            "Enter 10 words separated by commas (GloVe)",
            value=", ".join(default_words_glove),
            key="glove_query"
        )
        query_words_glove = [w.strip().lower() for w in query_words_glove.split(",") if w.strip()]

        glove_results = []
        for word in query_words_glove[:10]:
            sims = get_similar_words_glove(glove_model, word, topn=5)
            if sims:
                for sim_word, score in sims:
                    glove_results.append({
                        "query_word": word,
                        "similar_word": sim_word,
                        "similarity": round(score, 4)
                    })
            else:
                glove_results.append({
                    "query_word": word,
                    "similar_word": "OOV / not in vocabulary",
                    "similarity": ""
                })

        glove_sim_df = pd.DataFrame(glove_results)
        st.dataframe(safe_dataframe(glove_sim_df), width="stretch")

        st.subheader("Vector Arithmetic")
        col1, col2 = st.columns(2)
        with col1:
            positive_words_glove = st.text_input("Positive words (+) [GloVe]", "sevgi", key="glove_pos")
        with col2:
            negative_words_glove = st.text_input("Negative words (-) [GloVe]", "pis", key="glove_neg")

        pos_glove = [w.strip().lower() for w in positive_words_glove.split(",") if w.strip()]
        neg_glove = [w.strip().lower() for w in negative_words_glove.split(",") if w.strip()]

        arithmetic_results_glove = vector_arithmetic_glove(glove_model, positive=pos_glove, negative=neg_glove, topn=10)
        arithmetic_df_glove = pd.DataFrame(arithmetic_results_glove, columns=["word", "score"]) if arithmetic_results_glove else pd.DataFrame(columns=["word", "score"])
        st.dataframe(safe_dataframe(arithmetic_df_glove), width="stretch")

        st.subheader("Pairwise Similarity")
        c1, c2 = st.columns(2)
        with c1:
            w1_glove = st.text_input("Word 1 [GloVe]", "sevgi", key="glove_w1")
        with c2:
            w2_glove = st.text_input("Word 2 [GloVe]", "məhəbbət", key="glove_w2")

        sim_score_glove = cosine_similarity_glove(glove_model, w1_glove.lower(), w2_glove.lower())
        if sim_score_glove is not None:
            st.success(f"Cosine similarity between '{w1_glove}' and '{w2_glove}': {sim_score_glove:.4f}")
        else:
            st.warning("One or both words are not in the model vocabulary.")

# -----------------------------
# Tab 5
# -----------------------------
with tab5:
    st.header("Task 4: Word2Vec vs GloVe Comparison")

    w2v_model = get_trained_word2vec()
    glove_model = get_trained_glove()

    if w2v_model is None or glove_model is None:
        st.info("Train both Word2Vec and GloVe from the sidebar first.")
    else:
        st.markdown("""
This task compares the results from **Task 2 (Word2Vec)** and **Task 3 (GloVe)** to identify:
- Which model performs better for synonym detection
- Differences in semantic similarity patterns
- Strengths and weaknesses of each approach
""")

        st.subheader("1. Model Specifications Comparison")
        spec_comparison = pd.DataFrame({
            "Specification": ["Architecture", "Vector Size", "Window Size", "Training Method", "Vocabulary Size"],
            "Word2Vec": [
                "Skip-gram" if sg == 1 else "CBOW",
                str(vector_size),
                str(window),
                "Predictive (Neural Network)",
                f"{len(w2v_model.wv):,}"
            ],
            "GloVe": [
                "Count-based",
                str(glove_vector_size),
                str(glove_window),
                "Co-occurrence Matrix Factorization",
                f"{get_vocab_size(glove_model):,}"
            ]
        })
        st.dataframe(safe_dataframe(spec_comparison), width="stretch")

        st.subheader("2. Synonym Detection Comparison")
        comparison_words = st.text_input(
            "Enter words to compare (comma-separated)",
            value="sevgi, gözəl, şah, qəlb, dünya, gecə, işıq, qara, yol, gül",
            key="comparison_words"
        )
        comparison_words = [w.strip().lower() for w in comparison_words.split(",") if w.strip()]

        comparison_results = []
        for word in comparison_words[:10]:
            w2v_similar = []
            if word in w2v_model.wv:
                w2v_top = w2v_model.wv.most_similar(word, topn=3)
                w2v_similar = [f"{w} ({s:.3f})" for w, s in w2v_top]
            else:
                w2v_similar = ["OOV"]

            glove_similar = []
            word_id = glove_model.dictionary.get(word)
            if word_id is not None:
                glove_top = glove_model.most_similar(word_id, number=3)
                for sim_id, score in glove_top:
                    sim_word = glove_model.inverse_dictionary.get(sim_id)
                    if sim_word is not None:
                        glove_similar.append(f"{sim_word} ({float(score):.3f})")
            else:
                glove_similar = ["OOV"]

            comparison_results.append({
                "Query Word": word,
                "Word2Vec Top 3": ", ".join(w2v_similar),
                "GloVe Top 3": ", ".join(glove_similar)
            })

        comparison_df = pd.DataFrame(comparison_results)
        st.dataframe(safe_dataframe(comparison_df), width="stretch")

        st.subheader("3. Pairwise Similarity Comparison")
        col1, col2 = st.columns(2)
        with col1:
            pair_w1 = st.text_input("Word 1", "sevgi", key="comp_w1")
        with col2:
            pair_w2 = st.text_input("Word 2", "məhəbbət", key="comp_w2")

        w2v_sim = None
        glove_sim = None

        if pair_w1 in w2v_model.wv and pair_w2 in w2v_model.wv:
            w2v_sim = w2v_model.wv.similarity(pair_w1, pair_w2)

        glove_sim = cosine_similarity_glove(glove_model, pair_w1, pair_w2)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word2Vec Similarity", f"{w2v_sim:.4f}" if w2v_sim is not None else "OOV")
        with col2:
            st.metric("GloVe Similarity", f"{glove_sim:.4f}" if glove_sim is not None else "OOV")
        with col3:
            if w2v_sim is not None and glove_sim is not None:
                st.metric("Difference", f"{abs(w2v_sim - glove_sim):.4f}")
            else:
                st.metric("Difference", "N/A")

        st.subheader("4. Vector Arithmetic Comparison")
        col1, col2 = st.columns(2)
        with col1:
            arith_positive = st.text_input("Positive words (+)", "sevgi", key="comp_pos")
        with col2:
            arith_negative = st.text_input("Negative words (-)", "pis", key="comp_neg")

        pos_words = [w.strip().lower() for w in arith_positive.split(",") if w.strip()]
        neg_words = [w.strip().lower() for w in arith_negative.split(",") if w.strip()]

        try:
            w2v_arith = w2v_model.wv.most_similar(positive=pos_words, negative=neg_words, topn=5)
            w2v_arith_df = pd.DataFrame(w2v_arith, columns=["word", "score"])
        except Exception:
            w2v_arith_df = pd.DataFrame({"word": ["Error"], "score": [0]})

        glove_arith = vector_arithmetic_glove(glove_model, positive=pos_words, negative=neg_words, topn=5)
        glove_arith_df = (
            pd.DataFrame(glove_arith, columns=["word", "score"])
            if glove_arith else
            pd.DataFrame({"word": ["Error"], "score": [0]})
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Word2Vec Results:**")
            st.dataframe(safe_dataframe(w2v_arith_df), width="stretch")
        with col2:
            st.markdown("**GloVe Results:**")
            st.dataframe(safe_dataframe(glove_arith_df), width="stretch")

        st.subheader("5. Analysis & Conclusions")
        st.markdown("""
### Key Findings

**Methodology Differences**
- **Word2Vec:** predictive approach using local context
- **GloVe:** count-based approach using global co-occurrence statistics

**Performance Observations**
1. Compare synonym quality above
2. Compare similarity scores above
3. Compare vector arithmetic behavior
4. Compare vocabulary coverage

**Recommendations**
- For small poetry corpora, results can vary based on preprocessing and vocabulary size
- Word2Vec often handles local context well
- GloVe can capture broader global relations
""")

        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison Table (CSV)",
            data=csv_data,
            file_name="word2vec_vs_glove_comparison.csv",
            mime="text/csv"
        )

# -----------------------------
# Tab 6
# -----------------------------
with tab6:
    st.header("Task 5: RNN Text Classification")

    st.markdown("""
This task trains **RNN, Bidirectional RNN, and LSTM** models for text classification
using **5 different feature extraction methods**: Count Vectorizer, TF-IDF, PMI, Word2Vec, and GloVe.

**Classification Task:** Classify poems into **Love Poems** vs **Other Themes**
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        rnn_epochs = st.number_input("Training Epochs", min_value=5, max_value=50, value=10)
    with col2:
        rnn_units = st.number_input("RNN Units", min_value=16, max_value=128, value=32, step=16)
    with col3:
        max_features = st.number_input("Max Features", min_value=50, max_value=500, value=100, step=50)

    if st.button("Train All Models", key="train_rnn"):
        w2v_model = get_trained_word2vec()
        glove_model = get_trained_glove()

        if w2v_model is None or glove_model is None:
            st.warning("Train both Word2Vec and GloVe first.")
        else:
            with st.spinner("Training models... This may take a few minutes..."):
                labels = create_labels_from_corpus(tokenized_docs)
                love_count = int(np.sum(labels))
                other_count = int(len(labels) - love_count)

                st.success(f"Labels created: {love_count} Love Poems, {other_count} Other Themes")

                X_count, _ = extract_count_vectorizer_features(docs, max_features=max_features)
                X_count = prepare_features_for_rnn(X_count, max_len=50)

                X_tfidf, _ = extract_tfidf_features(docs, max_features=max_features)
                X_tfidf = prepare_features_for_rnn(X_tfidf, max_len=50)

                X_pmi, _ = extract_pmi_features(docs, tokenized_docs, max_features=max_features)
                X_pmi = prepare_features_for_rnn(X_pmi, max_len=50)

                X_w2v = extract_word2vec_features(docs, tokenized_docs, w2v_model, vector_size=vector_size)
                X_w2v = prepare_features_for_rnn(X_w2v, max_len=50)

                X_glove = extract_glove_features(docs, tokenized_docs, glove_model, vector_size=glove_vector_size)
                X_glove = prepare_features_for_rnn(X_glove, max_len=50)

                X_dict = {
                    "Count Vectorizer": X_count,
                    "TF-IDF": X_tfidf,
                    "PMI": X_pmi,
                    "Word2Vec": X_w2v,
                    "GloVe": X_glove
                }

                results = run_full_comparison(
                    X_dict,
                    labels,
                    model_types=["RNN", "BiRNN", "LSTM"],
                    epochs=rnn_epochs,
                    units=rnn_units,
                    verbose=0
                )

                results_df = pd.DataFrame(results)
                st.dataframe(safe_dataframe(results_df), width="stretch")

                results_df["Accuracy_float"] = results_df["Accuracy"].astype(float)
                results_df["F1_float"] = results_df["F1-Score"].astype(float)

                best_row = results_df.loc[results_df["Accuracy_float"].idxmax()]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Feature", str(best_row["Feature"]))
                col2.metric("Model", str(best_row["Model"]))
                col3.metric("Accuracy", str(best_row["Accuracy"]))
                col4.metric("F1-Score", str(best_row["F1-Score"]))

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                pivot_acc = results_df.pivot(index="Feature", columns="Model", values="Accuracy_float")
                pivot_acc.plot(kind="bar", ax=ax1)
                ax1.set_title("Accuracy Comparison")
                ax1.set_xlabel("Feature Type")
                ax1.set_ylabel("Accuracy")
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
                ax1.grid(axis="y", alpha=0.3)

                pivot_f1 = results_df.pivot(index="Feature", columns="Model", values="F1_float")
                pivot_f1.plot(kind="bar", ax=ax2)
                ax2.set_title("F1-Score Comparison")
                ax2.set_xlabel("Feature Type")
                ax2.set_ylabel("F1-Score")
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
                ax2.grid(axis="y", alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                csv_data = results_df[["Feature", "Model", "Accuracy", "F1-Score"]].to_csv(index=False)
                st.download_button(
                    label="Download Results Table (CSV)",
                    data=csv_data,
                    file_name="rnn_classification_results.csv",
                    mime="text/csv"
                )
    else:
        st.info("Click 'Train All Models' to start training and comparison.")

# -----------------------------
# Tab 7
# -----------------------------
with tab7:
    st.header("Interactive Model Search")

    model = get_trained_word2vec()
    if model is None:
        st.info("Train Word2Vec from the sidebar first.")
    else:
        user_word = st.text_input("Search similar words", "sevgi").strip().lower()
        topn = st.slider("Top N", 3, 20, 10)

        if user_word:
            similar = get_similar_words(model, user_word, topn=topn)
            if similar:
                sim_user_df = pd.DataFrame(similar, columns=["word", "similarity"])
                st.dataframe(safe_dataframe(sim_user_df), width="stretch")
            else:
                st.error("Word not found in vocabulary.")

st.caption("Built for poetry corpus analysis: statistics, matrices, Word2Vec, GloVe, and interactive similarity search.")