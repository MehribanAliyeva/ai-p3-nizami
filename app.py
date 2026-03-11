import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from corpus_utils import (
    read_corpus,
    split_into_documents,
    docs_to_tokens,
    corpus_summary,
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

st.set_page_config(page_title="Poetry NLP Analyzer", layout="wide")


@st.cache_data
def load_and_process_corpus(path):
    raw_text = read_corpus(path)
    docs = split_into_documents(raw_text)
    tokenized_docs = docs_to_tokens(docs)
    return raw_text, docs, tokenized_docs


@st.cache_resource
def build_model(tokenized_docs, vector_size, window, min_count, sg, epochs):
    return train_word2vec(
        tokenized_docs=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=4
    )


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


st.title("Poetry Corpus NLP Dashboard")

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

try:
    raw_text, docs, tokenized_docs = load_and_process_corpus(corpus_path)
except FileNotFoundError:
    st.error(f"File not found: {corpus_path}")
    st.stop()

summary = corpus_summary(
    tokenized_docs,
    frequent_threshold=freq_threshold,
    rare_threshold=rare_threshold
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Task 1: Dataset",
    "Matrices",
    "Task 2: Word2Vec",
    "Interactive Model"
])

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
    st.dataframe(freq_df, use_container_width=True)
    plot_bar(freq_df, "word", "frequency", "Top 20 Word Frequencies")

    st.subheader("Sample Documents")
    preview_df = pd.DataFrame({
        "document_id": [f"doc_{i+1}" for i in range(min(5, len(docs)))],
        "text": docs[:5]
    })
    st.dataframe(preview_df, use_container_width=True)

with tab2:
    st.header("Matrix Visualizations")

    st.subheader("Term-Document Matrix")
    tdm_df, _ = build_term_document_matrix(docs, max_features=tdm_features)
    st.dataframe(tdm_df.head(20), use_container_width=True)
    plot_heatmap(tdm_df, "Term-Document Matrix Heatmap", max_rows=min(25, len(tdm_df)), max_cols=min(25, len(tdm_df.columns)))

    st.subheader("Word-Word Co-occurrence Matrix")
    wwm_df = build_word_word_matrix(tokenized_docs, vocab_size=wwm_vocab_size, window_size=wwm_window)
    st.dataframe(wwm_df.iloc[:20, :20], use_container_width=True)
    plot_heatmap(wwm_df, "Word-Word Co-occurrence Matrix Heatmap", max_rows=25, max_cols=25)

with tab3:
    st.header("Word2Vec Results")

    model = build_model(tokenized_docs, vector_size, window, min_count, sg, epochs)

    st.markdown("### Chosen Parameters")
    param_df = pd.DataFrame({
        "Parameter": ["vector_size", "window", "min_count", "architecture", "epochs"],
        "Value": [vector_size, window, min_count, architecture, epochs]
    })
    st.table(param_df)

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
                "similarity": None
            })

    sim_df = pd.DataFrame(results)
    st.dataframe(sim_df, use_container_width=True)

    st.subheader("Vector Arithmetic")
    col1, col2 = st.columns(2)
    with col1:
        positive_words = st.text_input("Positive words (+)", "sevgi")
    with col2:
        negative_words = st.text_input("Negative words (-)", "pis")

    pos = [w.strip().lower() for w in positive_words.split(",") if w.strip()]
    neg = [w.strip().lower() for w in negative_words.split(",") if w.strip()]

    arithmetic_results = vector_arithmetic(model, positive=pos, negative=neg, topn=10)
    arithmetic_df = pd.DataFrame(arithmetic_results, columns=["word", "score"]) if arithmetic_results else pd.DataFrame()
    st.dataframe(arithmetic_df, use_container_width=True)

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

with tab4:
    st.header("Interactive Model Search")
    model = build_model(tokenized_docs, vector_size, window, min_count, sg, epochs)

    user_word = st.text_input("Search similar words", "sevgi").strip().lower()
    topn = st.slider("Top N", 3, 20, 10)

    if user_word:
        similar = get_similar_words(model, user_word, topn=topn)
        if similar:
            sim_user_df = pd.DataFrame(similar, columns=["word", "similarity"])
            st.dataframe(sim_user_df, use_container_width=True)
        else:
            st.error("Word not found in vocabulary.")

st.caption("Built for poetry corpus analysis: statistics, matrices, Word2Vec, and interactive similarity search.")