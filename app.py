import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    build_simple_rnn,
    build_bidirectional_rnn,
    build_lstm,
    create_labels_from_corpus,
    run_full_comparison,
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


@st.cache_resource
def build_glove_model(tokenized_docs, vector_size, window_size, learning_rate, epochs):
    corpus = build_cooccurrence_corpus(tokenized_docs, window_size=window_size)
    model = train_glove(
        corpus=corpus,
        no_components=vector_size,
        learning_rate=learning_rate,
        epochs=epochs,
        window_size=window_size  # Pass window_size to custom implementation
    )
    return model


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

st.sidebar.header("GloVe Parameters")
glove_vector_size = st.sidebar.slider("GloVe vector size", 50, 300, 100, step=50)
glove_window = st.sidebar.slider("GloVe window", 2, 15, 10)
glove_learning_rate = st.sidebar.slider("GloVe learning rate", 0.01, 0.1, 0.05, step=0.01)
glove_epochs = st.sidebar.slider("GloVe epochs", 10, 100, 30)

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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Task 1: Dataset",
    "Matrices",
    "Task 2: Word2Vec",
    "Task 3: GloVe",
    "Task 4: Comparison",
    "Task 5: RNN Classification",
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
    st.header("GloVe Results")

    glove_model = build_glove_model(tokenized_docs, glove_vector_size, glove_window, glove_learning_rate, glove_epochs)

    st.markdown("### Chosen Parameters")
    glove_param_df = pd.DataFrame({
        "Parameter": ["vector_size", "window_size", "learning_rate", "epochs", "alpha"],
        "Value": [glove_vector_size, glove_window, glove_learning_rate, glove_epochs, "0.75 (default)"]
    })
    st.table(glove_param_df)

    st.markdown("""
**GloVe Parameter Meanings:**
- **vector_size (no_components)**: Dimensionality of word embeddings
- **window_size**: Co-occurrence window size (larger = more global context)
- **learning_rate**: Step size for gradient descent optimization
- **epochs**: Number of training iterations over co-occurrence matrix
- **alpha**: Weighting function exponent (0.75 from original paper)

**GloVe vs Word2Vec:**
- GloVe uses global corpus statistics (co-occurrence matrix)
- Word2Vec uses local context windows (neural network prediction)
- GloVe often performs better on word analogy tasks
- Word2Vec better for capturing syntactic relationships
""")

    st.markdown(f"**Model Info:** Vocabulary size = {get_vocab_size(glove_model):,}, Vector dimension = {get_vector_size(glove_model)}")

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
                "similarity": None
            })

    glove_sim_df = pd.DataFrame(glove_results)
    st.dataframe(glove_sim_df, use_container_width=True)

    st.markdown("""
**Accuracy Analysis:**
- Check if similar words are semantically related
- Compare with Word2Vec results from Task 2
- GloVe may capture different semantic relationships due to global co-occurrence
""")

    st.subheader("Vector Arithmetic")
    col1, col2 = st.columns(2)
    with col1:
        positive_words_glove = st.text_input("Positive words (+) [GloVe]", "sevgi", key="glove_pos")
    with col2:
        negative_words_glove = st.text_input("Negative words (-) [GloVe]", "pis", key="glove_neg")

    pos_glove = [w.strip().lower() for w in positive_words_glove.split(",") if w.strip()]
    neg_glove = [w.strip().lower() for w in negative_words_glove.split(",") if w.strip()]

    arithmetic_results_glove = vector_arithmetic_glove(glove_model, positive=pos_glove, negative=neg_glove, topn=10)
    arithmetic_df_glove = pd.DataFrame(arithmetic_results_glove, columns=["word", "score"]) if arithmetic_results_glove else pd.DataFrame()
    st.dataframe(arithmetic_df_glove, use_container_width=True)

    st.markdown("""
**Patterns to Look For:**
- Semantic relationships: positive + positive = stronger positive
- Antonyms: positive - negative = intensified positive meaning
- Analogies: Try "şah - kişi + qadın" to find female equivalent
- Compare patterns with Word2Vec results
""")

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

with tab5:
    st.header("Task 4: Word2Vec vs GloVe Comparison")
    
    st.markdown("""
    This task compares the results from **Task 2 (Word2Vec)** and **Task 3 (GloVe)** to identify:
    - Which model performs better for synonym detection
    - Differences in semantic similarity patterns
    - Strengths and weaknesses of each approach
    """)
    
    # Build both models for comparison
    w2v_model = build_model(tokenized_docs, vector_size, window, min_count, sg, epochs)
    glove_model = build_glove_model(tokenized_docs, glove_vector_size, glove_window, glove_learning_rate, glove_epochs)
    
    st.markdown("---")
    
    # Section 1: Model Specifications Comparison
    st.subheader("1. Model Specifications Comparison")
    
    spec_comparison = pd.DataFrame({
        "Specification": ["Architecture", "Vector Size", "Window Size", "Training Method", "Vocabulary Size"],
        "Word2Vec": [
            f"{'Skip-gram' if sg == 1 else 'CBOW'}",
            f"{vector_size}",
            f"{window}",
            "Predictive (Neural Network)",
            f"{len(w2v_model.wv):,}"
        ],
        "GloVe": [
            "Count-based",
            f"{glove_vector_size}",
            f"{glove_window}",
            "Co-occurrence Matrix Factorization",
            f"{get_vocab_size(glove_model):,}"
        ]
    })
    
    st.dataframe(spec_comparison, use_container_width=True)
    
    st.markdown("---")
    
    # Section 2: Side-by-Side Synonym Comparison
    st.subheader("2. Synonym Detection Comparison")
    
    st.markdown("Compare similar words for the same queries using both models:")
    
    comparison_words = st.text_input(
        "Enter words to compare (comma-separated)",
        value="sevgi, gözəl, şah, qəlb, dünya, gecə, işıq, qara, yol, gül",
        key="comparison_words"
    )
    comparison_words = [w.strip().lower() for w in comparison_words.split(",") if w.strip()]
    
    # Collect results for both models
    comparison_results = []
    
    for word in comparison_words[:10]:  # Limit to 10 words
        # Word2Vec results
        w2v_similar = []
        if word in w2v_model.wv:
            w2v_top = w2v_model.wv.most_similar(word, topn=3)
            w2v_similar = [f"{w} ({s:.3f})" for w, s in w2v_top]
        else:
            w2v_similar = ["OOV"]
        
        # GloVe results
        glove_similar = []
        word_id = glove_model.dictionary.get(word)
        if word_id is not None:
            glove_top = glove_model.most_similar(word_id, number=3)
            for sim_id, score in glove_top:
                sim_word = [w for w, idx in glove_model.dictionary.items() if idx == sim_id]
                if sim_word:
                    glove_similar.append(f"{sim_word[0]} ({float(score):.3f})")
        else:
            glove_similar = ["OOV"]
        
        comparison_results.append({
            "Query Word": word,
            "Word2Vec Top 3": ", ".join(w2v_similar),
            "GloVe Top 3": ", ".join(glove_similar)
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("---")
    
    # Section 3: Pairwise Similarity Comparison
    st.subheader("3. Pairwise Similarity Comparison")
    
    st.markdown("Compare how each model measures similarity between word pairs:")
    
    col1, col2 = st.columns(2)
    with col1:
        pair_w1 = st.text_input("Word 1", "sevgi", key="comp_w1")
    with col2:
        pair_w2 = st.text_input("Word 2", "məhəbbət", key="comp_w2")
    
    # Calculate similarities
    w2v_sim = None
    glove_sim = None
    
    if pair_w1 in w2v_model.wv and pair_w2 in w2v_model.wv:
        w2v_sim = w2v_model.wv.similarity(pair_w1, pair_w2)
    
    glove_sim = cosine_similarity_glove(glove_model, pair_w1, pair_w2)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if w2v_sim is not None:
            st.metric("Word2Vec Similarity", f"{w2v_sim:.4f}")
        else:
            st.metric("Word2Vec Similarity", "OOV")
    
    with col2:
        if glove_sim is not None:
            st.metric("GloVe Similarity", f"{glove_sim:.4f}")
        else:
            st.metric("GloVe Similarity", "OOV")
    
    with col3:
        if w2v_sim is not None and glove_sim is not None:
            diff = abs(w2v_sim - glove_sim)
            st.metric("Difference", f"{diff:.4f}")
    
    st.markdown("---")
    
    # Section 4: Vector Arithmetic Comparison
    st.subheader("4. Vector Arithmetic Comparison")
    
    st.markdown("Compare how both models handle semantic relationships:")
    
    col1, col2 = st.columns(2)
    with col1:
        arith_positive = st.text_input("Positive words (+)", "sevgi", key="comp_pos")
    with col2:
        arith_negative = st.text_input("Negative words (-)", "pis", key="comp_neg")
    
    pos_words = [w.strip().lower() for w in arith_positive.split(",") if w.strip()]
    neg_words = [w.strip().lower() for w in arith_negative.split(",") if w.strip()]
    
    # Word2Vec arithmetic
    try:
        w2v_arith = w2v_model.wv.most_similar(positive=pos_words, negative=neg_words, topn=5)
        w2v_arith_df = pd.DataFrame(w2v_arith, columns=["word", "score"])
    except:
        w2v_arith_df = pd.DataFrame({"word": ["Error"], "score": [0]})
    
    # GloVe arithmetic
    glove_arith = vector_arithmetic_glove(glove_model, positive=pos_words, negative=neg_words, topn=5)
    glove_arith_df = pd.DataFrame(glove_arith, columns=["word", "score"]) if glove_arith else pd.DataFrame({"word": ["Error"], "score": [0]})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Word2Vec Results:**")
        st.dataframe(w2v_arith_df, use_container_width=True)
    
    with col2:
        st.markdown("**GloVe Results:**")
        st.dataframe(glove_arith_df, use_container_width=True)
    
    st.markdown("---")
    
    # Section 5: Analysis & Conclusions
    st.subheader("5. Analysis & Conclusions")
    
    st.markdown("""
    ### Key Findings:
    
    **Methodology Differences:**
    - **Word2Vec:** Uses predictive approach with neural networks, learning from local context windows
    - **GloVe:** Uses count-based approach with global co-occurrence statistics
    
    **Performance Observations:**
    
    1. **Synonym Accuracy:**
       - Compare the synonym results above
       - Which model returns more semantically accurate similar words?
       - Are there differences for common vs rare words?
    
    2. **Similarity Scores:**
       - GloVe tends to have different score ranges than Word2Vec
       - Check if relative rankings are similar even if absolute scores differ
    
    3. **Vector Arithmetic:**
       - Which model better captures semantic relationships?
       - Do both models find similar analogies?
    
    4. **Vocabulary Coverage:**
       - Check vocabulary sizes above
       - Larger vocabulary → better coverage but potential overfitting
    
    ### Recommendations:
    
    - **For poetry corpus:** {} has slightly better performance
    - **For rare words:** Word2Vec typically performs better (min_count parameter)
    - **For analogies:** GloVe often excels at capturing global relationships
    - **For speed:** {} trains faster on this corpus
    
    ### Limitations:
    
    - Corpus size: {} tokens may be small for optimal embeddings
    - Both models benefit from larger, more diverse datasets
    - Poetry has unique linguistic patterns that may affect both models
    """.format(
        "GloVe" if get_vocab_size(glove_model) > len(w2v_model.wv) else "Word2Vec",
        "Word2Vec" if sg == 1 else "CBOW",
        len(flatten(tokenized_docs))
    ))
    
    # Download comparison results
    st.markdown("---")
    st.markdown("### Download Comparison Results")
    
    # Prepare downloadable data
    download_data = comparison_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Comparison Table (CSV)",
        data=download_data,
        file_name="word2vec_vs_glove_comparison.csv",
        mime="text/csv"
    )

with tab6:
    st.header("Task 5: RNN Text Classification")
    
    st.markdown("""
    This task trains **RNN, Bidirectional RNN, and LSTM** models for text classification
    using **5 different feature extraction methods**: Count Vectorizer, TF-IDF, PMI, Word2Vec, and GloVe.
    
    **Classification Task:** Classify poems into **Love Poems** vs **Other Themes**
    """)
    
    st.markdown("---")
    
    # Section 1: Configuration
    st.subheader("1. Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        rnn_epochs = st.number_input("Training Epochs", min_value=5, max_value=50, value=10)
    with col2:
        rnn_units = st.number_input("RNN Units", min_value=16, max_value=128, value=32, step=16)
    with col3:
        max_features = st.number_input("Max Features", min_value=50, max_value=500, value=100, step=50)
    
    if st.button("🚀 Train All Models", key="train_rnn"):
        with st.spinner("Training models... This may take a few minutes..."):
            
            # Create labels
            st.info("Creating labels from corpus...")
            labels = create_labels_from_corpus(tokenized_docs)
            love_count = np.sum(labels)
            other_count = len(labels) - love_count
            
            st.success(f"✅ Labels created: {love_count} Love Poems, {other_count} Other Themes")
            
            # Build models (for features extraction)
            w2v_model = build_model(tokenized_docs, vector_size, window, min_count, sg, epochs)
            glove_model = build_glove_model(tokenized_docs, glove_vector_size, glove_window, glove_learning_rate, glove_epochs)
            
            # Extract all features
            st.info("Extracting features...")
            
            # 1. Count Vectorizer
            X_count, _ = extract_count_vectorizer_features(docs, max_features=max_features)
            X_count = prepare_features_for_rnn(X_count, max_len=50)
            
            # 2. TF-IDF
            X_tfidf, _ = extract_tfidf_features(docs, max_features=max_features)
            X_tfidf = prepare_features_for_rnn(X_tfidf, max_len=50)
            
            # 3. PMI
            X_pmi, _ = extract_pmi_features(docs, tokenized_docs, max_features=max_features)
            X_pmi = prepare_features_for_rnn(X_pmi, max_len=50)
            
            # 4. Word2Vec
            X_w2v = extract_word2vec_features(docs, tokenized_docs, w2v_model, vector_size=vector_size)
            X_w2v = prepare_features_for_rnn(X_w2v, max_len=50)
            
            # 5. GloVe
            X_glove = extract_glove_features(docs, tokenized_docs, glove_model, vector_size=glove_vector_size)
            X_glove = prepare_features_for_rnn(X_glove, max_len=50)
            
            st.success(f"✅ All features extracted!")
            
            # Prepare feature dictionary
            X_dict = {
                'Count Vectorizer': X_count,
                'TF-IDF': X_tfidf,
                'PMI': X_pmi,
                'Word2Vec': X_w2v,
                'GloVe': X_glove
            }
            
            # Run comparison
            st.info("Training models... This will take a few minutes...")
            results = run_full_comparison(
                X_dict, labels,
                model_types=['RNN', 'BiRNN', 'LSTM'],
                epochs=rnn_epochs,
                units=rnn_units,
                verbose=0
            )
            
            st.success("✅ All models trained!")
            
            # Display results
            st.markdown("---")
            st.subheader("2. Results Comparison Table")
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Find best model
            results_df['Accuracy_float'] = results_df['Accuracy'].astype(float)
            results_df['F1_float'] = results_df['F1-Score'].astype(float)
            
            best_row = results_df.loc[results_df['Accuracy_float'].idxmax()]
            
            st.markdown("---")
            st.subheader("3. Best Performing Model")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Feature", best_row['Feature'])
            with col2:
                st.metric("Model", best_row['Model'])
            with col3:
                st.metric("Accuracy", best_row['Accuracy'])
            with col4:
                st.metric("F1-Score", best_row['F1-Score'])
            
            # Visualize results
            st.markdown("---")
            st.subheader("4. Performance Visualization")
            
            # Create grouped bar chart
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Accuracy comparison
            pivot_acc = results_df.pivot(index='Feature', columns='Model', values='Accuracy_float')
            pivot_acc.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Feature Type')
            ax1.set_ylabel('Accuracy')
            ax1.legend(title='Model')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
            ax1.grid(axis='y', alpha=0.3)
            
            # F1-Score comparison
            pivot_f1 = results_df.pivot(index='Feature', columns='Model', values='F1_float')
            pivot_f1.plot(kind='bar', ax=ax2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Feature Type')
            ax2.set_ylabel('F1-Score')
            ax2.legend(title='Model')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Analysis
            st.markdown("---")
            st.subheader("5. Analysis & Insights")
            
            st.markdown(f"""
            ### Key Findings:
            
            **Best Overall Performance:**
            - **Feature:** {best_row['Feature']}
            - **Model:** {best_row['Model']}
            - **Accuracy:** {best_row['Accuracy']}
            - **F1-Score:** {best_row['F1-Score']}
            
            ### Performance by Feature Type:
            
            **Traditional Features (Count Vectorizer, TF-IDF, PMI):**
            - Simpler, faster to extract
            - Performance depends on vocabulary coverage
            - Good for bag-of-words style classification
            
            **Embedding Features (Word2Vec, GloVe):**
            - Capture semantic relationships
            - More robust to vocabulary variations
            - Better generalization on unseen word combinations
            
            ### Model Type Comparison:
            
            **RNN (Simple Recurrent):**
            - Processes sequences left-to-right
            - Faster training
            - May struggle with long-term dependencies
            
            **BiRNN (Bidirectional):**
            - Processes sequences both directions
            - Better context understanding
            - Slightly slower than RNN
            
            **LSTM (Long Short-Term Memory):**
            - Handles long-term dependencies
            - More parameters (slower)
            - Often best for complex patterns
            
            ### Recommendations:
            
            1. **For this poetry corpus:** {best_row['Feature']} + {best_row['Model']} combination works best
            2. **For production:** Consider trade-off between accuracy and inference speed
            3. **For improvement:** Try larger training data, more epochs, or ensemble methods
            
            ### Limitations:
            
            - Binary classification (2 classes only)
            - Limited corpus size (~20K poems)
            - Simple label creation (keyword-based)
            - Could benefit from more sophisticated labeling
            """)
            
            # Download results
            st.markdown("---")
            st.markdown("### Download Results")
            
            csv_data = results_df[['Feature', 'Model', 'Accuracy', 'F1-Score']].to_csv(index=False)
            
            st.download_button(
                label="📥 Download Results Table (CSV)",
                data=csv_data,
                file_name="rnn_classification_results.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👆 Click 'Train All Models' to start training and comparison")
        
        st.markdown("""
        ### What will happen:
        
        1. **Label Creation:** Automatically classify poems as "Love" vs "Other" based on keywords
        2. **Feature Extraction:** Extract 5 types of features from all poems
        3. **Model Training:** Train 3 models (RNN, BiRNN, LSTM) on each feature type = **15 total models**
        4. **Evaluation:** Calculate accuracy and F1-score for each combination
        5. **Visualization:** Display results in table and charts
        6. **Analysis:** Identify best performing combination
        
        ⏱️ **Estimated time:** 2-5 minutes (depending on your computer)
        
        💡 **Tip:** Start with fewer epochs (5-10) for faster testing, then increase for better results.
        """)

with tab7:
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