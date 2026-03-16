"""
RNN Classifier Module for Task 5
Simple implementations of RNN, Bidirectional RNN, and LSTM
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


def build_simple_rnn(input_shape, units=32, dropout=0.2):
    """
    Build a simple RNN model.
    
    Args:
        input_shape: Shape of input (timesteps, features)
        units: Number of RNN units
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        SimpleRNN(units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_bidirectional_rnn(input_shape, units=32, dropout=0.2):
    """
    Build a Bidirectional RNN model.
    
    Args:
        input_shape: Shape of input (timesteps, features)
        units: Number of RNN units
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Bidirectional(SimpleRNN(units, return_sequences=False), input_shape=input_shape),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_lstm(input_shape, units=32, dropout=0.2):
    """
    Build an LSTM model.
    
    Args:
        input_shape: Shape of input (timesteps, features)
        units: Number of LSTM units
        dropout: Dropout rate
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_and_evaluate(model, X, y, test_size=0.2, epochs=10, batch_size=32, verbose=0):
    """
    Train and evaluate a model.
    
    Args:
        model: Keras model
        X: Feature matrix
        y: Labels
        test_size: Test set size
        epochs: Number of training epochs
        batch_size: Batch size
        verbose: Verbosity level
    
    Returns:
        Dictionary with metrics (accuracy, f1_score)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train model
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=verbose
    )
    
    # Predict
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'y_test': y_test,
        'y_pred': y_pred
    }


def create_labels_from_corpus(tokenized_docs, love_words=None):
    """
    Create binary labels from corpus based on content.
    
    Label 1: Love poems (contain love-related words)
    Label 0: Other themes
    
    Args:
        tokenized_docs: List of tokenized documents
        love_words: Set of love-related words
    
    Returns:
        Binary labels (numpy array)
    """
    if love_words is None:
        # Default love-related words in Azerbaijani
        love_words = {
            'sevgi', 'məhəbbət', 'eşq', 'könül', 'yar', 
            'sevmək', 'sevilmək', 'qəlb', 'ürək', 'can'
        }
    
    labels = []
    
    for doc in tokenized_docs:
        # Check if document contains love words
        has_love_word = any(word in love_words for word in doc)
        labels.append(1 if has_love_word else 0)
    
    return np.array(labels)


def run_full_comparison(X_dict, y, model_types=['RNN', 'BiRNN', 'LSTM'], 
                       epochs=10, units=32, verbose=0):
    """
    Run full comparison of all models and features.
    
    Args:
        X_dict: Dictionary of features {feature_name: X_matrix}
        y: Labels
        model_types: List of model types to test
        epochs: Number of training epochs
        units: Number of RNN units
        verbose: Verbosity level
    
    Returns:
        Results dictionary
    """
    results = []
    
    for feature_name, X in X_dict.items():
        print(f"\nTesting feature: {feature_name}")
        
        # Ensure X has correct shape for RNN
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        input_shape = (X.shape[1], X.shape[2])
        
        for model_type in model_types:
            print(f"  Training {model_type}...")
            
            # Build model
            if model_type == 'RNN':
                model = build_simple_rnn(input_shape, units=units)
            elif model_type == 'BiRNN':
                model = build_bidirectional_rnn(input_shape, units=units)
            elif model_type == 'LSTM':
                model = build_lstm(input_shape, units=units)
            else:
                continue
            
            # Train and evaluate
            metrics = train_and_evaluate(
                model, X, y, 
                epochs=epochs, 
                batch_size=32,
                verbose=verbose
            )
            
            # Store results
            results.append({
                'Feature': feature_name,
                'Model': model_type,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
    
    return results
