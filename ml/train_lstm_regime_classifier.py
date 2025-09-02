#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-based Market Regime Classifier Training

This script trains an LSTM neural network to classify market regimes based on
sequences of technical indicators. The LSTM architecture is well-suited for
financial time series data as it can capture temporal dependencies and patterns.

Benefits over traditional ML models:
1. Processes data sequentially, maintaining the time-series nature
2. Captures long-term dependencies in the data
3. Can identify complex market regime transition patterns
4. Works with variable-length input sequences
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from trading_bot.strategies.strategy_template import MarketRegime
from ml.train_regime_classifier import create_features, label_regimes, load_and_prepare_data

# Configuration
OUTPUT_DIR = os.path.join(project_root, 'ml/models')
MODEL_FILENAME = 'forex_lstm_regime_classifier.h5'
RANDOM_SEED = 42
SEQUENCE_LENGTH = 10  # Number of time steps to look back
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 50
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def prepare_lstm_data(df, sequence_length=SEQUENCE_LENGTH):
    """
    Prepare data for LSTM training by creating sequences and labels.
    
    Args:
        df: DataFrame with features and target
        sequence_length: Number of time steps to include in each sequence
        
    Returns:
        X: Sequence data with shape (n_samples, sequence_length, n_features)
        y: Target values
    """
    # Extract features and target
    X_columns = [col for col in df.columns if col != 'regime']
    features = df[X_columns].values
    target = df['regime'].values
    
    # Convert target to one-hot encoding
    # First determine number of classes
    n_classes = len(np.unique(target))
    
    # Create sequences
    X = []
    y = []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i, :])
        y.append(target[i])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Convert to categorical (one-hot encoding)
    y_categorical = to_categorical(y - 1, num_classes=n_classes)  # Subtract 1 because regime values start at 1
    
    return X, y_categorical

def create_lstm_model(input_shape, n_classes):
    """
    Create an LSTM model for market regime classification.
    
    Args:
        input_shape: Shape of input sequences (sequence_length, n_features)
        n_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # LSTM layers
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        LSTM(LSTM_UNITS * 2, return_sequences=True),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        LSTM(LSTM_UNITS, return_sequences=False),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        
        # Output layer
        Dense(n_classes, activation='softmax')
    ])
    
    # Compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_lstm_model(model, X_test, y_test, class_names):
    """
    Evaluate the LSTM model and display metrics.
    
    Args:
        model: Trained LSTM model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        class_names: List of class names
    """
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1) + 1  # Add 1 to convert back to original class indices
    y_true = np.argmax(y_test, axis=1) + 1  # Add 1 to convert back to original class indices
    
    # Print accuracy metrics
    print("\nLSTM Model Evaluation:")
    print("======================")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title('LSTM Confusion Matrix')
    plt.ylabel('True Regime')
    plt.xlabel('Predicted Regime')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_confusion_matrix.png'))
    
    # Create training history visualization if model has history attribute
    if hasattr(model, 'history') and model.history is not None:
        history = model.history.history
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'lstm_training_history.png'))
    
    print("\nEvaluation plots saved to:", OUTPUT_DIR)

def save_model_with_metadata(model, feature_columns, scaler):
    """
    Save the model along with metadata for future use.
    
    Args:
        model: Trained LSTM model
        feature_columns: List of feature column names
        scaler: Fitted scaler for feature normalization
    """
    # Save keras model
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    model.save(model_path)
    
    # Save scaler and feature columns separately
    scaler_path = os.path.join(OUTPUT_DIR, 'lstm_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save feature columns as JSON
    feature_columns_path = os.path.join(OUTPUT_DIR, 'lstm_feature_columns.json')
    with open(feature_columns_path, 'w') as f:
        json.dump(feature_columns, f)
    
    # Save model metadata
    metadata_path = os.path.join(OUTPUT_DIR, 'lstm_model_metadata.json')
    metadata = {
        'model_type': 'LSTM',
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': len(feature_columns),
        'lstm_units': LSTM_UNITS,
        'dropout_rate': DROPOUT_RATE,
        'training_date': datetime.now().strftime('%Y-%m-%d'),
        'random_seed': RANDOM_SEED
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"Model and metadata saved to {OUTPUT_DIR}")

def main():
    """Main execution function."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting LSTM market regime classifier training...")
    
    # Load and prepare data
    print("Loading data...")
    data = load_and_prepare_data()
    
    # Create features
    print("Extracting technical features...")
    features_df = create_features(data)
    
    # Label the data
    print("Labeling market regimes...")
    labeled_df = label_regimes(features_df)
    
    # Prepare features and target for LSTM
    feature_columns = [col for col in labeled_df.columns if col not in 
                      ['datetime', 'open', 'high', 'low', 'close', 'volume', 
                       'regime', 'true_regime', 'higher_high', 'lower_low']]
    
    # Standardize features
    print("Scaling features...")
    scaler = StandardScaler()
    labeled_df[feature_columns] = scaler.fit_transform(labeled_df[feature_columns])
    
    # Create sequences for LSTM
    print(f"Creating sequences with length {SEQUENCE_LENGTH}...")
    X, y = prepare_lstm_data(labeled_df, sequence_length=SEQUENCE_LENGTH)
    
    # Print data shape
    print(f"Data shape: X = {X.shape}, y = {y.shape}")
    
    # Split data
    print("Splitting data into train/validation/test sets...")
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=np.argmax(y, axis=1)
    )
    
    # Then split train+val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VALIDATION_SIZE/(1-TEST_SIZE),  # Adjust for the first split
        random_state=RANDOM_SEED,
        stratify=np.argmax(y_train_val, axis=1)
    )
    
    print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples, "
          f"testing with {len(X_test)} samples")
    
    # Create the model
    input_shape = (SEQUENCE_LENGTH, X.shape[2])
    n_classes = y.shape[1]
    
    print(f"Creating LSTM model with input shape {input_shape} and {n_classes} output classes...")
    model = create_lstm_model(input_shape, n_classes)
    
    # Show model summary
    model.summary()
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'lstm_best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Train the model
    print(f"Training LSTM model for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Store history in the model for visualization
    model.history = history
    
    # Get class names for evaluation
    class_names = [regime.name for regime in MarketRegime]
    
    # Evaluate the model
    evaluate_lstm_model(model, X_test, y_test, class_names)
    
    # Save the model and metadata
    save_model_with_metadata(model, feature_columns, scaler)
    
    print("LSTM training completed successfully!")

if __name__ == "__main__":
    main()
