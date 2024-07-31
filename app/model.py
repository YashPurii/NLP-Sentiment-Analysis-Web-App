import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np

# Add the absolute path to the parent directory of 'app' to sys.path
sys.path.append(r'C:\Users\yashp\OneDrive\Desktop\sentiment-analysis')
from app.preprocessing import load_data, preprocess_data

def create_model(input_length):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(file_path):
    data = load_data(file_path)
    X, y, tokenizer, label_encoder = preprocess_data(data)
    
    # Ensure X and y are NumPy arrays with correct types
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_model(X.shape[1])
    model.fit(X_train, y_train, epochs=7, validation_data=(X_val, y_val), batch_size=32)
    
    model.save('sentiment_model.h5')
    return tokenizer, label_encoder
