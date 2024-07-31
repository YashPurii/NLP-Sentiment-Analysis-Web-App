from flask import Flask, request, jsonify, render_template
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import sys
import os

# Add the absolute path to the parent directory of 'app' to sys.path
sys.path.append(r'C:\Users\yashp\OneDrive\Desktop\sentiment-analysis')
from app.preprocessing import load_data, preprocess_data  # Import custom functions for data handling
from app.model import train_model

app = Flask(__name__)

# Load the model, tokenizer, and label encoder
model = load_model('sentiment_model.h5')
tokenizer, label_encoder = None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    prediction = model.predict(padded_sequence)
    sentiment = label_encoder.inverse_transform([int(prediction[0] > 0.5)])[0]
    
    return jsonify({'sentiment': sentiment})

if __name__ == "__main__":
    data_file_path = 'data/Tweets.csv'  # Replace with your data file path
    tokenizer, label_encoder = train_model(data_file_path)
    app.run(debug=True)
