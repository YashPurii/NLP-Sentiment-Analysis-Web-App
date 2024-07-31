import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data[['text', 'sentiment']]
    return data

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#\w+', '', text)  # remove hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # remove non-alphabetic characters
    text = text.lower()  # convert to lowercase
    text = text.split()
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

def preprocess_data(data):
    # Drop rows with NaN values in 'text' column
    data = data.dropna(subset=['text'])
    
    # Clean text data
    data.loc[:, 'text'] = data['text'].apply(clean_text)
    
    # Encode sentiment labels
    le = LabelEncoder()
    data.loc[:, 'sentiment'] = le.fit_transform(data['sentiment'])
    
    # Tokenize text data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    return padded_sequences, data['sentiment'], tokenizer, le
