import os
import sys

# Add the absolute path to the parent directory of 'app' to sys.path
sys.path.append(r'C:\Users\yashp\OneDrive\Desktop\sentiment-analysis')

# Import functions from preprocessing and model scripts
from app.preprocessing import load_data, preprocess_data
from app.model import train_model

def main():
    # Path to the Tweets.csv file
    file_path = r'C:\Users\yashp\OneDrive\Desktop\sentiment-analysis\data\Tweets.csv'  # Adjust the path if necessary
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Train the model
    tokenizer, label_encoder = train_model(file_path)
    
    print("Training completed. Model saved as 'sentiment_model.h5'.")

if __name__ == "__main__":
    main()

