from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pandas as pd

# Initialize MongoDB client and database
client = MongoClient("mongodb://localhost:27017/")
db = client['rag_project']
collection = db['documents']

# Load the dataset from CSV
def load_bbc_data(file_path):
    df = pd.read_csv(file_path,sep='\t')
    return df

# Generate embeddings and store in MongoDB
def process_and_store(dataframe):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight transformer model
    
    for idx, row in dataframe.iterrows():
        text = row['content']
        embedding = model.encode(text).tolist()
        doc = {
            "id": int(row['id']) if 'id' in row else idx,
            "title": row.get('title', f"Document {idx}"),
            "text": text,
            "embedding": embedding
        }
        collection.insert_one(doc)

# Run preprocessing
if __name__ == "__main__":
    csv_file_path = "bbc-news-data.csv"  # Replace with the path to your CSV file
    bbc_data = load_bbc_data(csv_file_path)
    process_and_store(bbc_data)
