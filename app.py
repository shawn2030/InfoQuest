from flask import Flask, request, jsonify, render_template
import numpy as np
import faiss
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import os

# Since i am working on cpu i assigned only 1 thread to all the libraries
# Uncomment the below lines if you are working on gpu
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


app = Flask(__name__)

# MongoDB and FAISS Setup
client = MongoClient("mongodb://localhost:27017/")
db = client['rag_project']
collection = db['documents']

def load_embeddings():
    documents = list(collection.find({}, {"embedding": 1, "id": 1}))
    embeddings = np.array([doc['embedding'] for doc in documents]).astype(np.float32)
    ids = np.array([doc['id'] for doc in documents])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, ids

faiss_index, doc_ids = load_embeddings()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")  
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
print("Model loaded successfully.")
model.eval()

def get_llm_answer(results, query):
    context = "\n\n".join([doc['text'] for doc in results])
    prompt = f"Answer the following question based on the given context:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    #tokeinze the prompt
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    with torch.no_grad():
        output = model.generate(
                                inputs['input_ids'], 
                                attention_mask=inputs['attention_mask'],  # Include attention mask
                                max_new_tokens=150, 
                                num_return_sequences=1, 
                                no_repeat_ngram_size=2, 
                                top_k=50, 
                                top_p=0.95,
                                do_sample=True
                            )
    answer = tokenizer.decode(output[0], skip_special_tokens = True)
    answer = answer.split("Answer:")[-1].strip()

    return answer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form.get('query')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query_text).astype(np.float32)

    # search in faiss index
    distance, indices = faiss_index.search(np.array([query_embedding]), k=5)
    print("FAISS query successful.")

    # retrieve relevant documents
    results = []
    for idx in indices[0]:
        document = collection.find_one({"id": int(doc_ids[idx])}, {"_id": 0, "title": 1, "text": 1})
        if document:
            results.append(document)

    answer  = get_llm_answer(results, query_text)

    return render_template('index.html', query=query_text, results = results, answer=answer)

                       
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=7090, debug=True)