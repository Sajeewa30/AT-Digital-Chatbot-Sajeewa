import json
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# 1. Load Hugging Face model for embeddings

@st.cache_resource 
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def get_embedding(text: str):
    return model.encode(text).tolist()


# 2. Load & preprocess dataset

with open("atdigital_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

pages = [item["text"] for item in dataset]
page_titles = [item["page"] for item in dataset]
page_urls = [item["url"] for item in dataset]


# 3. Create embeddings for dataset

@st.cache_data  
def compute_embeddings(pages):
    return np.array([get_embedding(text) for text in pages])

embeddings = compute_embeddings(pages)


# 4. Search function

def search_dataset(query, top_k=1):
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "page": page_titles[idx],
            "url": page_urls[idx],
            "text": pages[idx],
            "score": similarities[idx]
        })
    return results

# 5. Streamlit Web Interface

def run_web():
    st.title("🤖 AT Digital Chatbot")
    st.write("Ask me anything about AT Digital's services, team, or contact info.")

    user_query = st.text_input("Your question:")
    if user_query:
        results = search_dataset(user_query)
        best = results[0]
        st.markdown(f"**Answer:** {best['text']}")
        st.markdown(f"[Source]({best['url']})")

run_web()
