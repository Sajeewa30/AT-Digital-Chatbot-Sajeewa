# chatbot_atdigital.py
import json
import openai
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Load your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# -----------------------------
# 1. Load & preprocess dataset
# -----------------------------
with open("atdigital_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

pages = [item["text"] for item in dataset]
page_titles = [item["page"] for item in dataset]
page_urls = [item["url"] for item in dataset]

# -----------------------------
# 2. Create embeddings for dataset
# -----------------------------
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

embeddings = [get_embedding(text) for text in pages]
embeddings = np.array(embeddings)

# -----------------------------
# 3. Search function
# -----------------------------
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

# -----------------------------
# 4. CLI mode
# -----------------------------
def run_cli():
    print("AT Digital Chatbot (type 'quit' to exit)")
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            break
        results = search_dataset(query)
        best = results[0]
        print(f"Bot: {best['text']} (Source: {best['url']})\n")

# -----------------------------
# 5. Streamlit Web Interface
# -----------------------------
def run_web():
    st.title("🤖 AT Digital Chatbot")
    st.write("Ask me anything about AT Digital's services, team, or contact info.")

    user_query = st.text_input("Your question:")
    if user_query:
        results = search_dataset(user_query)
        best = results[0]
        st.markdown(f"**Answer:** {best['text']}")
        st.markdown(f"[Source]({best['url']})")

# Uncomment one of these to run
# run_cli()
run_web()