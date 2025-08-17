# 🤖 AT Digital Chatbot

A simple **question-answering chatbot** built with **Streamlit** and **Hugging Face Sentence Transformers**.  
It allows users to ask questions about AT Digital's website content (services, team, contact info, etc.), and retrieves the most relevant answer from a dataset.

---

## ✨ Features
- Uses **Hugging Face embeddings** (`all-MiniLM-L6-v2`) — ✅ no API key or quota limits.
- Search powered by **cosine similarity**.
- Two modes:
  - **Web App (Streamlit UI)**
  - **CLI mode (Command line chatbot)**
- Returns both **answer text** and **source link**.

---

## 🛠️ Tech Stack
- [Python 3.9+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [scikit-learn](https://scikit-learn.org/)

---

## 📂 Project Structure
At_Digital_Chatbot/
│── at_digital_chatbot.py # Main chatbot script
│── atdigital_dataset.json # Dataset (website pages & content)
│── requirements.txt # Dependencies
│── README.md # Documentation
└── .streamlit/ # (Optional) Streamlit config


---

## ⚡ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/At_Digital_Chatbot.git
   cd At_Digital_Chatbot

2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt

