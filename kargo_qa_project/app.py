# app.py
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# Başlık
st.title("📦 Kargo Soru-Cevap Sistemi")

# Modeli yükle
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# CSV'den veriyi oku
df = pd.read_csv("kargo_data.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Embedding'leri üret
question_embeddings = model.encode(questions)

# FAISS index oluştur
index = faiss.IndexFlatL2(len(question_embeddings[0]))
index.add(np.array(question_embeddings))

# Kullanıcıdan giriş al
user_question = st.text_input("Sorunuzu buraya yazın:")

if user_question:
    # Kullanıcı sorusunu encode et
    user_embedding = model.encode([user_question])
    
    # En yakın soruyu bul (top 1)
    D, I = index.search(np.array(user_embedding), k=1)
    idx = I[0][0]
    
    # Sonucu göster
    st.subheader("📌 Cevap")
    st.write(answers[idx])

