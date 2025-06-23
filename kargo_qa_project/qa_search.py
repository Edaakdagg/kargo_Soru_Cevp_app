# pandas: CSV dosyasını okuyacağım
import pandas as pd

# sentence-transformers: daha önce kullandığım modelle embedding yapacağım
from sentence_transformers import SentenceTransformer

# faiss: benzer soruyu bulmak için vektör arama kütüphanesi
import faiss

# numpy: vektörleri faiss ile uyumlu hale getirmek için kullanıyorum
import numpy as np

# Kullanıcıdan gelen soruyu alıyorum
user_question = input("Lütfen sorunuzu yazın: ")

# CSV dosyasındaki soru-cevap verisini okuyorum
df = pd.read_csv("kargo_data.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# Daha önce kullandığım modeli tekrar yüklüyorum
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Veri kümesindeki tüm soruları embedding'e dönüştürüyorum
question_embeddings = model.encode(questions)

# Kullanıcının sorusunu da embedding'e dönüştürüyorum
user_embedding = model.encode([user_question])

# FAISS index oluşturuyorum (L2 uzaklığına göre arama yapacak)
dimension = user_embedding.shape[1]
index = faiss.IndexFlatL2(dimension)

# Elimdeki tüm soru embedding'lerini FAISS index'ine ekliyorum
index.add(np.array(question_embeddings))

# Kullanıcının embedding'i ile en yakın sonucu arıyorum (k=1: en yakın 1 sonucu getir)
_, result_index = index.search(np.array(user_embedding), k=1)

# Bulunan en yakın sorunun cevabını alıyorum
closest_answer = answers[result_index[0][0]]

# Cevabı yazdırıyorum
print("\nEn uygun cevap:")
print(closest_answer)

