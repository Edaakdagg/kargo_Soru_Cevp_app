# pandas kütüphanesini içe aktardım, CSV okumak için kullanacağım
import pandas as pd

# sentence-transformers'dan model yüklemek için gerekli sınıfı aldım
from sentence_transformers import SentenceTransformer

# CSV dosyamı okudum ve veri çerçevesi (DataFrame) oluşturdum
df = pd.read_csv("kargo_data.csv")

# Veri çerçevesinden sadece "question" sütununu liste haline getirdim
questions = df["question"].tolist()

# Çok dilli, küçük ve hızlı bir model seçtim
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Soruların her birini vektöre (embedding) dönüştürdüm
question_embeddings = model.encode(questions)

# Kontrol etmek için ilk soruyu ve onun embedding vektörünün ilk 5 elemanını yazdırdım
print("İlk soru:", questions[0])
print("İlk vektörün ilk 5 elemanı:", question_embeddings[0][:5])

