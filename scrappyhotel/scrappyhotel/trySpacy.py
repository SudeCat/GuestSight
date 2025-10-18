import pandas as pd
from langdetect import detect
import spacy

# SpaCy'nin Hugging Face'den indirilen Türkçe transformer modeli yükleniyor
nlp = spacy.load("tr_core_news_trf")

# CSV dosya yolu (dizin ve dosya adını kontrol edin)
csv_path = r"C:\Users\catsu\PycharmProjects\scrappyhotel\final_processed_reviews_no_spellcheck.csv"

# CSV dosyasını oku
df = pd.read_csv(csv_path, engine="python")

# Yorum sütununu tespit et (processed_review veya processed_reviews)
if "processed_review" in df.columns:
    review_column = "processed_review"
elif "processed_reviews" in df.columns:
    review_column = "processed_reviews"
else:
    raise ValueError("CSV'de 'processed_review' veya 'processed_reviews' sütunu bulunamadı!")

def lemmatize_review(review):
    """
    Verilen yorumun dilini tespit eder; eğer yorum Türkçe ise,
    spaCy modelini kullanarak her token'ın lemma'sını alır ve
    lemmatized metni döndürür. Türkçe değilse orijinal yorumu döndürür.
    """
    try:
        lang = detect(review)
    except Exception:
        lang = ""
    if lang == "tr":
        doc = nlp(review)
        # Her token için lemmatization yap
        lemmatized_tokens = [token.lemma_ for token in doc]
        return " ".join(lemmatized_tokens)
    else:
        return review

# Her yoruma lemmatization işlemini uygula
df["lemmatized_review"] = df[review_column].apply(lemmatize_review)

# Sonuçları yeni bir CSV dosyasına kaydet
output_path = r"C:\Users\catsu\PycharmProjects\scrappyhotel\final_lemmatized_reviews.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Lemmatization tamamlandı. Sonuç dosyası:", output_path)
