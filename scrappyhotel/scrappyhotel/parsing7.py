import re
import pandas as pd
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gerekli NLTK paketlerini indiriyoruz
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Birleşik kelime bölme için wordninja (kurulu değilse işlev pas geçer)
try:
    import wordninja
except ImportError:
    wordninja = None

# İngilizce stopword listesi ve lemmatizer
english_stops = set(stopwords.words("english"))
eng_lemmatizer = WordNetLemmatizer()

extra_stopwords_eng = {
    "ive", "hadnt", "km", "going", "u", "im", "höt", "year", "old", "minutes",
    "18mpbs", "us", "lets", "come", "theres", "min", "ms", "mr", "yes", "11pm",
    "2am", "10pm", "id", "etc", "motors", "tl", "3rd", "its"
}
english_stops.update(extra_stopwords_eng)


def clean_english_text(text):
    """Metni küçük harfe çevirir, boşlukları trimler ve noktalama işaretlerini kaldırır."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def process_english_text(text):
    """Temizleme, tokenizasyon, stopword çıkarma ve lemmatization uygular."""
    text = clean_english_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in english_stops]
    tokens = [eng_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def full_process_review_nlp(review: str) -> str:
    if not isinstance(review, str):
        return ""
    try:
        lang = detect(review)
    except Exception:
        lang = "en"
    # Varsayalım ki processed_review sütunundaki metinler zaten İngilizce
    if lang != "en":
        review = clean_english_text(review)
    processed = process_english_text(review)
    return processed


def main():
    # Giriş dosyası: Google Translate API ile çevrilmiş metinlerin bulunduğu CSV
    input_file = "final_processed_reviews_translate_deepl.csv"
    # Çıkış dosyası: NLP işlemleri uygulanmış final metinler "final_review" sütununda
    output_file = "final_processed_reviews_nlp_deepl.csv"

    df = pd.read_csv(input_file, engine="python")
    if "processed_review" not in df.columns:
        print("CSV'de 'processed_review' sütunu bulunamadı!")
        return
    # İşlenecek metin, translate işleminden sonra oluşan processed_review sütunundan alınacak
    df["final_review"] = df["processed_review"].apply(full_process_review_nlp)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"İşlem tamamlandı. Çıktı dosya: {output_file}")


if __name__ == "__main__":
    main()
