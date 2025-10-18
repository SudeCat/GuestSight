import re
import pandas as pd
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import DeeplTranslator

# Gerekli NLTK paketlerini indiriyoruz
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# API anahtarınızı buraya girin (DeepL API Free için)
DEEPL_API_KEY = "5c31fa7f-f538-4f62-850d-4895cc99e5db:fx"  # Kendi API anahtarınızı buraya yazın

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

def translate_to_english(text):
    """
    Metnin dilini tespit eder; eğer İngilizce değilse DeepL API free kullanarak
    Türkçe'den İngilizce'ye çevirir. Çeviri sırasında hata oluşursa orijinal metni döndürür.
    """
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "en"
    if detected_lang != "en":
        try:
            # Dil kodlarını küçük harfle kullanıyoruz: source="tr", target="en"
            translated = DeeplTranslator(api_key=DEEPL_API_KEY, source="tr", target="en").translate(text)
            return translated
        except Exception as e:
            print(f"Translate error: {e} | Metin: {text}")
            return text
    else:
        return text

def full_process_review_translate(review: str) -> str:
    if not isinstance(review, str):
        return ""
    # Eğer yorum İngilizce değilse, DeepL API ile çevirisini uygula
    review_en = translate_to_english(review)
    # Ardından İngilizce metin üzerinde NLP işlemlerini uygula
    processed = process_english_text(review_en)
    return processed

def main():
    input_file = "cleaned_reviews_output3.csv"  # Giriş CSV dosyası (review_text sütunu bulunmalı)
    output_file = "final_processed_reviews_translate_deepl.csv"

    df = pd.read_csv(input_file, engine="python")
    if "review_text" not in df.columns:
        print("CSV'de 'review_text' sütunu bulunamadı!")
        return
    df["review_text"] = df["review_text"].fillna("")

    # Her yorumu önce çevirip ardından NLP işlemlerini uygulayarak işliyoruz
    df["processed_review"] = df["review_text"].apply(full_process_review_translate)

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"İşlem tamamlandı. Çıktı dosya: {output_file}")

if __name__ == "__main__":
    main()
