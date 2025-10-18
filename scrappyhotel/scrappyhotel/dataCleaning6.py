import re
import string
import os
import glob
import json
import pandas as pd
import nltk
from langdetect import detect
from nltk.corpus import stopwords


# Gerekli NLTK paketlerini indiriyoruz (zaten varsa hata vermez)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Stopword listeleri: İngilizce ve Türkçe (varsayılan NLTK stopword'leri)
english_stops = set(stopwords.words("english"))
turkish_stops = set(stopwords.words("turkish"))

# Kaggle üzerinden Türkçe stopword listesini indiriyoruz
import kagglehub

path = kagglehub.dataset_download("nezahatkk/tr-trke-stopwords-turkish")
print("Path to dataset files:", path)
dataset_files = glob.glob(os.path.join(path, "*"))
print("Dataset files found:", dataset_files)

# İlk önce stopwords.txt arıyoruz, yoksa JSON dosyasını deniyoruz
stopwords_file = os.path.join(path, "stopwords.txt")
if os.path.exists(stopwords_file):
    with open(stopwords_file, encoding="utf-8") as f:
        additional_turkish_stops = {line.strip() for line in f if line.strip()}
    turkish_stops.update(additional_turkish_stops)
    print("Loaded Turkish stopwords from stopwords.txt")
else:
    json_files = glob.glob(os.path.join(path, "*.json"))
    if json_files:
        with open(json_files[0], encoding="utf-8") as f:
            additional = json.load(f)
            if isinstance(additional, list):
                additional_turkish_stops = set(additional)
            elif isinstance(additional, dict):
                additional_turkish_stops = set(additional.get("stopwords", []))
            else:
                additional_turkish_stops = set()
            turkish_stops.update(additional_turkish_stops)
        print("Loaded Turkish stopwords from JSON:", json_files[0])
    else:
        print("No stopwords file found in dataset directory.")


# Ekstra Türkçe stopword eklemeleri (Örneğin, "gibi", "artık", "keşke" ekleniyor)
extra_stopwords = {"gibi", "artık", "keşke", "idi", "yıl", "yıllardır", "yıldır", "ki", "tl"}
turkish_stops.update(extra_stopwords)

# Önemli kelimelerin silinmesini engellemek için stopword listesinden çıkarıyoruz:
for word in ["konum", "manzara", "fiyat", "performans"]:
    turkish_stops.discard(word)

print(f"Toplam Türkçe stopword sayısı: {len(turkish_stops)}")
print(f"Toplam İngilizce stopword sayısı: {len(english_stops)}")


########################################
# Temizleme Fonksiyonları
########################################

def remove_emojis(text: str) -> str:
    """Metindeki emojileri kaldırır."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticonlar
        u"\U0001F300-\U0001F5FF"  # Semboller ve piktogramlar
        u"\U0001F680-\U0001F6FF"  # Ulaşım & harita sembolleri
        u"\U0001F1E0-\U0001F1FF"  # Bayraklar
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


def remove_urls(text: str) -> str:
    """Metindeki URL'leri kaldırır."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_numbers(text: str) -> str:
    """Metindeki rakamları kaldırır."""
    return re.sub(r"\b\d+\b", "", text)


def remove_punctuation(text: str) -> str:
    """
    Metindeki tüm noktalama işaretlerini kaldırır (alt çizgi dahil).
    """
    return re.sub("[" + re.escape(string.punctuation) + "]", "", text)


def minimal_cleaning(text: str) -> str:
    """
    Metni tamamen temizler:
      - Emojileri, URL'leri, rakamları ve tüm noktalama işaretlerini (alt çizgi dahil) kaldırır.
      - Küçük harfe çevirir ve fazla boşlukları teke indirir.
      - Dil tespiti yapar ve ilgili dilin stopword listesini kullanarak metindeki stopword'leri çıkarır.
    """
    if not isinstance(text, str):
        return ""

    # Temel temizleme: Emoji, URL, rakam ve noktalama kaldırma
    text = remove_emojis(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)

    # Küçük harfe çevirme ve ekstra boşlukları temizleme
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    # Dil tespiti; hata alınırsa varsayılan "en" kabul edilir
    try:
        lang = detect(text)
    except Exception:
        lang = "en"

    words = text.split()
    if lang == "tr":
        cleaned_words = [w for w in words if w not in turkish_stops]
    else:
        cleaned_words = [w for w in words if w not in english_stops]

    return " ".join(cleaned_words)


########################################
# Ana Fonksiyon: Dosya Okuma, Temizleme, Yazma
########################################

def process_reviews(input_file: str, output_file: str):
    # CSV dosyasını oku; engine belirterek format sorunu yaşamayın
    df = pd.read_csv(input_file, engine='python')

    # Gerekli sütunları kontrol et
    required_cols = {"hotel_name", "author_name", "review_text"}
    if not required_cols.issubset(df.columns):
        print(f"CSV dosyasında eksik sütunlar: {required_cols - set(df.columns)}")
        return

    cleaned_data = []
    for idx, row in df.iterrows():
        hotel = str(row["hotel_name"])
        author = str(row["author_name"])
        review_raw = str(row["review_text"])

        cleaned_review = minimal_cleaning(review_raw)

        cleaned_data.append({
            "hotel_name": hotel,
            "author_name": author,
            "review_text": cleaned_review
        })

    new_df = pd.DataFrame(cleaned_data)
    new_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Temizlenmiş yorumlar '{output_file}' dosyasına kaydedildi.")


if __name__ == "__main__":
    # Girdi dosyanız CSV formatında olsun
    input_file = "cleaned_reviews_output2.csv"  # Örneğin; daha önce temizlenmemiş yorumların bulunduğu CSV
    output_file = "cleaned_reviews_output3.csv"  # Çıktı dosyası
    process_reviews(input_file, output_file)
