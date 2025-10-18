import re
import string
import nltk
import pandas as pd
import os
import glob
import json

# Kagglehub modülü yardımıyla Kaggle’dan Türkçe stopword listesini indirelim
try:
    import kagglehub

    kaggle_path = kagglehub.dataset_download("nezahatkk/tr-trke-stopwords-turkish")
    print("Path to dataset files:", kaggle_path)

    # Klasördeki tüm dosyaları görelim (dosya adını tespit etmek için)
    all_files = os.listdir(kaggle_path)
    print("Files in the Kaggle dataset directory:", all_files)

    # İlk önce 'stopwords.txt' arıyoruz, yoksa JSON dosyası deniyoruz
    kaggle_turkish_stopwords = set()
    txt_files = [f for f in all_files if f.lower().endswith('.txt')]
    if txt_files:
        stopword_file_path = os.path.join(kaggle_path, txt_files[0])
        print(f"Kullanılacak stopword dosyası: {stopword_file_path}")
        with open(stopword_file_path, encoding='utf-8') as f:
            kaggle_turkish_stopwords = {line.strip() for line in f if line.strip()}
    else:
        json_files = glob.glob(os.path.join(kaggle_path, "*.json"))
        if json_files:
            with open(json_files[0], encoding="utf-8") as f:
                additional = json.load(f)
                if isinstance(additional, list):
                    kaggle_turkish_stopwords = set(additional)
                elif isinstance(additional, dict):
                    kaggle_turkish_stopwords = set(additional.get("stopwords", []))
                else:
                    kaggle_turkish_stopwords = set()
            print("Loaded Turkish stopwords from JSON:", json_files[0])
        else:
            print("Kaggle stopwords dosyası bulunamadı, sadece NLTK stopwords kullanılacak.")
            kaggle_turkish_stopwords = set()
except ImportError:
    print("kagglehub modülü bulunamadı. 'pip install kagglehub' ile kurabilirsiniz veya kodu bu kısımdan silin.")
    kaggle_turkish_stopwords = set()

# -- NLTK ve gerekli paketler -------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -- Stopword listeleri -------------------------------------------
english_stopwords = set(stopwords.words('english'))
nltk_turkish = set(stopwords.words('turkish'))
# NLTK + Kaggle'dan gelen Türkçe stopword'leri birleştiriyoruz
turkish_stopwords = nltk_turkish.union(kaggle_turkish_stopwords)

# İhtiyaca göre ekstra Türkçe stopword eklemeleri (örneğin "sebeple" eklenmiştir)
turkish_stopwords.update({"gibi", "artık", "keşke", "sebeple"})

print(f"Kaggle stopwords sayısı: {len(kaggle_turkish_stopwords)}")
print(f"Toplam Türkçe stopwords (NLTK + Kaggle): {len(turkish_stopwords)}")

# -- İngilizce için lemmatizer ------------------------------------
eng_lemmatizer = WordNetLemmatizer()

# -- Türkçe Stemmer -----------------------------------------------
try:
    from TurkishStemmer import TurkishStemmer

    turkish_stemmer = TurkishStemmer()
except ImportError:
    print("TurkishStemmer kütüphanesi yüklü değil. 'pip install TurkishStemmer' ile kurabilirsiniz.")
    turkish_stemmer = None


def detect_language(text: str) -> str:
    """
    Basit dil tespiti: Metindeki Türkçe karakterlere göre 'turkish' veya 'english' döndürür.
    """
    if any(char in text for char in "şğüöıÇŞĞÜÖİ"):
        return 'turkish'
    return 'english'


def basic_cleaning(text: str) -> str:
    """
    Metni küçük harfe çevirir, noktalama işaretlerini kaldırır, fazla boşlukları temizler.
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def process_text(text: str, language: str = 'english') -> str:
    """
    Metni token'lara ayırır, sayı içeren tokenları çıkarır,
    İngilizce ise lemmatization + stopword,
    Türkçe ise (varsa) stemming + stopword (NLTK + Kaggle) uygular.
    """
    text = basic_cleaning(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if not any(char.isdigit() for char in word)]

    if language == 'english':
        filtered = []
        for w in tokens:
            if w not in english_stopwords:
                lemma = eng_lemmatizer.lemmatize(w)
                filtered.append(lemma)
        tokens = filtered
    elif language == 'turkish':
        filtered = []
        for w in tokens:
            if w not in turkish_stopwords:
                if turkish_stemmer is not None:
                    w = turkish_stemmer.stem(w)
                filtered.append(w)
        tokens = filtered
    else:
        tokens = [w for w in tokens if w not in turkish_stopwords]

    return " ".join(tokens)


def preprocess_pipeline(text) -> str:
    """
    Metnin dilini tespit eder, eğer geçerli bir string değilse boş string döndürür,
    ardından işleme adımlarını uygular.
    """
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    lang = detect_language(text)
    return process_text(text, language=lang)


if __name__ == "__main__":
    input_file = "cleaned_reviews_output2.csv"  # Kaynak CSV dosyası
    output_file = "processed_reviews_turkemm5.csv"

    try:
        df = pd.read_csv(input_file)
        if 'review_text' not in df.columns:
            print("CSV dosyasında 'review_text' sütunu bulunamadı.")
        else:
            df['processed_review'] = df['review_text'].apply(preprocess_pipeline)
            df.to_csv(output_file, index=False)
            print(f"\nİşlenmiş veriler '{output_file}' dosyasına kaydedildi.\n")
            print(df[['review_text', 'processed_review']].head())
    except Exception as e:
        print("CSV dosyası okunurken veya yazılırken bir hata oluştu:", e)
