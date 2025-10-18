import re
import os
import glob
import json
import pandas as pd
import nltk
from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# (1) Gerekli NLTK paketlerini indiriyoruz
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# wordninja, birleşik kelime bölme için (kurulu değilse işlev pas geçer)
try:
    import wordninja
except ImportError:
    wordninja = None

# (2) Stopword listeleri: İngilizce ve Türkçe
english_stops = set(stopwords.words("english"))
turkish_stops = set(stopwords.words("turkish"))

# 2) Kaggle üzerinden ek Türkçe stopword'leri ekleyelim
try:
    import kagglehub

    kaggle_path = kagglehub.dataset_download("nezahatkk/tr-trke-stopwords-turkish")
    print("Path to dataset files:", kaggle_path)
    dataset_files = glob.glob(os.path.join(kaggle_path, "*"))
    print("Dataset files found:", dataset_files)
    stopwords_file = os.path.join(kaggle_path, "stopwords.txt")
    if os.path.exists(stopwords_file):
        with open(stopwords_file, encoding="utf-8") as f:
            additional_turkish_stops = {line.strip() for line in f if line.strip()}
        turkish_stops.update(additional_turkish_stops)
        print("Loaded Turkish stopwords from stopwords.txt")
    else:
        json_files = glob.glob(os.path.join(kaggle_path, "*.json"))
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
            print("No stopwords file found in Kaggle dataset directory.")
except ImportError:
    print("kagglehub modülü bulunamadı. 'pip install kagglehub' ile kurabilirsiniz.")

# Ekstra stopword ekleyebilirsiniz (örnek):
extra_stopwords = {"artık", "keşke", "lı", "imiş", "birşeyi", "birde", "felan", "ben", "dakikalık", "falan", "dada",
                   "gorunca", "u", "e", "tabiki",
                   "yüzde", "idi", "yıl", "b", "yolarda", "istinaden", "bilhassa", "tmm", "nin", "an", "nolu", "tlden",
                   "3bin", "fıkra", "sinide", "yüzden", "km", "den", "kişilik", "tane", "azından", "kişi", "dahaki",
                   "edrr",
                   "miyiz", "mi", "olurda", "herşeyin", "üzerine", "dahakine", "kat", "m2", "lik", "bişeyler", "kadar",
                   "le", "bişey",
                   "sene", "ani", "bana", "numaralı", "kısacası", "yani", "anlamda", "yaklaşık", "buçuk", "senedir",
                   "de", "dan",
                   "nolu", "dakikada", "da", "yi","yüzden", "tane", "birşeye", "el", "hemde", "ha", "gözle", "şekilde", "ki",
                   "daki", "nın","m","gündür",
                   "mesela", "kimse", "adet", "bence", "ötürü", "katı", "teymiş", "mt", "ni", "e", "den", "an",
                   "lar", "l", "ta", "nedir", "daki", "numaralı", "dklık", "300m", "kglik",
                   "yeterki", "içinde", "dk", "tekarr", "biryer", "herşeyiyle", "lira", "liraya", "ayrıyetten",
                   "numara", "birde","tarihinde","10y", "4y",
                   "bi", "haberin", "ayriyetten", "şeyiyle", "başka", "bişey", "anda", "bende", "metre", "agodadan",
                   "agodada", "2kisilik", "1kisilik","ikiüç",
                   "nu", "rabbim", "lik", "kazandırdıkları", "napolyonun", "askerini", " muhtemelen", "kanlarıyla",
                   "şehitlerimizin", "15dk", "10dk", "5dk", "50dk", "2dk", "vb", "yaşındaki",
                   "aylik", "dışında", "yokmu", "sebepten", "vs", "haberiniz", "zaman", "şimdiden", "gün", "verilse",
                   "300tl", "dakika","ay","te","şuan","yada","bey"
                   "ye", "100tl", "bide", "bir", "şeyin", "yıllardır", "yıldır", "ki", "hergün", "", "tl"}
turkish_stops.update(extra_stopwords)

extra_stopwords_eng = {"ive", "hadnt", "km","going", "u", "im", "höt", "year", "old", "minutes", "18mpbs", "us", "lets",
                       "come", "theres","min","ms","mr"
                       "yes", "11pm", "2am", "10pm", "id", "etc", "motors", "tl", "3rd", "its"}

english_stops.update(extra_stopwords_eng)

# Bazı önemli kelimeleri stopword listesinden çıkarıyoruz (örnek):
for word in ["her", "var", "iyi", "kötü", "yok", "değil", "en", "sağolsun", "sağ", "olsun", "maalesef", "rağmen",
             "olmaz", "değmez"]:
    turkish_stops.discard(word)

print(f"Türkçe stopwords: {len(turkish_stops)} adet")
print(f"İngilizce stopwords: {len(english_stops)} adet")

# (3) İngilizce lemma için
eng_lemmatizer = WordNetLemmatizer()


# (4) TurkBERT ile Türkçe Morfolojik Analiz / Lemmatizasyon
# Zemberek yerine TurkBERT kullanılacak. Eğer entegrasyon sağlanamazsa, işlem yapılmayacak.
def process_turkish_tokens(tokens):
    # TODO: TurkBERT entegrasyonu eklenebilir.
    # Şu an için, mevcut morfolojik çekimleri düzeltemiyoruz.
    return tokens


########################################################################
# (5) DİAKRİTİK / YAZIM DÜZELTMELERİ
########################################################################
diacritic_map = {
    "dahada": "daha da",
    "avantajki": "avantaj",
    "fiyatda": "fiyat",
    "olmuycaktır": "olmayacak",
    "şölenn": "şölen",
    "edecğim": "ederim",
    "biyer": "bir yer",
    "malesef": "maalesef",
    "beyede": "bey",
    "merdivrn": "merdiven",
    "heryönüyle": "her yönüyle",
    "herşey": "her şey",
    "heryer": "her yer",
}


def apply_diacritic_corrections(tokens):
    corrected = []
    for t in tokens:
        if t in diacritic_map:
            corrected.append(diacritic_map[t])
        else:
            corrected.append(t)
    return corrected


# İngilizce için diakritik düzeltme örneği
english_diacritic_map = {
    "checkin": "check in",
    "superrrrrrrrr": "super",
    "checkout": "check out",
    "apt": "apartment",
    "veey": "very",
    "beautıful": "beautiful",
    "anil": "anıl",
    "oludeniz": "ölüdeniz",
    "turkishstyle": "turkish style",
    "wellequiped": "well equiped",
    "niceeee": "nice",
    "priceperformance": "price performance",
    "dont": "do not",
    "didnt": "did not",
    "havent": "have not",
    "couldnt": "could not",
    "wasnt": "was not",
    "cannot": "can not",
    "isnt": "is not",
}


def apply_english_diacritic_corrections(tokens):
    corrected = []
    for t in tokens:
        if t in english_diacritic_map:
            corrected.append(english_diacritic_map[t])
        else:
            corrected.append(t)
    return corrected


########################################################################
# (6) Protected İfadeler
########################################################################
protected_expressions = [
    "güler yüz",
    "hayal kırıklığı",
    "samimi davranan",
    "içten gülümseyen",
    "dostça davranan",
    "sıcakkanlı",
    "nezaketli",
    "ilgili",
    "fiyat performans",
    "canayakın"
]


def protect_expressions(text):
    # Regex ile "güler yüz" çekimlerini normalleştiriyoruz.
    text = re.sub(r"\bgüler[-\s]?yüz(lü|ü|leri|lüydü|üyle)?\b", "güler_yüz", text, flags=re.IGNORECASE)
    for expr in protected_expressions:
        pattern = re.compile(re.escape(expr), flags=re.IGNORECASE)
        text = pattern.sub(expr.replace(" ", "_"), text)
    return text


def restore_expressions(tokens):
    return [t.replace("_", " ") for t in tokens]


########################################################################
# (7) Custom Manuel Düzeltmeler (diakritik dışı)
########################################################################
custom_corrections = {
    "fiyatperformans": "fiyat performans",
    "güler yüzden": "güler yüz",
    "güler yüzlüydü": "güler yüz",
    "güler yüzleri": "güler yüz",
    "güleryüzlü": "güler yüz",
    "güler yüzlü": "güler yüz",
    "güler yüzü": "güler yüz",
    "güler yüzüyle": "güler yüz",
    "güleryüz": "güler yüz",
    "foto": "fotoğraf",
    "sıcak kanlılığı": "sıcakkanlı",
    "sıcak kanlı": "sıcakkanlı",
    "hizmetlo": "hizmetli",
    "içiçe": "iç içe",
    "goriste": "giriş",
    "rahatdı": "rahat",
    "fp": "fiyat performans",
    "kilima": "klima",
    "fiyatkalite": "fiyat kalite",
    "rez": "rezervasyon",
}


def apply_custom_corrections(tokens):
    return [custom_corrections.get(t, t) for t in tokens]


########################################################################
# (8) Tokenization
########################################################################
def custom_tokenize(text):
    return text.split()


# (8.1) Birleşik kelimeleri ayırmak için ek fonksiyon
def split_merged_words(tokens):
    new_tokens = []
    if wordninja is not None:
        for token in tokens:
            splitted = wordninja.split(token)
            # Eğer bölünmüş hali, orijinali veriyorsa ve 2 veya daha fazla parçaya ayrılıyorsa, parçaları ekle
            if len(splitted) > 1 and "".join(splitted) == token:
                new_tokens.extend(splitted)
            else:
                new_tokens.append(token)
    else:
        new_tokens = tokens
    return new_tokens


########################################################################
# (9) Asıl Preprocessing Pipeline
########################################################################
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # 1) Lowercase, strip ve apostrof karakterlerini kaldır
    text = text.lower().strip().replace("'", "")

    # 2) Protected ifadeleri koru
    text = protect_expressions(text)

    # 3) Tokenize ve birleşik kelimeleri ayır
    tokens = custom_tokenize(text)
    tokens = apply_diacritic_corrections(tokens)
    tokens = split_merged_words(tokens)

    # 4) Dil tespiti
    try:
        lang = detect(text)
    except:
        lang = "en"

    # 5) İşleme: Türkçe ise veya İngilizce ise
    if lang == "tr":
        tokens = [t for t in tokens if t not in turkish_stops]
        tokens = process_turkish_tokens(tokens)
        tokens = apply_custom_corrections(tokens)
        tokens = [t for t in tokens if t not in turkish_stops]
    else:
        tokens = apply_english_diacritic_corrections(tokens)
        tokens = [t for t in tokens if t not in english_stops]
        tokens = [eng_lemmatizer.lemmatize(t) for t in tokens]

    # 6) Protected ifadeleri geri döndür
    tokens = restore_expressions(tokens)
    return " ".join(tokens)


########################################################################
# (10) Ana Fonksiyon
########################################################################
def main():
    input_file = "cleaned_reviews_output3.csv"  # Temiz CSV
    output_file = "final_processed_reviews_no_spellcheck.csv"

    df = pd.read_csv(input_file, engine="python")
    if "review_text" not in df.columns:
        print("CSV'de 'review_text' sütunu bulunamadı!")
        return

    df["review_text"] = df["review_text"].fillna("")
    df["processed_review"] = df["review_text"].apply(preprocess_text)
    try:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"İşlem tamamlandı. Çıktı dosya: {output_file}")
    except PermissionError:
        print("PermissionError: Dosyaya yazma izniniz yok.")


if __name__ == "__main__":
    main()
