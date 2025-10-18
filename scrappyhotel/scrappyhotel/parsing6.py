import re
import os
import glob
import json
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

# Stopword listeleri: İngilizce ve Türkçe
english_stops = set(stopwords.words("english"))
turkish_stops = set(stopwords.words("turkish"))

# Kaggle üzerinden ek Türkçe stopword'leri ekleme (varsa)
try:
    import kagglehub

    kaggle_path = kagglehub.dataset_download("nezahatkk/tr-trke-stopwords-turkish")
    dataset_files = glob.glob(os.path.join(kaggle_path, "*"))
    stopwords_file = os.path.join(kaggle_path, "stopwords.txt")
    if os.path.exists(stopwords_file):
        with open(stopwords_file, encoding="utf-8") as f:
            additional_turkish_stops = {line.strip() for line in f if line.strip()}
        turkish_stops.update(additional_turkish_stops)
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
except ImportError:
    print("kagglehub modülü bulunamadı. 'pip install kagglehub' ile kurabilirsiniz.")

# Ekstra stopword örnekleri
extra_stopwords = {
    "artık", "keşke", "lı", "imiş", "birşeyi", "birde", "felan", "ben", "dakikalık", "falan", "dada",
    "gorunca", "u", "e", "tabiki", "yüzde", "idi", "yıl", "b", "yolarda", "istinaden", "bilhassa",
    "tmm", "nin", "an", "nolu", "tlden", "3bin", "fıkra", "sinide", "yüzden", "km", "den", "kişilik",
    "tane", "azından", "kişi", "dahaki", "edrr", "miyiz", "mi", "olurda", "herşeyin", "üzerine",
    "dahakine", "kat", "m2", "lik", "bişeyler", "kadar", "le", "bişey", "sene", "ani", "bana",
    "numaralı", "kısacası", "yani", "anlamda", "yaklaşık", "buçuk", "senedir", "de", "dan", "nolu",
    "dakikada", "da", "yi", "yüzden", "tane", "birşeye", "el", "hemde", "ha", "gözle", "şekilde", "ki",
    "daki", "nın", "m", "gündür", "mesela", "kimse", "adet", "bence", "ötürü", "katı", "teymiş", "mt",
    "ni", "e", "den", "an", "lar", "l", "ta", "nedir", "daki", "numaralı", "dklık", "300m", "kglik",
    "yeterki", "içinde", "dk", "tekarr", "biryer", "herşeyiyle", "lira", "liraya", "ayrıyetten",
    "numara", "birde", "tarihinde", "10y", "4y", "bi", "haberin", "ayriyetten", "şeyiyle", "başka",
    "bişey", "anda", "bende", "metre", "agodadan", "agodada", "2kisilik", "1kisilik", "ikiüç",
    "nu", "rabbim", "lik", "kazandırdıkları", "napolyonun", "askerini", "muhtemelen", "kanlarıyla",
    "şehitlerimizin", "15dk", "10dk", "5dk", "50dk", "2dk", "vb", "yaşındaki", "aylik", "dışında",
    "yokmu", "sebepten", "vs", "haberiniz", "zaman", "şimdiden", "gün", "verilse", "300tl", "dakika",
    "ay", "te", "şuan", "yada", "bey", "ye", "100tl", "bide", "bir", "şeyin", "yıllardır", "yıldır",
    "ki", "hergün", "", "tl"
}
turkish_stops.update(extra_stopwords)

extra_stopwords_eng = {
    "km", "going", "u", "im", "höt", "year", "old", "minutes",
    "18mpbs", "us", "lets", "come", "theres", "min", "ms", "mr", "yes", "11pm",
    "2am", "10pm", "id", "etc", "motors", "tl", "3rd", "its","etc"
}
english_stops.update(extra_stopwords_eng)

for word in ["her", "var", "iyi", "kötü", "yok", "değil", "en", "sağolsun", "sağ", "olsun", "maalesef",
             "rağmen", "olmaz", "değmez"]:
    turkish_stops.discard(word)

print(f"Türkçe stopwords: {len(turkish_stops)} adet")
print(f"İngilizce stopwords: {len(english_stops)} adet")

# İngilizce lemmatizer (WordNet)
eng_lemmatizer = WordNetLemmatizer()


# Türkçe morfolojik analiz için placeholder (geliştirilebilir)
def process_turkish_tokens(tokens):
    return tokens


# Diakritik / yazım düzeltmeleri
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
    return [diacritic_map[t] if t in diacritic_map else t for t in tokens]


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
    return [english_diacritic_map[t] if t in english_diacritic_map else t for t in tokens]


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
# (8) Protected İfadeler: Fonksiyonlar
########################################################################
def protect_expressions(text, expressions):
    """Her protected ifadeyi benzersiz bir placeholder ile değiştirir."""
    placeholders = {}
    for i, expr in enumerate(expressions):
        placeholder = f"___PROTECTED_{i}___"
        pattern = re.compile(re.escape(expr), flags=re.IGNORECASE)
        text = pattern.sub(placeholder, text)
        placeholders[placeholder] = expr  # orijinal ifade
    return text, placeholders


def restore_protected_expressions(text, placeholders):
    """Placeholder'ları orijinal protected ifadelerle geri dönüştürür."""
    for placeholder, expr in placeholders.items():
        text = text.replace(placeholder, expr)
    return text


########################################################################
# (9) Tokenization ve Birleşik Kelime Ayırma
########################################################################
def custom_tokenize(text):
    return text.split()


def split_merged_words(tokens):
    new_tokens = []
    if wordninja is not None:
        for token in tokens:
            splitted = wordninja.split(token)
            if len(splitted) > 1 and "".join(splitted) == token:
                new_tokens.extend(splitted)
            else:
                new_tokens.append(token)
    else:
        new_tokens = tokens
    return new_tokens


########################################################################
# (10) Tek Aşamada Preprocessing ve Lemmatization
########################################################################
def full_process_review(review: str) -> str:
    if not isinstance(review, str):
        return ""
    # 1) Küçük harfe çevir, trim yap, apostrof karakterlerini kaldır
    text = review.lower().strip().replace("'", "")
    # 2) Protected ifadeleri placeholder'lara çevir
    text, placeholders = protect_expressions(text, protected_expressions)
    # 3) Tokenize, diakritik düzeltmeleri uygula, birleşik kelimeleri ayır
    tokens = custom_tokenize(text)
    tokens = apply_diacritic_corrections(tokens)
    tokens = split_merged_words(tokens)
    text = " ".join(tokens)
    # 4) Dil tespiti
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang == "tr":
        # Türkçe: stopword'leri çıkar, custom düzeltmeleri uygula
        tokens = text.split()
        tokens = [t for t in tokens if t not in turkish_stops]
        tokens = process_turkish_tokens(tokens)
        tokens = apply_custom_corrections(tokens)
        tokens = [t for t in tokens if t not in turkish_stops]
        cleaned_text = " ".join(tokens)
        # 5) Lemmatization (spaCy) – placeholder'ları korumak için
        import spacy
        nlp = spacy.load("tr_core_news_trf")
        doc = nlp(cleaned_text)
        lemmatized_tokens = []
        for token in doc:
            if token.text.startswith("___PROTECTED_"):
                original = placeholders.get(token.text, token.text)
                lemmatized_tokens.append(original)
            else:
                lemmatized_tokens.append(token.lemma_)
        final_text = " ".join(lemmatized_tokens)
    else:
        # İngilizce: diakritik düzeltmeleri, stopword temizliği ve WordNet lemmatization
        tokens = apply_english_diacritic_corrections(tokens)
        tokens = [t for t in tokens if t not in english_stops]
        tokens = [eng_lemmatizer.lemmatize(t) for t in tokens]
        final_text = " ".join(tokens)
    # 6) Protected ifadeleri geri yükle
    final_text = restore_protected_expressions(final_text, placeholders)
    return final_text


########################################################################
# (11) Ana Fonksiyon: Tek CSV Üzerinde İşlem
########################################################################
def main():
    # Giriş CSV: "cleaned_reviews_output3.csv" (içinde "review_text" sütunu)
    input_file = "cleaned_reviews_output3.csv"
    # Çıkış CSV: İşlenmiş yorumlar "processed_review" sütununda
    output_file = "final_processed_reviews_spcy.csv"

    df = pd.read_csv(input_file, engine="python")
    if "review_text" not in df.columns:
        print("CSV'de 'review_text' sütunu bulunamadı!")
        return
    df["review_text"] = df["review_text"].fillna("")

    # Her yorumu full_process_review fonksiyonuyla işle
    df["processed_review"] = df["review_text"].apply(full_process_review)

    try:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"İşlem tamamlandı. Çıktı dosya: {output_file}")
    except PermissionError:
        print("PermissionError: Dosyaya yazma izniniz yok.")


if __name__ == "__main__":
    main()
