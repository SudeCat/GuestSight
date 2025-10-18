import pandas as pd
import ast
import json
from langdetect import detect, DetectorFactory

# Langdetect sonuçlarının tutarlı olması için seed ayarı yapıyoruz.
DetectorFactory.seed = 0

def parse_reviews(raw_reviews):
    """
    Verilen raw_reviews string'ini önce ast.literal_eval, hata verirse json.loads ile parse etmeye çalışır.
    Başarılı parse ederse yorumlar listesini döner, aksi halde boş liste döner.
    """
    if isinstance(raw_reviews, str):
        try:
            reviews = ast.literal_eval(raw_reviews)
        except Exception as e:
            try:
                reviews = json.loads(raw_reviews)
            except Exception as e2:
                reviews = []
    else:
        reviews = raw_reviews
    return reviews

# 1. Excel dosyasını, otel bilgileriyle birlikte oku.
df_raw = pd.read_excel("mugla_hotels_expanded2.xlsx")
print("Excel sütunları:", df_raw.columns)

# 2. Her otelin 'reviews' sütunundaki veriyi parse edip, otel adını ekleyerek tek bir liste oluştur.
reviews_all = []
for idx, row in df_raw.iterrows():
    hotel_name = row['name']
    raw_reviews = row['reviews']
    reviews_list = parse_reviews(raw_reviews)
    if not reviews_list:
        print(f"{hotel_name} için yorum bulunamadı veya parse edilemedi.")
        continue
    for review in reviews_list:
        review['hotel_name'] = hotel_name
        reviews_all.append(review)

print("Toplam ham yorum sayısı:", len(reviews_all))

# 3. Tüm yorumları içeren DataFrame oluştur.
df_reviews = pd.DataFrame(reviews_all)
print("df_reviews sütunları:", df_reviews.columns)

# df_reviews'te orijinal yorum metni "text" olarak geliyor; bunu "review_text" olarak yeniden adlandıralım.
if 'text' in df_reviews.columns:
    df_reviews.rename(columns={'text': 'review_text'}, inplace=True)

# 4. df_reviews için normalize edilmiş merge anahtarı oluşturuyoruz.
df_reviews['merge_key'] = (
    df_reviews['hotel_name'].astype(str).str.strip().str.lower() + "_" +
    df_reviews['author_name'].astype(str).str.strip().str.lower() + "_" +
    df_reviews['review_text'].astype(str).str.strip().str.lower()
)

# 5. Aynı merge_key için language bilgisini, varsa ilk değeri alacak şekilde sözlüğe dönüştürelim.
lang_map = df_reviews.groupby('merge_key')['language'].agg(lambda x: x.iloc[0]).to_dict()

# 6. Temizlenmiş CSV'yi oku. (Bu dosyada her yorum 1 satır, örn. 1956 satır)
df_clean = pd.read_csv("cleaned_reviews_output2.csv")
print("df_clean satır sayısı:", len(df_clean))

# 7. df_clean için de aynı şekilde normalize edilmiş merge anahtarını oluşturalım.
df_clean['merge_key'] = (
    df_clean['hotel_name'].astype(str).str.strip().str.lower() + "_" +
    df_clean['author_name'].astype(str).str.strip().str.lower() + "_" +
    df_clean['review_text'].astype(str).str.strip().str.lower()
)

# 8. Her satır için lang_map üzerinden language bilgisini getir; eğer yoksa langdetect kullan.
def get_language(row):
    key = row['merge_key']
    lang = lang_map.get(key, None)
    if lang is None or pd.isna(lang):
        try:
            lang = detect(row['review_text'])
        except Exception as e:
            lang = None
    return lang

df_clean['language'] = df_clean.apply(get_language, axis=1)

# 9. Merge anahtarını kaldırıyoruz.
df_clean.drop(columns=['merge_key'], inplace=True)

# 10. Sonuç CSV'sini yazıyoruz. (Temizlenmiş CSV'deki satır sayısı aynen korunacak: 1956 satır)
df_clean.to_csv("temizlenmis_sonuc.csv", index=False, encoding='utf-8-sig')
print("temizlenmis_sonuc.csv oluşturuldu.")
print("Final satır sayısı:", len(df_clean))
