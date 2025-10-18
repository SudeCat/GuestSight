import pandas as pd
import re

# Emoji desenimizi güncelliyoruz; burada ekstra aralığı (U+1F900-U+1F9FF) ekledik.
emoji_pattern = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # Emoticonlar
    u"\U0001F300-\U0001F5FF"  # Semboller ve piktogramlar
    u"\U0001F680-\U0001F6FF"  # Ulaşım ve harita sembolleri
    u"\U0001F1E0-\U0001F1FF"  # Bayraklar
    u"\U00002702-\U000027B0"  # Diğer semboller
    u"\U000024C2-\U0001F251"  # Ek semboller
    u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs (🤩 gibi emojileri kapsar)
    "]+",
    flags=re.UNICODE
)


def remove_emojis(text: str) -> str:
    """Metin içerisindeki emojileri temizler."""
    return emoji_pattern.sub(r'', text)


def minimal_cleaning(text):
    """
    Metin temizleme:
      - Emojileri kaldırır
      - Küçük harfe çevirir
      - Fazla boşlukları teke indirir
    """
    if not isinstance(text, str):
        return ""

    text = remove_emojis(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_reviews(input_file, output_file):
    # Excel'den veriyi okuyun
    df = pd.read_excel(input_file)

    # Gerekli sütunların varlığını kontrol edin
    required_cols = {'hotel_name', 'author_name', 'review_text'}
    if not required_cols.issubset(df.columns):
        print(f"Excel dosyasında şu sütunlar eksik: {required_cols - set(df.columns)}")
        return

    cleaned_data = []

    for idx, row in df.iterrows():
        hotel = str(row['hotel_name'])
        author = str(row['author_name'])
        review_raw = str(row['review_text'])

        cleaned_review = minimal_cleaning(review_raw)

        cleaned_data.append({
            'hotel_name': hotel,
            'author_name': author,
            'review_text': cleaned_review
        })

    new_df = pd.DataFrame(cleaned_data)
    new_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Temizlenmiş yorumlar '{output_file}' dosyasına kaydedildi.")


if __name__ == "__main__":
    input_file = "mugla_hotels_reviews_fixed.xlsx"  # Girdi dosyanız
    output_file = "cleaned_reviews_output2.csv"  # Çıktı dosyası
    process_reviews(input_file, output_file)
