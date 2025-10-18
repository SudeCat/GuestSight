import spacy
import pandas as pd

# SpaCy modelini yükleyin (örn. İngilizce için)
nlp = spacy.load("en_core_web_sm")

# CSV dosyanızı okuyun (dosya yolunu ve adını güncelleyin)
df = pd.read_csv("final_processed_reviews_deepl_use.csv")

# CSV dosyasındaki sütun adlarını yazdırın, böylece hangi sütunu kullanacağınızı belirleyin
print("Mevcut sütunlar:", df.columns)

# Örneğin, eğer doğru sütun adı 'final_processed_review' ise:
for idx, text in enumerate(df["processed_final_review"], start=1):
    print(f"--- Review {idx} ---")
    doc = nlp(text)

    # POS Tagging İşlemi
    print("\n[POS Tagging]")
    for token in doc:
        # token.text: orijinal kelime, token.pos_: genel POS etiketi, token.tag_: detaylı POS etiketi
        print(f"{token.text}\t{token.pos_}\t{token.tag_}")

    # Named Entity Recognition (NER) İşlemi
    print("\n[Named Entities]")
    if doc.ents:
        for ent in doc.ents:
            print(f"{ent.text}\t({ent.label_})")
    else:
        print("Bu metinde tespit edilmiş varlık bulunamadı.")

    print("\n" + "=" * 50 + "\n")
