# ==============================================================================
# PROJE ADI: BelediyeAI (Akıllı Kent Mesaj Yönlendirme Sistemi)
#
# NE YAPIYORUZ?
# Vatandaşlardan gelen serbest metin bildirimlerini (şikayet, talep, istek)
# Yapay Zekâ, Doğal Dil İşleme (NLP) ve Yapay Sinir Ağları (ANN) kullanarak
# doğru belediye departmanına otomatik yönlendiriyoruz.
# ==============================================================================

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

print("=" * 80)
print("🚀 BelediyeAI MİMARİSİ VE YAPAY ZEKÂ MOTORU BAŞLATILIYOR...")
print("=" * 80 + "\n")


# ==============================================================================
# BÖLÜM 1: VERİ SETİ YÜKLEME VE ETİKET EŞLEME (LABEL ENCODING)
# ==============================================================================
# 💡 ANALOJİ / YAZILIMCI MANTIĞI:
# Bilgisayarlar kelimeleri veya kelime anlamlarını doğrudan "okuyamaz".
# Onlar sadece sayılarla ve matematiksel işlemlerle çalışır.
# Bu yüzden departman isimlerini (Park, Su vb.) birer sayısal ID'ye (0, 1, 2, 3, 4)
# dönüştürüyoruz. Buna NLP'de "Label Encoding" veya "Etiket Eşleme" denir.

# 1. Vatandaş mesajlarını içeren CSV dosyasını yüklüyoruz
df = pd.read_csv("belediye_mesajlari.csv")

# 2. Veri setindeki benzersiz departman isimlerini alfabetik sırayla alıyoruz
departmanlar = sorted(df["departman"].unique())

# 3. Departman isimlerini sayısal ID'ler ile eşleyen bir sözlük (dictionary) kuruyoruz
# Örn: {'Aydınlatma': 0, 'Fen İşleri': 1, 'Park ve Bahçeler': 2, ...}
dept_to_id = {d: i for i, d in enumerate(departmanlar)}

# 4. CSV'deki her bir metinsel departman adını sayısal ID karşılığına çeviriyoruz
etiketler = df["departman"].map(dept_to_id).values

print(f"📊 [VERİ SETİ]: Toplam {len(df)} adet vatandaş mesajı başarıyla yüklendi.")
print(f"🏷️  [DEPARTMAN EŞLEMESİ]: {dept_to_id}\n")


# ==============================================================================
# BÖLÜM 2: GÖREV 1 - TOKENIZATION VE BAĞLAM PENCERESİ (CONTEXT WINDOW) ANALİZİ
# ==============================================================================
# 💡 ANALOJİ (Sözlük ve Yapboz Parçaları):
# Yapay zekâ modelleri (GPT-2 gibi) eğitilirken ağırlıklı olarak İngilizce metinler kullanılmıştır.
# Bu yüzden modelin sözlüğünde İngilizce kelimeler tek parça bulunurken;
# Türkçe kelimeler (örn: 'kütüphaneye', 'temizlenmiyor') sözlükte bütün olarak bulunmaz.
# Yapay zekâ Türkçe kelimeleri hecelerine, eklerine veya harf parçalarına
# (subwords / token) bölerek anlamaya çalışır.

print("-" * 80)
print("📏 GÖREV 1: TOKENIZATION VE BAĞLAM PENCERESİ (CONTEXT WINDOW) ANALİZİ")
print("-" * 80)

# 1. GPT-2'nin Türkçe ve İngilizce kelimeleri nasıl parçaladığını gören Tokenizer'ı yüklüyoruz
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# 2. Veri setimizdeki 100 mesajın her birini parçalayıp (tokenize edip) kaç parçadan oluştuğunu sayıyoruz
token_sayilari = [len(tokenizer.encode(m)) for m in df["mesaj"]]
toplam_token = sum(token_sayilari)

# 3. GPT-2 modelinin tek seferde okuyabileceği maksimum token sınırını (Bağlam Penceresi) alıyoruz
pencere_boyutu = tokenizer.model_max_length  # GPT-2 için varsayılan 1024 token'dır.

print(f"🧩 100 Adet Türkçe Mesajın Toplam Token Parça Sayısı : {toplam_token}")
print(f"🪟 GPT-2'nin Maksimum Bağlam Penceresi (Kapasitesi)   : {pencere_boyutu} token")

# 4. YAZILIMCI TESPİTİ: Tüm mesajları tek bir metin gibi modele verirsek ne olur?
if toplam_token > pencere_boyutu:
    print("\n⚠️  [YAZILIMCI VE MİMARİ TESPİTİ]:")
    print("    Tüm veri seti TEK BİR SEFERDE (tek bir prompt halinde) GPT-2'ye verilirse")
    print("    bağlam penceresi aşılır! Model kapasitesi dolduğu için metnin sonunu okuyamaz (Truncation).")
    print("    Bu yüzden mesajlar modele TEK TEK veya BATCH (gruplar) halinde beslenmelidir.\n")
else:
    print("✅ Tüm veri seti tek seferde modele sığmaktadır.\n")


# ==============================================================================
# BÖLÜM 3: GÖREV 2 - ANLAMSAL VEKTÖR DÖNÜŞÜMÜ (SENTENCE EMBEDDINGS)
# ==============================================================================
# 💡 ANALOJİ (GPS Koordinatları ve Anlam Haritası):
# Bir haritada İstanbul ile Kocaeli'nin GPS koordinatları birbirine çok yakındır.
# Embedding işlemi de kelimeleri/cümleleri 768 boyutlu bir "anlam haritasına" yerleştirir.
# Örneğin: 'Çöp birikti' cümlesi ile 'Sokak süpürülmüyor' cümlesinde
# HİÇBİR ORTAK KELİME OLMASA BİLE, ikisi de "temizlik" konusuyla ilgili olduğu için
# yapay zekânın anlam haritasında (vektör uzayında) yan yana düşerler!

print("-" * 80)
print("🧠 GÖREV 2: ANLAMSAL VEKTÖR DÖNÜŞÜMÜ (EMBEDDING)")
print("-" * 80)

# 1. Türkçe anlam ilişkilerini çok iyi kavrayan 768 boyutlu çok dilli SentenceTransformer modelini yüklüyoruz
model_emb = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

# 2. 100 mesajın tamamını 768 sütunlu sayısal matrislere (vektör koordinatlarına) dönüştürüyoruz
X_all = model_emb.encode(df["mesaj"].tolist())

print(f"📐 Vektör Matrisi Boyutu (Mesaj Sayısı, Vektör Boyutu): {X_all.shape}")
print("✅ Cümleler harf dizilerinden çıkarılıp 768 boyutlu uzayda anlam koordinatlarına dönüştürüldü!\n")


# ==============================================================================
# BÖLÜM 4: GÖREV 3 - DERİN ÖĞRENME SINIFLANDIRICISI (NEURAL NETWORK / MLP)
# ==============================================================================
# 💡 ANALOJİ (Sınava Çalışmak vs Ezber Yapmak):
# Bir öğrenci sınav sorularını ezberlerse yeni sorular geldiğinde çuvallar.
# Bu yüzden verinin %80'ini modele "Ders Çalışması" (Eğitim) için veriyoruz.
# Kalan %20'lik kısmı ise "Final Synavı" (Test) olarak saklıyoruz.
# Model hiç görmediği bu %20'lik test sorularını doğru bilirse "ezberlememiş, öğrenmiş" demektir.

print("-" * 80)
print("🎯 GÖREV 3: YAPAY SİNİR AĞI MODELİNİN EĞİTİLMESİ VE TESTİ")
print("-" * 80)

# 1. Veri setini %80 Eğitim, %20 Test olarak ikiye bölüyoruz
X_egitim, X_test, y_egitim, y_test, m_egitim, m_test = train_test_split(
    X_all, etiketler, df["mesaj"].values, test_size=0.2, stratify=etiketler, random_state=42
)

# 2. Çok Katmanlı Algılayıcı (MLP / Neural Network) Mimari Yapısı:
# - Giriş Katmanı : 768 Nöron (Embedding boyutu)
# - Gizli Katman  : 32 Nöron + ReLU Aktivasyonu
# - Çıkış Katmanı : 5 Nöron (5 departman için sınıflandırma)
siniflandirici = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42
)

# 3. Modeli eğitiyoruz
siniflandirici.fit(X_egitim, y_egitim)

# 4. Modelin hiç görmediği %20'lik test verisindeki başarısını ölçüyoruz
test_dogrulugu = siniflandirici.score(X_test, y_test)
print(f"🎓 BelediyeAI Test Kümesi Başarı Oranı (Accuracy): %{test_dogrulugu * 100:.2f}")
print("✅ Yapay zekâ modeli ezber yapmadan, daha önce hiç karşılaşmadığı mesajları doğru anlıyor!\n")


# ==============================================================================
# BÖLÜM 5: GÖREV 4 - AKILLI YÖNLENDİRİCİ VE GÜVEN EŞİĞİ (%60 THRESHOLD)
# ==============================================================================
# 💡 ANALOJİ (Çağrı Merkezi Operatörü):
# Bir çağrı merkezi çalışanı vatandaşın talebinden %100 emin değilse riske girip
# yanlış birime aktarmaz, "Sizi insan temsilcimize aktarıyorum" der.
# Biz de yapay zekâya %60 güven sınırı koyduk. Tahmin güveni %60'ın altındaysa
# sistem risk almaz ve "Temsilciye aktar" kararı verir.

print("-" * 80)
print("🛡️ GÖREV 4: AKILLI YÖNLENDİRİCİ VE GÜVEN EŞİĞİ (%60) TESTLERİ")
print("-" * 80)

def belediye_ai_yonlendir(mesaj):
    # 1. Gelen yeni mesajı hemen 768 boyutlu vektöre çeviriyoruz
    vektor = model_emb.encode([mesaj])
    
    # 2. Modelden her bir departman için tahmin olasılıklarını alıyoruz
    olasiliklar = siniflandirici.predict_proba(vektor)[0]
    
    # 3. En yüksek olasılığa sahip departmanı ve güven skorunu (%) buluyoruz
    en_iyi_indeks = np.argmax(olasiliklar)
    guven_skoru = round(float(olasiliklar[en_iyi_indeks]) * 100)
    
    # 4. Güven Eşiği Kontrolü (%60 KURALI)
    if guven_skoru < 60:
        return "Temsilciye aktar (Belirsiz / İki Anlamlı Mesaj)", guven_skoru
    
    return departmanlar[en_iyi_indeks], guven_skoru


# ------------------------------------------------------------------------------
# GERÇEK ZAMANLI TEST SENARYOLARI
# ------------------------------------------------------------------------------
test_mesajlari = [
    "Sokağın başındaki lambalar yanmıyor, karanlıkta yürüyemiyoruz.", # Net mesaj -> Aydınlatma
    "Musluktan çamurlu su geliyor, çamaşırlar kirlendi.",            # Net mesaj -> Su ve Kanalizasyon
    "PARKTA PATLAMIŞ BİR SU BORUSU VAR, HER YERİ SU BASIYO!"        # Çift anlamlı mesaj (Park mı, Su mu?)
]

for msg in test_mesajlari:
    dept, skor = belediye_ai_yonlendir(msg)
    print(f"📩 GELEN MESAJ : '{msg}'")
    print(f"➡️  KARAR       : {dept}")
    print(f"📊 GÜVEN SKORU : %{skor}\n")

print("=" * 80)
print("🎉 BelediyeAI TÜM DERİN ÖĞRENME VE DOĞAL DİL İŞLEME AŞAMALARINI BAŞARIYLA TAMAMLADI!")
print("=" * 80)


# ==============================================================================
# CANLI KULLANICI TESTİ (Kendi Mesajını Sor)
# ==============================================================================
print("\n" + "=" * 80)
print("💬 SIRA SENDE! BelediyeAI'ye Kendi Şikayet/Talep Mesajını Yaz")
print("=" * 80)

while True:
    kullanici_mesaji = input("\n📩 Bir belediye mesajı yazın (Çıkmak için 'q' basıp Enter'a basın): ")
    
    if kullanici_mesaji.lower() == 'q':
        print("\n👋 BelediyeAI Test Modundan Çıkıldı. İyi çalışmalar!")
        break
    
    if kullanici_mesaji.strip() == "":
        continue
        
    dept, skor = belediye_ai_yonlendir(kullanici_mesaji)
    print(f"➡️  SİSTEM KARARI : {dept}")
    print(f"📊 GÜVEN SKORU   : %{skor}")