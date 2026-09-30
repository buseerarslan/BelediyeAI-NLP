"""
================================================================================
BELEDIYE_AI v2: Sınıflandırıcı (MLP) ve Sohbet Modeli (Qwen2.5) Karşılaştırma
================================================================================
Bu betik, belediye müşteri mesajlarını ilgili departmanlara yönlendirmek için:
1. Geleneksel/Hafif Yaklaşım: Sentence-Transformers Vektörleştirme + MLP Sinir Ağı
2. Üretken Yaklaşım (LLM): Qwen2.5-0.5B-Instruct Açık Kaynak Sohbet Modeli
yöntemlerini aynı 20 test mesajı üzerinde çalıştırır, doğruluk ve hız farklarını
canlı olarak kıyaslar.
================================================================================
"""

import time
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ------------------------------------------------------------------------------
# ADIM 1: Veri Setinin Hazırlanması ve Yüklenmesi
# ------------------------------------------------------------------------------
print("=" * 70)
print("1. VERİ SETİ YÜKLENİYOR VE EĞİTİM/TEST OLARAK AYRILIYOR")
print("=" * 70)

# Veri seti (Gerekli 5 departman: Fen İşleri, Zabıta, Park ve Bahçeler, Temizlik İşleri, Su ve Kanalizasyon)
try:
    df = pd.read_csv("belediye_mesajlari.csv")
    print(f"-> 'belediye_mesajlari.csv' başarıyla yüklendi. Toplam {len(df)} mesaj var.")
except FileNotFoundError:
    print("-> UYARI: CSV bulunamadı, örnek veri kümesi oluşturuluyor...")
    data = {
        "mesaj": [
            "Sokakta çöpler birikti, koku yapıyor.", "Kaldırım taşları kırık, yüyüyemiyoruz.",
            "Parktaki salıncak kırılmış, tehlikeli.", "Kaçak dükkan işletiliyor, denetim yapın.",
            "Şebeke suyu bulanık akıyor.", "Çöp konteyneri devrilmiş.",
            "Asfalt patlamış, arabalar geçemiyor.", "Ağaç dalları elektrik tellerine değiyor.",
            "Pazarda fahiş fiyat uygulanıyor.", "Kanalizasyon tıkandı, sular bastı."
        ] * 10,
        "departman": [
            "Temizlik İşleri", "Fen İşleri", "Park ve Bahçeler", "Zabıta", "Su ve Kanalizasyon",
            "Temizlik İşleri", "Fen İşleri", "Park ve Bahçeler", "Zabıta", "Su me Kanalizasyon"
        ] * 10
    }
    df = pd.DataFrame(data)

# %80 Eğitim, %20 Test olarak veriyi bölüyoruz
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["mesaj"].tolist(), 
    df["departman"].tolist(), 
    test_size=0.2, 
    random_state=42, 
    stratify=df["departman"]
)

print(f"-> Eğitim Verisi Sayısı: {len(X_train_raw)}")
print(f"-> Test Verisi Sayısı  : {len(X_test_raw)}\n")

# ------------------------------------------------------------------------------
# ADIM 2: Yöntem 1 - Vektörleştirme (Embedding) + MLP Sinir Ağı Eğitimi
# ------------------------------------------------------------------------------
print("=" * 70)
print("2. YÖNTEM 1: EMBEDDING + MLP SİNİR AĞI MODELİ EĞİTİLİYOR")
print("=" * 70)

print("-> Sentence-Transformer (multilingual-mpnet-base-v2) yükleniyor...")
embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Metinleri 768 boyutlu vektör uzayına taşıyoruz
X_train_emb = embedder.encode(X_train_raw)
X_test_emb = embedder.encode(X_test_raw)

# MLP (Yapay Sinir Ağı) Yapılandırması (768 Girdi -> 32 Nöron ReLU -> Softmax Çıktı)
mlp_model = MLPClassifier(
    hidden_layer_sizes=(32,),
    activation='relu',
    max_iter=300,
    random_state=42
)

# Eğitimi başlatıyoruz ve süresini ölçüyoruz
start_mlp_train = time.time()
mlp_model.fit(X_train_emb, y_train)
end_mlp_train = time.time()

print(f"-> MLP Eğitimi Tamamlandı! Süre: {end_mlp_train - start_mlp_train:.4f} saniye\n")

# ------------------------------------------------------------------------------
# ADIM 3: Yöntem 2 - Qwen2.5-0.5B-Instruct Sohbet Modelinin Hazırlanması
# ------------------------------------------------------------------------------
print("=" * 70)
print("3. YÖNTEM 2: QWEN2.5-0.5B SOHBET MODELİ HAZIRLANIYOR (ZERO-SHOT)")
print("=" * 70)

llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"-> Hugging Face üzerinden '{llm_model_name}' yükleniyor...")

try:
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    print("-> Qwen2.5 Modeli başarıyla hafızaya yüklendi.\n")
    llm_available = True
except Exception as e:
    print(f"-> HATA: LLM yüklenirken bir sorun oluştu ({e}). LLM testi atlanacak.")
    llm_available = False

# Qwen modeli için yönlendirme talimatı (System Prompt)
SYSTEM_PROMPT = """Sen bir belediye mesaj yönlendirme asistanısın. 
Vatandaştan gelen mesajı incele ve aşağıdaki 5 departmandan SADECE BİRİNİ seçerek yaz.
Departmanlar:
- Fen İşleri
- Zabıta
- Park ve Bahçeler
- Temizlik İşleri
- Su ve Kanalizasyon

Cevabında kesinlikle başka hiçbir kelime veya açıklama yazma. Sadece departman adını döndür."""

def predict_qwen(mesaj_metni):
    """Qwen2.5 modeline prompt vererek departman tahmini alan yardımcı fonksiyon."""
    if not llm_available:
        return "Model Yüklenemedi"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": mesaj_metni}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
    
    with torch.no_grad():
        generated_ids = llm_model.generate(**model_inputs, max_new_tokens=15, temperature=0.1)
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response

# ------------------------------------------------------------------------------
# ADIM 4: CANLI BENCHMARK VE KARŞILAŞTIRMA (20 TEST MESAJI ÜZERİNDE)
# ------------------------------------------------------------------------------
print("=" * 70)
print("4. CANLI MODEL KARŞILAŞTIRMASI VE PERFORMANS TESTİ")
print("=" * 70)

CONFIDENCE_THRESHOLD = 0.60  # %60 Güven Eşiği (MLP için)

mlp_preds = []
llm_preds = []

mlp_times = []
llm_times = []

print(f"{'No':<3} | {'Gerçek Departman':<20} | {'MLP Tahmini (Güven)':<30} | {'Qwen2.5 Tahmini':<20}")
print("-" * 80)

for idx, (mesaj, gercek_dept) in enumerate(zip(X_test_raw, y_test), 1):
    # --- 1. MLP Tahmini ve Süre Ölçümü ---
    t0_mlp = time.time()
    vec = embedder.encode([mesaj])
    probs = mlp_model.predict_proba(vec)[0]
    max_prob = np.max(probs)
    mlp_raw_pred = mlp_model.classes_[np.argmax(probs)]
    
    # %60 Güven Eşiği Kontrolü
    if max_prob >= CONFIDENCE_THRESHOLD:
        mlp_final_pred = mlp_raw_pred
        mlp_display = f"{mlp_final_pred} (%{max_prob*100:.1f})"
    else:
        mlp_final_pred = "Temsilciye Aktar"
        mlp_display = f"Temsilciye Aktar (%{max_prob*100:.1f})"
        
    t1_mlp = time.time()
    mlp_times.append(t1_mlp - t0_mlp)
    mlp_preds.append(mlp_final_pred)

    # --- 2. Qwen2.5 LLM Tahmini ve Süre Ölçümü ---
    t0_llm = time.time()
    if llm_available:
        llm_final_pred = predict_qwen(mesaj)
    else:
        llm_final_pred = "N/A"
    t1_llm = time.time()
    
    llm_times.append(t1_llm - t0_llm)
    llm_preds.append(llm_final_pred)

    print(f"{idx:<3} | {gercek_dept:<20} | {mlp_display:<30} | {llm_final_pred:<20}")

# ------------------------------------------------------------------------------
# ADIM 5: SONUÇLARIN ÖZETLENMESİ VE MİMARİ ANALİZ
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. ÖZET METRİKLER VE RAPOR KARŞILAŞTIRMASI")
print("=" * 70)

# Hatalı/Eşik altı çıktıları filtreleyerek doğruluk hesabı
mlp_acc = accuracy_score(y_test, mlp_preds)
avg_mlp_time = sum(mlp_times) / len(mlp_times)

print(f"📊 MLP Sınıflandırıcı Doğruluğu (%60 Eşik Dâhil) : %{mlp_acc * 100:.2f}")
print(f"⚡ MLP Ortalama Yanıt Hızı (Mesaj Başına)        : {avg_mlp_time * 1000:.2f} ms")

if llm_available:
    # LLM tahminlerinde metin eşleşmesi kontrolü
    llm_correct = sum([1 for p, g in zip(llm_preds, y_test) if g.lower() in p.lower()])
    llm_acc = llm_correct / len(y_test)
    avg_llm_time = sum(llm_times) / len(llm_times)
    
    print(f"📊 Qwen2.5 LLM Doğruluğu (Zero-Shot Prompt)        : %{llm_acc * 100:.2f}")
    print(f"⚡ Qwen2.5 Ortalama Yanıt Hızı (Mesaj Başına)       : {avg_llm_time:.2f} saniye")
    print(f"🚀 Hız Farkı: MLP Modeli, Qwen LLM'den yaklaşık {avg_llm_time / avg_mlp_time:.0f} KAT DAHA HIZLI!")

print("\n--- TEORİK VE MİMARİ SONUÇ ---")
print("1. HIZ VE PERFORMANS: MLP modeli milisaniyeler içinde yanıt verirken, LLM saniyeler sürer.")
print("2. EĞİTİM İHTİYACI : MLP etiketli veri setiyle eğitilmelidir; LLM sıfır veriyle (prompt) çalışabilir.")
print("3. DONANIM MAALİYETİ: MLP CPU üzerinde rahatça çalışır; LLM yüksek GPU/VRAM kaynakları talep eder.")
print("========================================================================")

# ------------------------------------------------------------------------------
# ADIM 6: CANLI KULLANICI TESTİ (SIRA SENDE!)
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("💬 SIRA SENDE! Her iki modeli kendi mesajınla test et.")
print("   (Çıkmak için 'q' yazıp Enter'a basabilirsin)")
print("=" * 70)

while True:
    try:
        kullanici_mesaji = input("\n📝 Test etmek istediğin şikayet/talep mesajını yaz: ")
        
        # 'q', 'Q', 'cikis' veya 'exit' yazıldığında döngüden çıkış yap
        if kullanici_mesaji.lower().strip() in ['q', 'exit', 'cikis', 'quit']:
            print("\n👋 Canlı test sonlandırıldı. Başarılar!")
            break
            
        if not kullanici_mesaji.strip():
            continue

        print("\n--- TAHMİNLER HESAPLANIYOR ---")
        
        # 1. MLP Tahmini ve Süre Ölçümü
        t0_mlp = time.time()
        vec = embedder.encode([kullanici_mesaji])
        probs = mlp_model.predict_proba(vec)[0]
        max_prob = np.max(probs)
        mlp_raw_pred = mlp_model.classes_[np.argmax(probs)]
        
        if max_prob >= CONFIDENCE_THRESHOLD:
            mlp_final_pred = mlp_raw_pred
        else:
            mlp_final_pred = "Temsilciye Aktar (Belirsiz)"
        t1_mlp = time.time()
        mlp_time_ms = (t1_mlp - t0_mlp) * 1000

        # 2. Qwen2.5 Tahmini ve Süre Ölçümü
        t0_llm = time.time()
        if llm_available:
            llm_final_pred = predict_qwen(kullanici_mesaji)
        else:
            llm_final_pred = "N/A"
        t1_llm = time.time()
        llm_time_sec = t1_llm - t0_llm

        # Ekrana Kıyaslamalı Çıktı Yazdırma
        print(f"🔹 [MLP (Bizim Model)] : {mlp_final_pred:<28} | Güven: %{max_prob*100:.1f} | Süre: {mlp_time_ms:.2f} ms")
        print(f"🔸 [Qwen2.5 (LLM)]     : {llm_final_pred:<28} | Süre: {llm_time_sec:.2f} saniye")

    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Canlı test kullanıcı tarafından sonlandırıldı. Başarılar!")
        break