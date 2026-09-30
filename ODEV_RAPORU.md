# BELEDİYE AI: MÜŞTERİ MESAJLARI OTOMATİK YÖNLENDİRME VE MODEL KARŞILAŞTIRMA RAPORU

**Öğrenci / Geliştirici:** Yapay Zekâ Mühendisliği Adayı  
**Proje Adı:** BelediyeAI - Doğal Dil İşleme ve Sınıflandırma Mimarisi  
**Kod Dosyaları:** `belediye_ai.py` (Ana Uygulama & Görev 1-4) ve `belediye_ai_v2.py` (LLM Benchmark & Görev 5)  

---

## ❓ SORU 1: Projenin Amacı ve Mimari Yapısı Nedir?

**CEVAP:**  
Bu projenin temel amacı, vatandaşlar tarafından belediyeye iletilen şikayet, talep ve öneri mesajlarını Doğal Dil İşleme (NLP) ve Derin Öğrenme teknikleri kullanarak ilgili belediye departmanına otomatik olarak yönlendirmektir. 

Sistem iki farklı mimari yaklaşımı ve kodu içermektedir:
1. **`belediye_ai.py` (Özel Sınıflandırıcı Mimarisi):** Geleneksel metin temsil yöntemleri yerine, önceden eğitilmiş bir Transformer modeli (`paraphrase-multilingual-mpnet-base-v2`) üzerinden metinleri 768 boyutlu anlamsal vektörlere (embedding) dönüştürür. Bu vektörler, özel olarak eğitilmiş Çok Katmanlı Algılayıcı (**MLP - Multi-Layer Perceptron**) yapay sinir ağına beslenir.
2. **`belediye_ai_v2.py` (Canlı Benchmark & LLM Karşılaştırması):** Aynı veri kümesi üzerinde üretken sohbet modeli olan **Qwen2.5-0.5B-Instruct LLM** (Large Language Model) sıfır-örnekli (zero-shot) olarak çalıştırılmış ve her iki sistem canlı olarak **hız**, **doğruluk**, **maliyet** ve **güvenilirlik** yönünden kıyaslanmıştır.

---

## ❓ SORU 2: Metin İşleme ve Tokenizer (Anlamsal Parçalama) Adımında Ne Yapılmıştır?

**CEVAP:**  
Metinlerin yapay zekâ modelleri tarafından işlenebilmesi için sayısal dizilere dönüştürülmesi gerekir. Projede **GPT-2 Tokenizer** kullanılarak bağlam penceresi (Context Window) analizi gerçekleştirilmiştir.

* **Context Window (Bağlam Penceresi) Tespiti:** GPT-2 Tokenizer, maksimum $1024$ token'lık bir bağlam penceresine sahiptir.
* **Uzunluk Analizi:** Veri setindeki mesajların ortalama token uzunluğu hesaplanmış ve gelen mesajların modelin taşıyabileceği kapasitenin oldukça altında olduğu, dolayısıyla hiçbir veri kaybı/kırpma (truncation) yaşanmadığı doğrulanmıştır.
* **Tokenizasyon Mekanizması:** Kelimeler alt sözcük birimlerine (subwords) ayrılarak modelin bilinmeyen kelimelerle (out-of-vocabulary) karşılaştığında da anlamsal kökleri koruması sağlanmıştır.

---

## ❓ SORU 3: Anlamsal Vektörler (Text Embedding) Nasıl Elde Edilmiştir?

**CEVAP:**  
Geleneksel TF-IDF veya Bag-of-Words gibi yöntemler kelimelerin sadece frekansına bakar, anlamsal ilişkileri kuramaz (örneğin "çöp" ile "atık" kelimelerini farklı kabul eder). 

Bu sorunu çözmek için **Sentence-Transformers** kütüphanesinden `paraphrase-multilingual-mpnet-base-v2` modeli kullanılmıştır.
* **Çok Dilli Destek:** Türkçe metinler üzerindeki anlamsal bağlamı yüksek hassasiyetle yakalar.
* **768-Boyutlu Vektör Alanı:** Her bir vatandaş mesajı, 768 elemanlı sürekli bir reel sayı vektörüne $\mathbb{R}^{768}$ dönüştürülür. Bu sayede "Musluktan çamurlu su akıyor" mesajı ile "Su borusu patladı" mesajı vektör uzayında birbirine çok yakın konumlandırılır.

---

## ❓ SORU 4: Sınıflandırma Modeli (MLP) Nasıl Eğitilmiş ve Konfigüre Edilmiştir?

**CEVAP:**  
Elde edilen 768 boyutlu vektörler, Scikit-Learn kütüphanesi kullanılarak bir **MLPClassifier** (Çok Katmanlı Algılayıcı) modeline beslenmiştir.

* **Mimari Yapı:**
  * **Girdi Katmanı:** 768 Nöron (Embedding boyutu)
  * **Gizli Katmanlar (Hidden Layers):** `(100, 50)` — 100 nöronlu ilk gizli katman ve 50 nöronlu ikinci gizli katman.
  * **Aktivasyon Fonksiyonu:** `ReLU` (Rectified Linear Unit)
  * **Optimizasyon:** `Adam` optimizer (Öğrenme oranı: $0.001$)
  * **Çıktı Katmanı:** 5 Nöron (Departman sınıfları) + `Softmax` fonksiyonu
* **Çıktı Sınıfları:** `Temizlik ve Çöp`, `Park ve Bahçeler`, `Su ve Kanalizasyon`, `Ulaşım ve Trafik`, `Zabıta`.

---

## ❓ SORU 5: Güven Eşiği (Confidence Threshold) Mekanizması Nasıl Çalışır ve Neden Önemlidir?

**CEVAP:**  
Yapay zekâ modellerinin en büyük risklerinden biri, emin olmadıkları konularda da rastgele veya yanlış bir tahmin üretmeleridir. Projede bu riski engellemek için **%60 Güven Eşiği (Confidence Threshold = 0.60)** mekanizması kurulmuştur.

* **Çalışma Mantığı:** Model bir mesaj için Softmax olasılık dağılımı üretir: $P(Y=k|X)$.
* **Karar Kuralı:** 
  $$\text{Karar} = \begin{cases} \text{Sınıf}_k, & \text{eğer } \max(P) \ge 0.60 \\ \text{"Temsilciye Aktar (Belirsiz)"}, & \text{eğer } \max(P) < 0.60 \end{cases}$$
* **Sistem Dışı (Out-of-Scope) Örnek Testi:**
  * Mesaj: *"etkinlik takvimi var mı"* (Verisetinde Kültür/Etkinlik sınıfı yok).
  * Sonuç: MLP Modeli %49.9 güven üretti. Skor %60'ın altında kaldığı için sistem yanlış yönlendirme yapmak yerine otomatik olarak **"Temsilciye Aktar"** kararı vererek sistemi güvenli tarafta tuttu.

---

## ❓ SORU 6: Görev 5 - MLP Sınıflandırıcısı ile Qwen2.5 (LLM) Karşılaştırması Sonuçları Nelerdir?

**CEVAP:**  
`belediye_ai_v2.py` scripti üzerinden gerçekleştirilen canlı testlerde, 20 adet test mesajı her iki modele de eşzamanlı verilmiş ve kronometre ile ölçüm yapılmıştır.

### 📊 Canlı Deneysel Benchmark Tablosu:

| Karşılaştırma Metriği | Sınıflandırıcı Model (Embedding + MLP) | Sohbet Modeli (Qwen2.5-0.5B-Instruct LLM) |
| :--- | :--- | :--- |
| **Model Tipi** | Özel Eğitilmiş Derin Sinir Ağı | Genel Amaçlı Üretken Dil Modeli (Zero-Shot) |
| **Ortalama İşlem Süresi (Inference)** | **23.8 ms - 31.4 ms** (Milisaniye) | **2.10 s - 3.09 s** (Saniye) |
| **Hız Farkı** | **~76 Kat Daha Hızlı** | Oldukça Yavaş |
| **Test Doğruluk Oranı (Accuracy)** | **%70 - %85** | **%30** (Metin formatı sapmaları nedeniyle) |
| **Olasılıksal Güven Skoru** | Üretir (%0 - %100 arası net skor) | Üretmez (Sadece metin çıktısı verir) |
| **Donanım & Kaynak Tüketimi** | Çok Düşük (CPU üzerinde anlık) | Yüksek (VRAM / GPU ihtiyacı) |

### 🔍 Kıyaslama Analizi ve Nedenleri:
1. **Hız:** MLP modeli sadece vektör matris çarpımı yaptığı için saniyenin 40'ta birinde karar verir. Qwen2.5 ise token token üretim (autoregressive generation) yaptığı için 2.5 saniye sürer.
2. **Doğruluk ve Halüsinasyon:** Qwen2.5 modeli önceden eğitilmediği (zero-shot) ve sadece prompt ile yönlendirildiği için "Park ve Bahçeler" demek yerine "Park ve Bahçeler Departmanı" gibi ekstra kelimeler uydurmuş veya "ağaç devrilmiş" şikayetine "Fen İşleri" yanıtı vererek sınıf dışına taşmıştır.
3. **Güvenilirlik:** MLP modeli belirsiz durumlarda güven eşiği ile "Temsilciye Aktar" diyebilirken, LLM her durumda kesin bir metin uydurmaya çalışmaktadır.

---

## ❓ SORU 7: Projenin Sonuçları ve Mühendislik Değerlendirmesi Nedir?

**CEVAP:**  
Bu çalışma, yapay zekâ projelerinde **"En büyük/popüler model her zaman en iyi çözüm değildir"** prensibini somut verilerle kanıtlamıştır.

* **Sonuç:** Belediye şikayet yönlendirmesi gibi dar kapsamlı, spesifik ve yüksek hız gerektiren görevlerde; özel olarak dönüştürülmüş anlamsal vektörler (Sentence-Embeddings) ve hafif bir sınıflandırıcı (MLP) kullanmak, devasa dil modellerine (LLM) göre **76 kat daha hızlı**, **daha yüksek doğruluklu** ve **maliyetsizdir**.