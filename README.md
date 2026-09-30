# BelediyeAI: Yapay Zekâ Destekli Müşteri Mesajları Otomatik Yönlendirme ve Karşılaştırma Sistemi

**BelediyeAI**, vatandaşlardan gelen şikayet ve talep mesajlarını Doğal Dil İşleme (NLP) ve Derin Öğrenme yöntemleri kullanarak ilgili belediye departmanlarına yönlendiren üretim odaklı bir yapay zekâ projesidir.

Bu projede geleneksel **Özel Sınıflandırıcı (Sentence-Transformers + MLP)** mimarisi ile üretken **Sohbet Modeli (Qwen2.5-0.5B-Instruct LLM)** yaklaşımı; **doğruluk**, **işlem hızı (inference latency)**, **donanım maliyeti** ve **güvenilirlik** kriterleri açılarından canlı olarak karşılaştırılmıştır.

---

## 🛠️ Proje Mimarisi ve Kod Yapısı

Proje iki ana çalıştırma script'inden ve bir rapor dosyasından oluşur:

1. **`belediye_ai.py` (Ana Uygulama):**
   * **Görev 1-4** adımlarını gerçekleştirir.
   * GPT-2 Tokenizer ile bağlam penceresi (Context Window) analizi yapar.
   * `paraphrase-multilingual-mpnet-base-v2` modeli ile 768 boyutlu anlamsal vektörler (Embedding) üretir.
   * `MLPClassifier` (Yapay Sinir Ağı) eğiterek %60 Güven Eşiği (Confidence Threshold) mekanizmasını işletir.
   * Etkileşimli terminal arayüzü sunar.

2. **`belediye_ai_v2.py` (Görev 5 - Canlı LLM Karşılaştırma ve Benchmark):**
   * Aynı test mesaj kümesini hem eğitilmiş **MLP** modeline hem de **Qwen2.5-0.5B-Instruct** (Zero-shot LLM) sohbet modeline sunar.
   * Her iki modelin yanıtlarını, milisaniye düzeyinde işlem sürelerini ve doğruluk oranlarını yan yana tablo halinde raporlar.
   * Canlı kullanıcı etkileşimi ile iki modelin anlık kıyaslanmasını sağlar.

3. **`ODEV_RAPORU.md`:**
   * Tüm derin öğrenme adımlarının, akademik gerekçelerin ve karşılaştırma metriklerinin detaylandırıldığı teslimat raporudur.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimlerin Yüklenmesi
Sanal ortamınızı (venv) aktif ettikten sonra gerekli kütüphaneleri yükleyin:

```bash
pip install pandas numpy scikit-learn sentence-transformers transformers torch accelerate
```

### 2. Ana Uygulamanın Çalıştırılması (Görev 1-4)
```bash
python belediye_ai.py
```

### 3. Canlı LLM Benchmark ve Karşılaştırma Testi (Görev 5)
```bash
python belediye_ai_v2.py
```

---

## 📊 Deneysel Benchmark Sonuçları

Aşağıdaki veriler aynı 20 test mesajı üzerinde gerçekleştirilen `belediye_ai_v2.py` çalıştırmasından elde edilmiştir:

| Karşılaştırma Kriteri | Sınıflandırıcı Model (Embedding + MLP) | Sohbet Modeli (Qwen2.5-0.5B-Instruct LLM) |
| :--- | :--- | :--- |
| **Ön Eğitim İhtiyacı** | Etiketli Veri Seti Gerektirir (100 Örnek) | Ön Eğitim Gerektirmez (Zero-Shot Prompt) |
| **Ortalama Yanıt Hızı** | **~25 - 40 ms** (Milisaniye) | **~2.10 - 3.09 s** (Saniye) |
| **Doğruluk Oranı (Accuracy)**| **%70 - %85** | **%30** (Format ve Kapsam Sapmaları) |
| **Güven Eşiği Desteği** | Var (%60 altı $\rightarrow$ Temsilciye Aktar) | Yok (Skor Üretmez, Halüsinasyona Açık) |
| **Donanım İhtiyacı** | Standart CPU Yeterli | Yüksek VRAM / GPU İhtiyacı |

### 🔑 Temel Çıkarım
Özel olarak eğitilmiş hafif sınıflandırıcı (MLP) modeli, genel amaçlı bir dil modeline (LLM) kıyasla **~76 kat daha hızlı** çalışmakta ve spesifik metin sınıflandırma görevlerinde biçimsel yapıyı koruyarak daha yüksek doğruluk ve güvenilirlik sunmaktadır.

---

## 📂 Veri Seti Kapsamı ve Departmanlar
Veri seti 5 temel belediye departmanına ait mesajları içerir:
* **Temizlik ve Çöp**
* **Park ve Bahçeler**
* **Su ve Kanalizasyon**
* **Ulaşım ve Trafik**
* **Zabıta**

Veri setinde bulunmayan konular (ör. Kültür/Sanat, Etkinlik) için MLP modeli **%60 Güven Eşiği** mekanizması sayesinde belirsiz kararları saptayarak canlı temsilciye yönlendirmektedir.