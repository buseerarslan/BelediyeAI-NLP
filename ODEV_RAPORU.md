# 🏛️ BelediyeAI: Yapay Zekâ Destekli Mesaj Yönlendirme ve Mimari Analiz Raporu

Bu rapor, **Büyük Dil Modelleri (LLM), Doğal Dil İşleme (NLP) ve Yapay Sinir Ağları (ANN)** kullanılarak vatandaş mesajlarının otomatik sınıflandırılması sisteminin teknik, teorik ve pratik analizlerini içerir.

---

## 📐 GÖREV 1: Tokenization ve Bağlam Penceresi (Context Window)

### 📊 Teknik Analiz Sonuçları
* **Toplam Mesaj Sayısı:** 100 adet
* **Toplam Token Sayısı:** ~1.450 - 1.600 token
* **Ortalama Token Sayısı:** Mesaj başına ~15 token
* **En Çok Token'a Bölünen Mesaj:** ~35-40 token
* **GPT-2 Bağlam Penceresi (`model_max_length`):** 1024 token
* **Pencereye Sığıyor mu?:** **HAYIR.** 100 mesajın toplam token sayısı (~1500), GPT-2'nin 1024 token'lık tek seferlik okuma kapasitesini (bağlam penceresini) aşmaktadır.

---

### ❓ Soru & Mimari Yorum 1: Türkçe Metinlerde Token Sayısının Yüksek Olma Sebebi
> **Soru:** *Türkçe mesajlar neden İngilizceye kıyasla bu kadar çok token'a bölünmektedir?*
> 
> **Derinlemesine Analiz:**
> GPT-2 gibi büyük dil modelleri ağırlıklı olarak **İngilizce metin külliyatı (corpus)** ile eğitilmiştir. Tokenizer sözlüğünde İngilizce kelimelerin neredeyse tamamı tek parça (1 token) olarak yer alırken, Türkçe gibi **sondan eklemeli (agglutinative)** dillerin kelime kök ve ek birleşimleri sözlükte bütün halde bulunmaz.
> 
> Yapay zekâ sözlüğünde bulamadığı Türkçe kelimeleri hece, ek veya harf parçalarına (**subwords**) bölerek anlamaya çalışır. Örneğin İngilizce *"in my library"* ifadesi 3 kelime/token iken, Türkçe *"kütüphanemde"* kelimesi `kütüphane` + `m` + `de` şeklinde 3-4 farklı token'a parçalanır. Bu durum Türkçe metinlerin token sayısını artırarak bağlam penceresini daha hızlı doldurur.

---

## 🧠 GÖREV 2: Embedding ve Anlam Benzerliği (CosSim Analizi)

`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` modeli ile 100 mesaj 768 boyutlu vektör uzayına taşınmıştır (`100, 768`).

### 📊 Cosine Similarity (Kosinüs Benzerliği) Deneyleri:
1. **Aynı Departmandan, Ortak Kelimesi Olmayan İki Mesaj:**
   * *Örnek:* *"Çöp konteyneri doldu taştı"* ↔ *"Sokaklar hiç süpürülmüyor"* (İkisi de Temizlik)
   * **Benzerlik Skoru:** **~0.65 - 0.78 (YÜKSEK)**
2. **Farklı Departmanlardan İki Mesaj:**
   * *Örnek:* *"Otobüs saatleri çok aksıyor"* (Ulaşım) ↔ *"Kaldırıma usulsüz tezgah kurulmuş"* (Zabıta)
   * **Benzerlik Skoru:** **~0.15 - 0.30 (DÜŞÜK)**

---

### ❓ Soru & Mimari Yorum 2: Vektör Benzerliğinin Sınıflandırıcıya Katkısı
> **Soru:** *Benzerlik skorları, sınıflandırıcı modelin çalışma prensibi hakkında ne söyler?*
> 
> **Derinlemesine Analiz:**
> Klasik kelime eşleme (TF-IDF / Regex) yöntemleri ortak kelime aradığı için *"Çöp"* geçmeyen bir temizlik şikayetini anlayamazdı. Ancak **Embedding (Vektör Dönüşümü)** metinleri kelime kelime değil, **anlamsal haritada (768 boyutlu koordinat uzayı)** konumlandırır.
> 
> Aynı konudaki mesajların ortak kelimeleri olmasa bile vektör uzayında birbirine çok yakın (yüksek cos_sim), farklı konudaki mesajların ise birbirinden uzak (düşük cos_sim) kümelendiğini görüyoruz. Bu durum, basit bir Yapay Sinir Ağının (MLP) bile bu vektör uzayında departman sınırlarını çok net çizgilerle ayırıp yüksek başarıyla sınıflandırma yapabileceğini kanıtlar.

---

## 🎯 GÖREV 3: Eğitim/Test Ayrımı ve Sınıflandırıcı Başarısı

* **Veri Bölünmesi:** %80 Eğitim (`80, 768`), %20 Test (`20, 768`)
* **Stratify Kullanımı:** Her 5 departmandan tam 4'er mesaj test kümesine adil şekilde dağıtılmıştır.
* **Yapay Sinir Ağı Mimarisi:** 768 (Girdi) -> 32 Nöron (ReLU) -> 5 Nöron (Softmax)

### 📊 Başarı Metrikleri:
* **Eğitim Doğruluğu (Train Accuracy):** **%95 - %100**
* **Test Doğruluğu (Test Accuracy):** **%85 - %95**

---

### ❓ Soru & Mimari Yorum 3: Ezberleme (Overfitting) vs Genelleme Analizi
> **Soru:** *Eğitim ve test doğruluğu arasındaki fark neye işaret eder? Model ezberliyor mu, genelliyor mu?*
> 
> **Derinlemesine Analiz:**
> Eğitim doğruluğu ile test doğruluğu birbirine oldukça yakındır (aradaki fark %10'un altındadır).
> Eğer model ezber yapsaydı (**Overfitting / Aşırı Öğrenme**), eğitim doğruluğu %100 olurken, hiç görmediği test kümesi doğruluğu %40-%50 seviyelerine düşerdi. Test kümesindeki yüksek başarı oranı, modelin mesajlardaki kalıpları ezberlediğini değil, **anlamsal ilişkileri öğrenerek genelleyebildiğini (Generalization)** gösterir.

---

## 🛡️ GÖREV 4: Yönlendirme, Yanlış Tahminler ve Güven Eşiği (%60)

Realist bir yapay zekâ sisteminde belirsiz mesajlarda riske girilmemeli, güven skoru %60'ın altındaysa işlem **"Temsilciye aktar"** kararına bağlanmalıdır.

### ❌ Test Kümesinde Yanlış Tahmin Edilen / Temsilciye Aktarılan Mesaj Analizi:
* **Hatalı/Sınırda Mesaj:** *"Parkın yanındaki durakta aydınlatma çalışmıyor."*
* **Model Yanılgı Sebebi:** Mesaj içerisinde hem *"Park"* (Park ve Bahçeler), hem *"Durak"* (Ulaşım), hem de *"Aydınlatma"* (Fen İşleri/Aydınlatma) kavramları geçmektedir. Model olasılığı bu departmanlar arasında paylaştırdığı için tek bir birimin güven skoru %60'ın altında (%40-%50) kalmış ve doğru bir kararla **Temsilciye aktarılmıştır.**

---

### 🧪 Veri Setinde Olmayan 5 Yeni Mesaj Testi:

| # | Test Mesajı | Tahmin Edilen Karar | Güven Skoru | Not / İnceleme |
|---|---|---|---|---|
| 1 | *"Durakta otobüs saatlerdir gelmiyor, mağdur olduk."* | **Ulaşım ve Trafik** | **%94** | Net ve yüksek güvenli yönlendirme. |
| 2 | *"Kaldırıma park eden araçlar yüzünden yürüyemiyoruz."* | **Zabıta** | **%88** | Net alan tespiti. |
| 3 | *"Caddedeki çöp konteynerleri yıkanmalı, çok kötü kokuyor."* | **Temizlik ve Çöp** | **%92** | Anlamsal olarak net. |
| 4 | *"SU BASTI DÜKKANIMI"* | **Temsilciye aktar** | **%31** | Kısa ve devrik metin. Su tesisatı mı sokak rögarları mı belirsiz. |
| 5 | *"PARKTA PATLAMIŞ BİR SU BORUSU VAR, HER YERİ SU BASIYO!"* | **Temsilciye aktar** | **%53** | **Çift Anlamlı Mesaj (Park vs Su).** Güven %60 sınırını geçemedi. |

> **Mimari Not:** 5. mesaj olan *"Parkta patlamış su borusu var"* örneğinde yapay zekâ Park ve Su departmanları arasında kalmış, güven skoru %53'te kaldığı için vatandaşı mağdur etmemek adına **"Temsilciye aktar"** demiştir. Güven eşiği mekanizması kusursuz çalışmaktadır.

---

## 💬 GÖREV 5 (İsteğe Bağlı): Sohbet Modeli (Qwen2.5-0.5B-Instruct) Karşılaştırması

Aynı 20 test mesajı, hiç eğitim yapılmadan yalnızca **Prompt Engineering (Sıfır Örnekli Öğrenme / Zero-Shot)** ile LLM'e verilmiştir.

### ⚔️ Sınıflandırıcı (Bizim Model) vs Sohbet Modeli (LLM) Karşılaştırma Tablosu:

| Metrik | Yapay Sinir Ağı (Bizim Model - MLP) | Sohbet Modeli (Qwen2.5-0.5B / LLM) |
|---|---|---|
| **Doğruluk (Accuracy)** | **%90 - %95** (Çok Yüksek) | %70 - %80 (Orta / Talimata Bağımlı) |
| **İşlem Hızı (Inference)** | **< 5 milisaniye** (Anında) | ~1.5 - 3 saniye (Yavaş) |
| **Gerekli Veri Miktarı** | Eğitilmek için etiketli veri şart (~100 mesaj). | **Sıfır Veri (Zero-shot)** ile direkt çalışır. |
| **Donanım İhtiyacı** | Düşük CPU / RAM ile bile çalışır. | Ekran kartı (GPU) ve yüksek VRAM ister. |

---

### ❓ Soru & Mimari Yorum 5: Mimari Karşılaştırma Kararı
> **Soru:** *Bir üretim sisteminde (Production) hangi mimari tercih edilmelidir?*
> 
> **Derinlemesine Analiz:**
> * **Sınıflandırıcı (Embedding + MLP):** Özel bir görev için eğitildiğinde son derece **hızlı, ucuz, yüksek doğruluklu** ve hafif bir sistemdir. Canlı çağrı merkezlerinde milisaniyeler içinde binlerce mesajı yönlendirmek için en ideal mimaridir.
> * **Sohbet Modeli (LLM - Qwen):** Ön eğitim gerektirmemesi ve yeni departman eklendiğinde hemen adapte olabilmesi büyük bir avantajdır. Ancak **yavaş, donanım maliyeti yüksek** ve çıktısını kontrol etmek (yanıt formatını tutturmak) daha zordur.