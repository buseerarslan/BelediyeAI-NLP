# 🏛️ BelediyeAI: Yapay Zekâ Destekli Müşteri Mesajları Yönlendirme Sistemi

**BelediyeAI**, vatandaşlardan gelen serbest metin bildirimlerini (şikayet, talep, istek) Doğal Dil İşleme (NLP) ve Yapay Sinir Ağları (ANN) kullanarak ilgili belediye departmanlarına otomatik olarak yönlendiren modüler bir yapay zekâ mimarisidir.

---

## 🧩 Proje Mimari Adımları ve Mantığı

Projemiz 4 ana derin öğrenme adımı üzerine kurgulanmıştır:

### 1. Tokenization ve Bağlam Penceresi (Context Window)
* **Mantık:** GPT-2 tokenizer'ı kullanılarak Türkçe mesajların parçalama karakteristiği incelenmiştir[cite: 1, 3].
* **Gözlem:** GPT-2 modeli İngilizce ağırlıklı eğitildiği için Türkçe kelimeleri eklerine ve küçük alt birimlerine (subwords) böler[cite: 1]. Bu nedenle 100 mesajın toplam token sayısı GPT-2'nin 1024 token'lık bağlam penceresini aşmaktadır[cite: 1, 3].

### 2. Anlamsal Vektörleştirme (Sentence Embeddings)
* **Mantık:** `paraphrase-multilingual-mpnet-base-v2` çok dilli modeli ile mesajlar 768 boyutlu bir anlam uzayına taşınmıştır[cite: 1, 3].
* **Gözlem:** İçinde hiç ortak kelime geçmeyen ancak aynı anlamı taşıyan mesajlar (örneğin *'Çöp birikti'* ile *'Sokaklar süpürülmüyor'*) vektör uzayında birbirine çok yakın konumlanır[cite: 1, 2].

### 3. Derin Öğrenme Sınıflandırıcısı
* **Mimarisi:** 768 Girdi -> 32 Nöronlu ReLU Katmanı -> 5 Nöronlu Softmax Çıkış Katmanı[cite: 1, 3].
* **Sonuç:** Model %80 eğitim verisiyle eğitilmiş, hiç görmediği %20'lik test verisinde yüksek doğruluk oranına ulaşmıştır[cite: 3].

### 4. Akıllı Güven Eşiği (Thresholding)
* **Mantık:** Karar mekanizmasında %60 güven eşiği tanımlanmıştır[cite: 3].
* **Gözlem:** İki departmanı birden ilgilendiren belirsiz mesajlarda (örn: *"Parkta patlamış su borusu var"*) model olasılığı böldüğü için güven skoru %60'ın altında kalır ve mesaj hatalı yönlendirilmek yerine otomatik olarak **'Temsilciye aktar'** çıktısını verir[cite: 3].

---

## 💻 Projeyi Çalıştırma

```bash
# 1. Sanal Ortamı Aktifleştirin
.\venv\Scripts\activate

# 2. Projeyi Çalıştırın
python belediye_ai.py