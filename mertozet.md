# BirunAI - CardioFusion-5 Proje Özeti ve Strateji Rehberi

Selam Mert ve AI Asistanı! 👋

Bu doküman, TEKNOFEST 2026 Sağlıkta Yapay Zeka yarışması için geliştirdiğimiz **CardioFusion-5** modelinin bugüne kadarki gelişim sürecini, denediğimiz taktikleri, yarışma jürisinin beklentilerini ve son ulaştığımız "Anti-Overfitting" mimarisini detaylıca açıklamak için hazırlanmıştır. Biz bu aşamaya kadar veri işleme, model mimarisi tasarımı ve eğitim boru hattı (pipeline) üzerinde çalıştık. Buradan sonrasını siz paralel olarak devam ettirebilirsiniz.

Aşağıda her bir bileşeni, *neden* o şekilde yaptığımızın felsefesiyle birlikte bulacaksınız.

---

## 1. Projenin Amacı ve Kapsamı
**Hedef:** 12-lead (derivasyon) EKG sinyallerini kullanarak 5 ana sınıfı tespit etmek:
1. Normal EKG
2. AFIB (Atriyal Fibrilasyon)
3. AFL (Atriyal Flutter)
4. LBBB (Sol Dal Bloğu)
5. RBBB (Sağ Dal Bloğu)

**Kritik Kural:** Başarımızı `Accuracy` (Doğruluk) belirlemez! Veri setinde "Normal" sınıfı çok baskın olduğu için Accuracy yanıltıcıdır. Bizim için tek ve yegane gerçek metrik **Macro F1 Score**'dur.

---

## 2. Veri Ön İşleme (Preprocessing) Stratejimiz

Fizyolojik sinyal işlemede yapılan ufacık bir hata, modelin hiçbir şey öğrenememesine neden olur. PhysioNet yarışmacılarının (Triage, HeartBeats vb.) stratejilerinden ilham alarak şu katı kuralları uyguladık:

*   **Sabit Örnekleme (Resampling):** Tüm veriler (PTB-XL, CPSC vb.) 500 Hz'den **250 Hz'e** düşürüldü. (Hem klinik olarak 125 Hz altı bileşenleri korur hem de VRAM tüketimini yarıya indirir).
*   **Filtreleme:** 0.5 Hz - 40 Hz aralığında 4. Derece Butterworth Bandpass filtre kullandık.
    *   *Neden 0.5 Hz?* Hastanın nefes almasından kaynaklı taban çizgisi kaymasını (baseline wander) siler.
    *   *Neden 40 Hz?* Kas seğirmesi (EMG) ve şebeke elektriği (50 Hz) gürültülerini siler.
    *   *Neden filtfilt?* İleri-geri filtreleme yaparak faz kaymasını engeller, böylece QRS dalgasının zamanlaması bozulmaz.
*   **Segmentasyon:** Sabit 10 saniyelik (2500 time-step) pencereler. Kısa olanlara zero-padding, uzun olanlara center-crop yapıldı.
*   **Z-Score Normalizasyonu (ÇOK KRİTİK):**
    *   **Asla Global Normalizasyon yapmadık!** (Tüm lead'leri aynı ortalamaya bölmek yasak). Çünkü V1'in genliği ile V5'in genliği fizyolojik olarak farklıdır.
    *   Sadece **Lead-Wise** (her lead kendi içinde) normalizasyon uyguladık.
    *   **Data Leakage Önlemi:** Z-Score istatistikleri (mean, std) SADECE Train setinden hesaplanıp, Val ve Test setlerine dışarıdan uygulandı.
*   **Hasta Bazlı Bölme (Per-Patient Split):** Aynı hastanın farklı günlerde çekilmiş 2 EKG'si varsa, biri train diğeri test setine DÜŞEMEZ. Aksi takdirde model hastayı ezberler. Veriler hasta ID'lerine göre bölündü.

---

## 3. Model Mimarisi: CardioFusion-5

Sadece derin bir CNN kullanmak EKG için yetersizdir. Biz hibrit bir mimari kurduk:

1.  **CNN Backbone (1D SE-ResNet):** Sinyalin içindeki lokal ve hızlı değişen morfolojileri (QRS genişliği, P dalgası şekli vb.) çıkarır. Squeeze-and-Excitation (SE) blokları ile hangi derivasyonun (lead) o an daha önemli olduğuna karar verir (Örn: RBBB için Lead V1'e daha çok dikkat eder).
2.  **Transformer Encoder:** CNN'den çıkan özellikleri alır. Zamansal ilişkileri (P dalgası ile QRS arasındaki PR mesafesi gibi uzun vadeli ilişkileri) öğrenir.
3.  **Wide Features:** Yaş, cinsiyet, kalp atım hızı (HR) ve `adim07b_wide_features.py` ile çıkardığımız sinyal kalite/şekil metrikleri (kurtosis, skewness, bandpower). Bu özellikler doğrudan sınıflandırıcıya (Classifier) beslenir.

---

## 4. Eğitim Boru Hattı (Training Pipeline) ve Curriculum Learning

Eğitimi 3 faza (P1, P2, P3) böldük. Bu bizim gizli silahımız:

*   **Phase 1 (Sadece TEKNOFEST):** Model önce hedef domaini (yarışma verisini) tanır.
*   **Phase 2 (Karma İnternet Verisi + TEKNOFEST):** Model çok büyük verilerle (PTB-XL vs.) genel kardiyoloji bilgisini öğrenir.
    *   *DANN (Domain-Adversarial Neural Network):* Bu aşamada Gradient Reversal Layer ile modelin "Bu veri internetten mi yoksa TEKNOFEST'ten mi geldi?" ayrımını yapamamasını sağlıyoruz. Böylece domain shift (veri seti uyuşmazlığı) engelleniyor.
    *   *Mixup Augmentation:* Farklı hastaların EKG'lerini karıştırarak modelin karar sınırlarını yumuşatıyoruz.
*   **Phase 3 (Genelleme ve Fine-Tuning):** P2'den sonra düşük LR ile ince ayar. Burada **SWA (Stochastic Weight Averaging)** devreye girer.

### Loss Fonksiyonlarımız
*   **Ana Loss (Focal Loss + Label Smoothing):**
    *   Normal sınıfı çok fazla olduğu için `CrossEntropy` yerine `FocalLoss` kullandık (Kolay öğrenilen "Normal" sınıfının loss etkisini azaltır, zor olan "AFL" sınıfına odaklandırır).
    *   `Label Smoothing (0.10)`: Modelin "%100 eminim" demesini engeller.
*   **Multi-Task Loss:** Modeli sadece ana sınıfları değil, üst sınıfları da (Ritim bozukluğu mu, iletim bozukluğu mu?) tahmin etmeye zorlar. Bu, feature'ların kalitesini artırır. Toplam Loss = Main + 0.3 * Aux.

---

## 5. Overfitting Krizi ve Çözümlerimiz (Son Revizyonlar)

Modelin bir noktada ezbere kaydığını (Train loss ≈ 0, Val Loss fırlaması) fark ettik. Buna karşı geliştirdiğimiz ve güncel kodda (adım 7 ve 8) aktif olan "Anti-Overfitting" kalkanımız:

1.  **P3 Backbone Freeze:** 3. Fazda (Fine-Tuning) devasa CNN omurgasını tamamen donduruyoruz. Sadece Transformer ve Classifier eğitiliyor. Böylece model internet verisinden öğrendiği "genel EKG okuma" yeteneğini unutup küçük veri setini ezberlemiyor.
2.  **Time Masking Augmentation:** Sinyalin rastgele bir zaman dilimini (örneğin 0.5 saniyelik bir kısmı) sıfırlıyoruz (SpecAugment mantığı). Model sinyalin belirli bir saniyesindeki küçük bir gürültüyü ezberleyemiyor.
3.  **Spatial Dropout:** CNN katmanlarındaki feature map'lere %10 dropout uyguladık.
4.  **Wide Features Normalizasyonu:** Modele verdiğimiz 8 adet "Wide Feature", BatchNorm ve Dropout(0.3) işleminden geçirilmeden ana classifier'a girmiyor. Aksi halde model sadece hastanın yaşına veya nabzına bakıp kestirme yapıyordu.
5.  **Güçlü Regularizasyon:** Weight Decay (5e-4), Label Smoothing (0.1) ve Augmentasyon olasılığı (%90) artırıldı.
6.  **Akıllı Early Stopping:** P2 aşamasında patience 15'e çekildi ve "Val Loss Divergence Check" eklendi (Eğer validation loss, en iyi loss'un 2.5 katına çıkarsa F1 skoru artsa bile o fazı derhal bitiriyor).

---

## 6. Sizin İçin "To-Do" Listesi ve Sonraki Adımlar

Model mimarimiz ve pipeline'ımız şu an çok sağlam. Siz paralel çalışırken şunlara odaklanabilirsiniz:

1.  **Ensemble Stratejisi (Çok Önemli):**
    *   Modeli farklı `seed` değerleriyle (örn: 42, 100, 2026) 3 kere baştan eğitin.
    *   Son çıkarımda (inference), bu 3 modelin ürettiği olasılıkları (softmax çıktılarını) ortalayın. PhysioNet birincileri bu taktikle F1 skorunu tek kalemde %2-3 artırdılar.
2.  **Test-Time Augmentation (TTA):**
    *   Şu an kodumuzda TTA var (1 orijinal + 4 augmentasyonlu sinyalin tahmini ortalanıyor). Farklı augmentasyon stratejileri eklenebilir.
3.  **Eşik (Threshold) Optimizasyonu:**
    *   Modelin en iyi F1 skorunu vermesi için `adim08_egitim.py` sonunda çalışan eşik optimizasyon algoritmasını inceleyin. "Normal" için 0.35, "RBBB" için 0.55 gibi eşikler macro F1'i çok etkiliyor.

Bütün kodlar detaylı yorum satırlarıyla dolu. `adim07_model_mimarisi.py` ve `adim08_egitim.py` dosyaları projenin kalbidir.

Başarılar, harika iş çıkaracağız! 🚀
