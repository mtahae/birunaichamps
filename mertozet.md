# BirunAI - CardioFusion-5 Proje Özeti ve Kapsamlı Geliştirici Rehberi

Selam Mert ve AI Asistanı! 👋

Bu doküman, TEKNOFEST 2026 Sağlıkta Yapay Zeka yarışması için geliştirdiğimiz **CardioFusion-5** modelinin bugüne kadarki gelişim sürecini, dosya yapısını, denediğimiz taktikleri ve ulaştığımız "Anti-Overfitting" mimarisini tüm şeffaflığıyla açıklamak için hazırlanmıştır. Biz bu aşamaya kadar veri işleme, model mimarisi tasarımı ve eğitim boru hattı (pipeline) üzerinde ciddi bir temel kurduk. Bu rehber sayesinde projeyi hızlıca anlayıp bizim bıraktığımız noktadan paralel olarak geliştirmeye devam edebilirsiniz.

> **ÖNEMLİ NOT:** Bu rehber, modelin temel güçlü mimarisini ve overfit'i engelleme yöntemlerini açıklar. Geliştirmelere tam olarak bu felsefe üzerinden devam etmeniz beklenmektedir.

---

## 1. Projenin Amacı ve Temel Hedefler

**Hedef:** 12-lead (derivasyon) EKG sinyallerini kullanarak 5 ana sınıfı tespit eden ve klinik olarak anlamlı sonuçlar üreten bir derin öğrenme modeli geliştirmek.
Sınıflarımız:
1. Normal EKG
2. AFIB (Atriyal Fibrilasyon)
3. AFL (Atriyal Flutter)
4. LBBB (Sol Dal Bloğu)
5. RBBB (Sağ Dal Bloğu)

**Kritik Kural:** Başarımızı `Accuracy` (Doğruluk) belirlemez! Veri setinde "Normal" sınıfı çok baskın olduğu için Accuracy son derece yanıltıcıdır. Bizim için yegane optimizasyon hedefi ve gerçek performans metriği **Macro F1 Score**'dur. Tüm loss fonksiyonlarımız ve eşik (threshold) ayarlamalarımız Macro F1'i maksimize etmek üzerine kuruludur.

---

## 2. Proje Klasör ve Dosya Yapısı (Kılavuz)

Projedeki her bir kod dosyasının (script) belirli ve tek bir sorumluluğu vardır. İşte adım adım dosya yapımız ve işlevleri:

*   **`config.py`:** Projenin kalbi ve beyin sapıdır. Dosya yolları, hiperparametreler (Learning Rate, Epoch sayıları, Batch Size), sınıf ağırlıkları ve model mimarisinin sabit boyutları tek bir merkezden buradan yönetilir. Kodun hiçbir yerinde "hardcoded" değer bulunmaz, her şey `config`'den çekilir.
*   **`adim00_veri_birlestirme.py`:** Farklı kaynaklardan (TEKNOFEST verisi, PTB-XL, CPSC vb.) gelen farklı formatlardaki EKG verilerini standart bir formata (NumPy `.npy` ve tek bir global CSV manifestosu) dönüştürür.
*   **`adim01_kalite_kontrol_genel.py` & `adim01_veri_yukleme.py`:** EKG sinyallerini diske yüklemeden önce bozuk, sinyal içermeyen veya gürültüden ibaret olan "çöp" verileri ayıklamak için ilk kalite kontrol testlerini yapar.
*   **`adim02_filtreleme.py`:** Ham EKG sinyallerini işler. Sinyalleri 250 Hz'e resample eder ve Butterworth Bandpass filtre (0.5 Hz - 40 Hz) uygulayarak şebeke gürültüsü ve taban kaymalarını temizler.
*   **`adim03_kalite_kontrol.py`:** Filtrelenmiş sinyallerin (SQI - Signal Quality Index) metriklerine bakarak klinik olarak tanı koyulamayacak kadar kötü olanları veri setinden dışlar.
*   **`adim04_segmentasyon.py`:** Uzun veya kısa EKG kayıtlarını modelin beklediği sabit 10 saniyelik (2500 time-step) pencerelere böler (Kısa olanlara zero-padding, uzunlara center-crop uygular).
*   **`adim05_ozellik_cikarma.py` & `adim07b_wide_features.py`:** Sinyalden geleneksel istatistiksel ve morfolojik özellikleri (kurtosis, skewness, bandpower, dominant frekans) çıkararak derin öğrenme modeline "Wide Features" (Geniş Özellikler) olarak beslenmek üzere hazırlar.
*   **`adim06_veri_bolme.py` & `adim06b_oversampling.py`:** Veriyi Train/Validation/Test olarak böler. **Çok Kritik:** Bölme işlemi "Hasta Bazlı (Per-Patient)" yapılır. Aynı hastanın iki farklı EKG'si asla iki farklı sete düşmez. Ayrıca sınıflar arası dengesizliği gidermek için oversampling teknikleri barındırır.
*   **`adim07_model_mimarisi.py`:** CardioFusion-5 modelinin PyTorch ile yazılmış tam mimarisini (CNN + Transformer + Wide Features + DANN + Multi-Task Aux Başlıkları) içerir. Ayrıca Dataset sınıfları, veri augmentasyon algoritmaları ve özel kayıp (loss) fonksiyonları bu dosyada tanımlıdır.
*   **`adim08_egitim.py`:** Tüm eğitim döngüsünü (Training Loop) yönetir. 3 Fazlı Curriculum Learning, Early Stopping, Validation doğrulaması, ve eğitim sonu eşik (threshold) optimizasyonunu gerçekleştirir. Eğitimi başlatacağımız dosyadır.
*   **`adim09_degerlendirme.py`:** Eğitilmiş modelin test seti üzerindeki final Macro F1 skorunu, karışıklık matrisini (confusion matrix) ve sınıf bazlı performansını detaylıca raporlar.
*   **`adim10_gradcam.py`:** Modelin karar verirken sinyalin neresine (hangi dalgalara) baktığını görselleştirmek için XAI (Explainable AI) tekniklerini içerir.
*   **`dashboard.py` & `baslat.py`:** Eğitimi canlı olarak terminalde görsel ve renkli bir şekilde takip etmemizi sağlayan CLI arayüzleri.
*   **`threshold_opt.py`:** Model çıktılarını (0-1 arası olasılıklar) doğrudan 0.5 eşiği ile kesmek yerine, her sınıf için Macro F1 skorunu maksimize edecek optimal eşikleri (örn: Normal=0.35, RBBB=0.55) bulan logik dosyası.

---

## 3. Veri Ön İşleme (Preprocessing) Stratejimiz

Fizyolojik sinyal işlemede yapılan ufacık bir hata, modelin hiçbir şey öğrenememesine veya veriyi "ezberlemesine" neden olur. PhysioNet yarışmacılarının (Triage, HeartBeats vb.) stratejilerinden ilham alarak şu katı kuralları uyguladık:

1.  **Sabit Örnekleme (Resampling):** Tüm veriler (PTB-XL, CPSC, TEKNOFEST) orijinal frekanslarından bağımsız olarak **250 Hz'e** sabitlendi. Bu işlem 125 Hz altı kritik klinik bileşenleri korurken, GPU VRAM tüketimini ve hesaplama yükünü dramatik şekilde azaltır.
2.  **Butterworth Filtreleme:** Sinyallere 0.5 Hz - 40 Hz aralığında 4. Derece Butterworth Bandpass filtre uyguladık.
    *   *Neden 0.5 Hz?* Hastanın nefes almasından kaynaklı taban çizgisi kaymasını (baseline wander) ortadan kaldırır.
    *   *Neden 40 Hz?* Kas seğirmesi (EMG) ve şebeke elektriği (50 Hz / 60 Hz) gürültülerini filtreler.
    *   *Neden `filtfilt`?* Sinyale ileri-geri filtreleme yaparak faz kaymasını engeller. Bu sayede P-QRS-T dalgalarının zamanlaması bozulmaz.
3.  **Z-Score Normalizasyonu (ÇOK KRİTİK):**
    *   **Global Normalizasyon ASLA yapılmadı.** Tüm lead'leri aynı ortalamaya bölmek yasaktır. Çünkü göğüs derivasyonlarının (V1-V6) fizyolojik genliği ekstremite derivasyonlarından farklıdır.
    *   Sadece **Lead-Wise** (her lead kendi içinde) standartlaştırma (z-score) uygulandı.
    *   **Data Leakage Önlemi:** İstatistikler (Ortalama ve Standart Sapma) SADECE eğitim (Train) setinden hesaplanıp, Val ve Test setlerine doğrudan dışarıdan aktarıldı.
4.  **Data Split Data Leakage Önlemi:** Aynı hastaya ait birden fazla EKG kaydı bulunuyorsa, modelin hastanın kalp anatomisini ezberlememesi için bölme işlemi kesinlikle "Hasta ID" üzerinden (GroupKFold mantığıyla) yapıldı.

---

## 4. Model Mimarisi: CardioFusion-5

Sadece derin bir CNN kullanmak kompleks aritmiler için yetersizdir. Bu yüzden hibrit ve çok dallı bir yapı kurguladık:

1.  **CNN Backbone (1D SE-ResNet):** Sinyalin içindeki lokal ve hızlı değişen morfolojileri (QRS genişliği, P dalgası şekli vb.) çıkarır. Squeeze-and-Excitation (SE) blokları ile hangi derivasyonun o anki hastalık için daha önemli olduğuna karar verir (Örn: RBBB tespiti için ağı Lead V1 ve V2'ye odaklanmaya zorlar).
2.  **Transformer Encoder:** CNN'den çıkan sıralı özellikleri (sequence) alır. Zamansal ilişkileri (örneğin P dalgası ile QRS arasındaki uzun vadeli PR mesafesi gibi) çoklu başlıklarla (Multi-Head Attention) öğrenir.
3.  **Wide Features (Geniş Özellikler):** Yaş, cinsiyet ve `adim07b_wide_features.py` ile çıkardığımız sinyal kalite/şekil metrikleri modelin "Deep" (derin) özellikleriyle birleştirilip son sınıflandırıcıya aktarılır.
4.  **DANN (Domain-Adversarial Neural Network):** Modelin, internetten indirdiğimiz devasa açık kaynaklı veri setleri (Domain 1) ile TEKNOFEST'in kendi kısıtlı verisi (Domain 2) arasındaki uçurumu hissetmesini engeller. Özellik çıkarıcı, Gradient Reversal Layer (GRL) sayesinde iki domain arasındaki farkı gizlemeyi öğrenir.
5.  **Multi-Task Classification Başlıkları:** Model sadece ana 5 sınıfı tahmin etmez. Aynı zamanda yan görev olarak EKG'nin genel durumunu (Normal mi? İletim Bozukluğu mu? Ritim Bozukluğu mu?) 3 sınıfta tahmin etmeye zorlanır. Bu "Auxiliary Loss", modelin öğrendiği feature'ların kalitesini ciddi şekilde artırır.

---

## 5. Eğitim Boru Hattı (Curriculum Learning) ve Anti-Overfitting

Modelin veriyi ezberlemesini önlemek (Overfitting) için eğitimi 3 Faza böldük. Bu bizim en güçlü taktiklerimizden biridir:

*   **Phase 1 (Sadece TEKNOFEST):** Model ilk başta sadece hedef yarışma verisiyle hızlıca "Isınma (Warmup)" yapar ve domaini tanır.
*   **Phase 2 (Karma İnternet Verisi + Mixup + DANN):** Model tüm harici veri setleriyle genel kardiyoloji kütüphanesini oluşturur. Bu fazda Mixup augmentasyonu aktifleşerek farklı hastaların EKG'lerini karıştırır ve modelin karar sınırlarını yumuşatır.
*   **Phase 3 (Genelleme ve İnce Ayar):** Öğrenme oranı çok düşürülür. Bu fazda model son ince ayarlarını yapar.

### Kullandığımız Loss (Kayıp) Fonksiyonları
*   **Focal Loss:** Normal sınıfı veri setini domine ettiği için standart `CrossEntropy` yerine `FocalLoss` kullandık. Böylece model zaten çok iyi öğrendiği kolay sınıflar (Normal) için loss üretmeyi bırakıp, zor olan (AFL) gibi sınıflara odaklanır.
*   **Multi-Task Loss:** Toplam Loss = Main_Loss + 0.3 * Aux_Loss + DANN_Loss olarak kurgulanmıştır.

### Overfitting Krizi ve Kalkanlarımız
Eğitim sırasında train loss'un sıfıra yaklaşıp validation loss'un patlaması (ezberleme) sorununa karşı şu kalkanları koda entegre ettik:

1.  **Phase 3 Backbone Freeze (Omurga Dondurma):** 3. Fazda devasa CNN omurgasının ağırlıklarını (`requires_grad=False`) tamamen donduruyoruz. Sadece Transformer ve Classifier eğitiliyor. Bu sayede model genel "EKG okuma" şablonunu unutup spesifik seti ezberlemiyor.
2.  **Time Masking Augmentation:** Sinyalin rastgele bir zaman dilimini (örneğin 0.5 saniyelik bir kısmı) tamamen sıfırlıyoruz. Bu, modelin sinyalin belli bir noktasındaki spesifik bir gürültüyü referans almasını engeller.
3.  **Spatial Dropout:** CNN feature map'lerine kanal bazlı %10 dropout uyguluyoruz.
4.  **Wide Features Dropout & BatchNorm:** Yaş/cinsiyet gibi özelliklerin modelin kararını gereğinden fazla domine etmesini engellemek için bu özelliklere classifier'a girmeden önce yüksek oranda Dropout uyguluyoruz.
5.  **Akıllı Early Stopping:** P2 aşamasında patience oldukça katı tutuldu. Ayrıca "Val Loss Divergence Check" ekledik: Eğer Val Loss, kaydedilen en iyi loss'un 2.5 katına fırlarsa, F1 artsa dahi o faz derhal kesiliyor.
6.  **SWA (Stochastic Weight Averaging):** P3'ün sonunda modelin son epoch'lardaki ağırlıklarının ortalaması alınarak daha geniş ve daha genellenebilir bir "düz minima (flat minima)" elde ediliyor.

---

## 6. Geliştiriciler İçin To-Do (Sonraki Adımlar)

Model mimarisi ve pipeline'ın temelleri şu an harika bir şekilde atıldı. Siz projeyi devam ettirirken şu başlıklara odaklanmalısınız:

1.  **Ensemble Stratejisi (Şiddetle Önerilir):**
    *   Tek bir model hiçbir zaman yeterli olmaz. Modeli farklı `seed` değerleriyle (örn: 42, 100, 2026 vb.) baştan sona 3 kere eğitin.
    *   Tahminleme aşamasında (Inference), bu 3 modelin ürettiği olasılıkları (Softmax) basitçe toplayıp ortalamasını (Soft Voting) alın. Sadece bu basit taktik, F1 skorunu tek kalemde %2-%3 civarında artıracaktır.
2.  **Daha Agresif Test-Time Augmentation (TTA):**
    *   Şu an pipeline'ımızda 5x TTA (1 orijinal + 4 augmentasyon: zaman çevirme, genlik ölçekleme vb.) bulunuyor. Çıkarım sırasında bu sayıyı ve çeşitliliği artırabilirsiniz.
3.  **İleri Özellik Mühendisliği (Feature Engineering):**
    *   `adim07b_wide_features.py` dosyasına daha karmaşık sinyal kalitesi metrikleri ve frekans uzayı özellikleri ekleyerek Transformer'ın yükünü hafifletebilirsiniz.

**Son Not:** `adim07_model_mimarisi.py` ve `adim08_egitim.py` projeyi bir arada tutan kalptir. Kodların içindeki detaylı yorum satırlarını okumanız mimari kararlarımızı daha iyi anlamanızı sağlayacaktır.

Başarılar, şampiyonluğa adım adım! 🚀
