# EKG Zaman Serisi Sınıflandırma: 3 Aşamalı Mimari ve Eğitim Raporu

> [!NOTE]
> Bu belge, 12 derivasyonlu EKG sinyallerini kullanarak 5 farklı teşhis sınıfında (Normal, AFIB, AFL, LBBB, RBBB) görev alan **CardioFusion-5 v2** modelimizin tüm mimarisini, veri hiperparametrelerini ve eğitim stratejisini detaylandırmak amacıyla hazırlanmıştır. Başarısız veya tıkanmış (0.82-0.83 Macro F1 bandında kalan) EKG sınıflandırma projelerine teknik bir referans ve rehber olması hedeflenmektedir.

---

## 1. Veri Dağılımı ve Ön İşleme

### Sınıf Dağılımı ve Dengesizlik
Eğitim setimizdeki (Teknofest + Dış Kaynak) sınıfların genel dağılımı ciddi bir dengesizlik (class imbalance) göstermekteydi:
- **Normal (NSR):** ~8.634 örnek (%25.8) - *En baskın sınıf*
- **AFIB:** ~5.400 örnek (%16.4)
- **LBBB:** ~3.173 örnek (%9.5)
- **RBBB:** ~3.000 örnek (%9.0)
- **AFL:** ~1.500 örnek (%4.5) - *En nadir sınıf, tespit darboğazı*

### Sinyal Temizleme ve Segmentasyon
* **Frekans (Sampling Rate):** Orijinal 500 Hz sinyaller, hesaplama maliyetini düşürmek ve dış kaynak (250 Hz) verileriyle eşlemek için **250 Hz**'e düşürüldü (Resampling).
* **Segmentasyon Uzunluğu:** Tüm sinyaller tam **10 saniyelik** (2500 sample) sabit pencerelere bölündü veya ortadan (center-crop / zero-pad) sabitlendi. Örtüşme (overlap) kullanılmadı.
* **Filtreleme:** Şebeke gürültüsü ve hasta nefes almasından kaynaklı baseline wander'ı yok etmek için **0.5 - 40 Hz** aralığında 4. Derece **Butterworth Bandpass** (filtfilt) uygulandı.

### Data Augmentation (Veri Çoğaltma) Stratejisi
* **Lead-Wise Amplitude Scale:** Rastgele seçilen derivasyonların genliği 0.9x ile 1.1x arasında ölçeklendi (%60 ihtimal).
* **Gaussian Noise:** SNR > 20 dB olacak şekilde hafif Gauss gürültüsü (0.02 standart sapma) eklendi (%60 ihtimal).
* **Lead Dropout:** 12 derivasyonun 1 veya 2 tanesi tamamen sıfırlanarak cihaz temas kopuklukları simüle edildi (%30 ihtimal).
* **Baseline Wander:** Düşük frekanslı (0.1 - 0.5 Hz) sinüzoidal dalgalar eklenerek nefes alma simüle edildi (%50 ihtimal).
* **Time Masking:** SpecAugment'in zaman serisi versiyonu olarak 10-50 sample arası kısa rastgele zaman dilimleri sıfırlandı (%10 ihtimal).
* **Signal Inversion:** Kablo ters bağlanması (kalibrasyon) simülasyonu eklendi (%5 ihtimal).

---

## 2. Veri Bölme Stratejisi (Data Split)

* **Oranlar:** Eğitim süreci için Train/Val split oranı yaklaşık **%80 Train - %20 Validation** şeklinde uygulandı. (Tam 580 EKG kaydı sadece nihai Test/Ensemble Validation için izole bırakıldı).
* **Bölme Yöntemi:** Hastaların birden fazla teşhise sahip olabileceği (Multi-label) bir senaryo olduğu için basit rastgele bölme (train_test_split) YERİNE **MultilabelStratifiedKFold** kullanıldı. Bu sayede nadir sınıf olan AFL'nin Train ve Val setlerine eşit oranlarda düşmesi garanti altına alındı. (Aynı hastaya ait EKG'lerin hem train hem val'e düşmemesi için hasta ID'leri bazlı Group stratification gözetildi).

---

## 3. Model Mimarisi: CardioFusion-5 v2

Ana omurga, yerel ve global (zamansal) özellikleri birleştiren **CNN + Transformer** hibrit mimarisidir:

* **Ana Omurga (Backbone):** **SE-ResNet** (Squeeze-and-Excitation). Filtre sayıları aşamalı olarak `[64, 128, 256, 256, 384]` olarak tasarlandı. İlk katmanda QRS'i geniş görmek için kernel boyutu 15 kullanıldı.
* **Multi-Scale CNN:** Transformer'a girmeden hemen önce sinyal, **3, 7 ve 15** kernel boyutlarına sahip paralel 3 CNN dalından geçirildi (Hızlı değişimler ve uzun vadeli RR aralıklarını aynı anda okumak için).
* **Rhythm Branch:** Sadece **Lead II, III ve aVF** derivasyonlarını giriş alan ve spesifik olarak P-dalgasına odaklanan yalıtılmış bir CNN kolu (64 boyutlu çıktı).
* **Transformer Encoder:** 384 boyutlu CNN çıktılarını alıp, zaman içindeki kalp atışlarının birbirleriyle olan ilişkilerini çözen 2 katmanlı (8 Head) Transformer. Modelin çıktısı Global Average Pooling yerine **Attention Pooling** ile zaman ekseninde daraltıldı.
* **Wide Features (Klinik Özellikler):** Yaş, Cinsiyet, ve `NeuroKit2` ile hesaplanmış 4 P-Dalgası özelliği (rr_cv, p_present, p_regularity, atrial_rate) dahil olmak üzere toplam **12 boyutlu** geniş özellikler, Transformer'dan çıkan 384 boyutlu vektörle, Rhythm Branch (64 dim) vektörüyle ve SQI Gating vektörüyle **en sonda (`torch.cat`)** Dense (Classifier) katmanından hemen önce birleştirildi.

---

## 4. 3 Aşamalı Eğitim Dinamikleri (Curriculum Learning)

Toplam 100 Epoch'luk eğitim süreci, modelin bebek adımlarıyla ilerlemesi için 3 faza bölündü:

### Faz 1 (Epoch 1 - 15): "Isınma ve Temel Öğrenme"
* **Veri:** Sadece dengeli olan Ana Veri Seti (Domain 0). İnternet verileri kapalı.
* **Amaç:** Modelin en temel EKG morfolojilerini (Normal, bloklar) öğrenmesi ve başlangıç ağırlıklarının oturması.
* **Ayarlar:** DANN (Domain Adaptation) kapalı (`lambda = 0.0`), Hard Mining kapalı. 

### Faz 2 (Epoch 16 - 70): "Dış Dünya ve Domain Adaptation (DANN)"
* **Veri:** Ana Veri Seti + İnternet Verileri (Tümü aktif).
* **Amaç:** Modele farklı cihaz ve hastane profillerini göstererek dayanıklılığını (robustness) artırmak.
* **Ayarlar:** DANN aktif edildi. `lambda` değeri 0.0'dan maksimum **0.1**'e doğru progressive (aşamalı) olarak artırıldı (Gradient Reversal Layer üzerinden).

### Faz 3 (Epoch 71 - 100): "Hard Mining ve Fine-Tuning"
* **Veri:** İnternet verileri tekrar kapatıldı, sadece Ana Veri Seti.
* **Amaç:** Dış verilerden kazanılan geniş bilgi havuzuyla, ana jüri veri setinin spesifik dağılımına "ince ayar" yapmak ve zorlu AFIB/AFL sınıflarına odaklanmak.
* **Ayarlar:** Modelin ana CNN omurgası (`layer1`'den `layer4`'e kadar) tamamen **DONDURULDU**. Sadece son Classifier katmanları eğitildi. Öğrenme oranı (LR) aniden **%10 (0.1x)** oranında düşürüldü. DANN kapatıldı. Hard Example Mining (AFIB ve AFL'ye 2.5 kat ceza) aktifleştirildi.

### Optimizer ve Scheduler
* **Optimizer:** AdamW (Weight Decay: `5e-4`).
* **Initial LR:** `1e-3` (Faz 3 başlangıcında manuel olarak `1e-4`'e düşürüldü).
* **Scheduler:** CosineAnnealingWarmRestarts (`T_0=20`, `T_mult=2`, `eta_min=1e-6`).

---

## 5. Sınıf Ağırlıklandırması (Weighting) ve Loss Stratejisi

Modelin en büyük zafiyeti AFIB ve AFL olduğu için sıradan CrossEntropyLoss yerine **Hibrit Kayıp Fonksiyonu (LDAM + Focal)** kullanıldı.

### Hibrit Loss Yapısı
Toplam Loss = `0.7 * LDAM Loss + 0.3 * Focal Loss`

1. **LDAM Loss (Label-Distribution-Aware Margin):**
   - Azınlık sınıfların karar sınırlarını (margin) dağılımdaki sayılarıyla ters orantılı olarak genişletmek için kullanıldı. 
   - Parametreler: `max_m = 0.5`, `s = 30.0`.
   
2. **Focal Loss + Hard Example Mining (Aşama 3):**
   - Kolay tahminleri baskılamak için: `gamma = 2.0`.
   - Etiket hatasını tolere etmek için: `label_smoothing = 0.10`.
   - **Hard Mining Alpha Çarpanı:** Eğitim Faz 3'te iken, modelin odak noktasını AFIB ve AFL'ye çevirmek için alpha ağırlıkları şu şekilde set edildi: `[Normal: 1.0, AFIB: 2.5, AFL: 2.5, LBBB: 1.0, RBBB: 1.0]`.

---

## 6. Early Stopping, Metrikler ve Regularizasyon

* **Early Stopping:** Sadece doğrulama verisindeki F1-Score (Macro) dalgalanmalarına kör olmamak adına, `Val Macro F1` metriği takip edildi. 
* **Patience:** `15 epoch`. (15 epoch boyunca yeni bir "Best Val F1" gelmezse eğitim kesilir).
* **SWA (Stochastic Weight Averaging):** Son 20 epoch boyunca (Epoch 80-100 arası) modelin ağırlıklarının hareketli ortalaması alınarak daha geniş ve stabil bir lokal minimuma oturması sağlandı. SWA_LR: `1e-4`.
* **Regularizasyon Değerleri:** 
  * Convolutional bloklarda spatial dropout: `0.1`.
  * Dense (Classifier) katmanlarında genel dropout: `0.3` ve `0.4`.
  * Weight Decay: `5e-4`.

---

## 7. Çıkarım (Inference) ve Eşik Optimizasyonu

Çıkarım (Test/Tahmin) aşamasında tek modelin zafiyetlerini kapatmak için en güçlü silahlar devreye sokulmuştur:

### Ensemble (Soft-Voting)
Sistem **3 farklı rastgele tohumla (Seed: 42, 100, 2026)** tamamen sıfırdan eğitilmiş 3 farklı CardioFusion-5 modelinden oluşur. Tahmin sırasında sinyal 3 modele de sokulur, her modelin ürettiği Softmax olasılıkları (probabilities) birleştirilip matematiksel ortalaması alınır. Hata korelasyonu düşük modeller birbirlerinin yanlışlarını kapatır.

### TTA (Test-Time Augmentation)
Her bir test EKG'si için, orijinal sinyalin yanına hafif baseline wander veya gaussian noise eklenmiş 4 sentetik kopya daha üretilip modele toplam 5 sinyalmiş gibi sunulur (Bu sayede elektrot temas gürültülerine karşı direnç kazanılır).

### Karar Eşikleri (Threshold Optimization)
Modelin Softmax çıktıları her zaman 0.5 sınırında en iyi F1'i vermez (özellikle AFIB ve AFL'de). Validation seti üzerinde `Nelder-Mead` optimizasyon algoritması kullanılarak her sınıfın F1-Score'unu maksimize edecek eşik değerleri bulunmuştur.
* **Nihai Optimize Eşikler:** `[Normal: 0.105, AFIB: 0.100, AFL: 0.100, LBBB: 0.600, RBBB: 0.100]`
* *Not:* LBBB dışındaki tüm eşiklerin düşmesi, modelin bu nadir hastalıkları kaçırmamak (recall'u artırmak) için daha esnek bir karar sınırına ihtiyaç duyduğunu kanıtlamıştır. Bu optimizasyon tek başına Ensemble Macro F1 skorunu %1-2 oranında yukarı taşımıştır.

> [!IMPORTANT]
> **Özet Sonuç:** Bu mimari ve eğitim yaklaşımıyla, standart modellerde 0.82 bandında takılan Macro F1 skoru **0.86+** seviyelerine, Cohen Kappa değeri ise **0.82+ (Mükemmel Uyum)** seviyesine çıkartılmıştır. Başarısız modellerin ana sebebi; veri dengesizliğine karşı Hard Mining uygulanmaması ve Multi-Scale/Wide özellikleri gözetmeden salt CNN/ResNet'e güvenilmesidir.
