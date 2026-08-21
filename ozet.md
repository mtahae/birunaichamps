# BirunAI CardioFusion-5 — Kapsamlı Proje Özeti ve PDR Teknik Dokümanı

**Proje:** TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — 2. Aşama  
**Görev:** 12 Derivasyonlu EKG Kayıtlarından 5-Sınıf Otomatik Sınıflandırma  
**Sınıflar:** Normal (0) · AFIB (1) · AFL (2) · LBBB (3) · RBBB (4)  
**Metrik:** Macro F1-Score (tek ve birincil)  
**En İyi Sonuç:** Val Macro F1 = **0.8317** | Eşik Opt. Macro F1 = **0.8352**

---

## İçindekiler

1. [Problem Tanımı ve Stratejik Çerçeve](#1-problem-tanımı-ve-stratejik-çerçeve)
2. [Veri Kaynakları ve Portföy](#2-veri-kaynakları-ve-portföy)
3. [Ön İşleme Pipeline'ı (Preprocessing)](#3-ön-i̇şleme-pipelineı-preprocessing)
4. [Augmentasyon Kuralları](#4-augmentasyon-kuralları)
5. [Model Mimarisi: CardioFusion-5 v2](#5-model-mimarisi-cardiofusion-5-v2)
6. [Eğitim Stratejisi: 3 Aşamalı Curriculum Learning](#6-eğitim-stratejisi-3-aşamalı-curriculum-learning)
7. [Kayıp Fonksiyonları ve Sınıf Dengesi](#7-kayıp-fonksiyonları-ve-sınıf-dengesi)
8. [Test Zamanı Stratejileri (Post-Processing)](#8-test-zamanı-stratejileri-post-processing)
9. [Açıklanabilirlik (XAI): 1D-GradCAM](#9-açıklanabilirlik-xai-1d-gradcam)
10. [Tekrar Üretilebilirlik (Reproducibility)](#10-tekrar-üretilebilirlik-reproducibility)
11. [Sakınılması Gereken Kritik Hususlar](#11-sakınılması-gereken-kritik-hususlar)
12. [5 Sınıfın Fizyolojik Özellikleri ve 12 Lead Anatomisi](#12-5-sınıfın-fizyolojik-özellikleri-ve-12-lead-anatomisi)
13. [Denenen Ancak Terk Edilen Yöntemler](#13-denenen-ancak-terk-edilen-yöntemler)
14. [Sonuç ve Stratejik Değerlendirme](#14-sonuç-ve-stratejik-değerlendirme)

---

## 1. Problem Tanımı ve Stratejik Çerçeve

### 1.1. Yarışma Kapsamı
TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması 2. Aşama kapsamında, 12 derivasyonlu EKG kayıtlarından 5 kardiyak sınıfın otomatik sınıflandırılması hedeflenmektedir.

- **Girdi:** 12 derivasyonlu EKG kaydı, 10 saniye, 500 Hz (TEKNOFEST) veya 250 Hz (İnternet)
- **Çıktı:** 5 sınıf olasılık dağılımı → argmax ile tek sınıf tahmini
- **Test:** Yarışma finalinde **görülmemiş, farklı bir hastaneden** gelen EKG'lerle test edilecek — ezberleyen model patlar.

### 1.2. Neden Macro F1? (PDF Bölüm 6)
Accuracy metriği aldatıcıdır. Normal sınıfı %95 doğru tahmin edip azınlık sınıfları (AFL) karıştıran bir model, yüksek accuracy gösterebilir ancak Macro F1'de çöker. Bu yarışmada **tek önemli metrik Macro F1'dir.** Her teknik karar *"bu Macro F1'i artırır mı?"* sorusuna göre verilmiştir.

### 1.3. 1. Aşama Deneyimi
1. Aşamada 3 sınıf (Sağlıklı / Ritim Bozukluğu / İletim Bozukluğu) sınıflandırması yapılmış ve **Macro F1 = 96.7** skoru elde edilmiştir. Bu skorun temeli:
- Sağlam ön işleme hattı (lead-wise Z-Score, per-patient split)
- Fizyolojik bilginin modele yansıtılması
- Dengeli veri seti üzerinde stabil eğitim

2. Aşama bu temel üzerine 5 ayrı sınıf ve çoklu veri seti ile genişletilmiştir.

---

## 2. Veri Kaynakları ve Portföy

### 2.1. Veri Setleri (PDF Çoklu Veri Seti Stratejisi)

| Veri Seti | Kaynak | Kullanılan Kayıt | Hz | Format |
|---|---|---|---|---|
| **TEKNOFEST 2026** | Yarışma | ~3,400 | 500 | .mat/.hea |
| PhysioNet ECG Arrhythmia | ABD | ~35,000 | 250-500 | .mat/.hea |
| PTB-XL | Almanya | ~18,000 | 250-500 | .mat/.hea |
| CPSC 2018 + Extra | Çin | ~8,000 | 250-500 | .mat/.hea |
| Georgia 12-Lead | ABD | ~8,000 | 250-500 | .mat/.hea |
| Chapman-Shaoxing/Ningbo | Çin | ~20,000 | 250-500 | .mat/.hea |
| **TOPLAM** | | **~90,253** | | |

### 2.2. SNOMED-CT Etiket Harmonizasyonu (PDF Bölüm 4)
Farklı veri setlerinde farklı SNOMED kodları bulunur. İki seviyeli harmonizasyon uygulanmıştır:

**Seviye 1 (Üst Sınıf — 1. Aşama Uyumlu):**
- 0: Normal · 1: Ritim Bozukluğu · 2: İletim Bozukluğu

**Seviye 2 (Alt Sınıf — 2. Aşama Hedefi):**
- 0: Normal · 1: AFIB · 2: AFL · 3: LBBB · 4: RBBB

Örnek SNOMED eşleştirmeler:
- `164889003` → AFIB | `164890007` → AFL
- `164909002` → LBBB | `59118001` → RBBB  
- `426783006` → Normal (Sinus Rhythm)
- `426177001` → Normal (Sinus Bradycardia)
- Ara kodlar (SVT, AT, APB, VPB, AVB çeşitleri) en yakın 5 hedef sınıfa atanmış, belirsiz olanlar atılmıştır.

### 2.3. Veri Dengeleme Stratejisi
PDF'te önerilen "undersampling" yerine **tam veri kullanımı** tercih edilmiştir. 90K'lık verinin tamamı eğitime sokulmuş, dengesizlik **matematiksel yöntemlerle** (Focal Loss, Class-Balanced Sampler, Hard Example Mining) çözülmüştür. Bu karar, Val Macro F1'i 0.79'dan **0.83**'e çıkarmıştır.

---

## 3. Ön İşleme Pipeline'ı (Preprocessing)

PDF Bölüm 1, 4, 7'deki tüm kurallar eksiksiz uygulanmıştır.

### Adım 1 — Sinyal Okuma
WFDB formatından `(N, 12)` sinyal çıkarımı yapılır. Her kayıt 12 derivasyon × değişken uzunluk olarak okunur.

### Adım 2 — Frekans Uyumu (Hz Harmonizasyonu)
Tüm sinyaller **250 Hz** hedef frekansına `scipy.signal.resample_poly` ile dönüştürülmüştür. PDF Bölüm 1 gerekçesi:
- Nyquist frekansı 125 Hz → EKG'deki en yüksek anlamlı frekans ~40 Hz olduğundan P-QRS-T morfolojisi korunur
- 250 Hz'de daha hızlı eğitim ve daha az RAM kullanımı
- İnternet verileri zaten 250 Hz

### Adım 3 — Segmentasyon
Tüm sinyaller **10 saniyelik sabit pencereye (2500 örneklem)** sabitlenmiştir:
- Kısa sinyaller: Ortadan simetrik sıfır padding
- Uzun sinyaller: Ortadan kırpma (P ve T dalgalarının korunması için)

### Adım 4 — Butterworth Bandpass Filtreleme (PDF Bölüm 4)
```
Butterworth Bandpass: 0.5 – 40 Hz, 4. derece, filtfilt (çift yönlü — sıfır faz kayması)
```
- **0.5 Hz alt kesim:** Baseline wander (nefes/konuşma artefaktı) kaldırır
- **40 Hz üst kesim:** EMG gürültüsü ve güç hattı gürültüsünü (50 Hz) kaldırır
- **50 Hz notch filtresi KULLANILMAZ** (PDF Bölüm 6.6: T dalgasına zarar verebilir)
- **4. derece Butterworth:** Düşük geçiş bantlı, faz bozulması minimal
- **filtfilt:** İleri-geri filtreleme ile faz kaymasını sıfıra indirir (P-QRS-T temporal sırası bozulmaz)
- **Kontrol:** Filtreleme sonrası P dalgası hâlâ görünür mü, QRS genişliği bozulmadı mı, ST segment düz mü kontrol edilmiştir

### Adım 5 — Lead-Wise Z-Score Normalizasyonu (PDF Bölüm 1 — EN KRİTİK)
```python
for lead in range(12):
    if std[lead] > 1e-6:
        signal[lead] = (signal[lead] - train_mean[lead]) / train_std[lead]
```
**Neden Global Z-Score YASAK?**
- V1 QRS genliği 0.5 mV, V5 QRS genliği 2.5 mV olabilir
- Global ortalama alınırsa V1 sinyali "yok" olur, V5 baskın hale gelir
- RBBB tanısı V1'e, LBBB tanısı V6'ya bağlıdır — bu lead'ler yok edilirse model öğrenemez

**Kritik Kurallar:**
1. Her 12 derivasyon kendi içinde bağımsız normalize edilir
2. μ ve σ değerleri **SADECE Eğitim (Train) setinden** hesaplanır
3. Aynı istatistikler validation ve test setlerine uygulanır (**data leakage önlenir**)
4. std = 0 olan flatline derivasyonlarda bölme yapılmaz

### Adım 6 — Sinyal Kalite İndeksi / SQI (PDF Bölüm 7)
Her derivasyon için 0-1 arası kalite skoru hesaplanır:
- **kSQI (Kurtosis):** QRS varsa yüksek kurtosis → QRS varlığını kontrol eder
- **bSQI (Baseline Wander):** Baseline standard sapması / sinyal standard sapması → düşük olmalı
- **pSQI (QRS Band Gücü):** 5-15 Hz frekans bandındaki güç / toplam güç → QRS kalitesi
- **rSQI (R-Peak Detection):** Kalp hızı 40-150 bpm aralığında mı → fizyolojik uygunluk

Formül: `SQI = 0.3*kSQI + 0.3*bSQI + 0.2*pSQI + 0.2*rSQI`

**SQI Kullanımı (PDF Bölüm 7):**
- Eğitimde: SQI < 0.2 olan kayıtlar eğitimden çıkarılır
- Derivasyon **ASLA tamamen atılmaz**, sadece ağırlığı düşürülür
- 12 derivasyonun tamamı her zaman modele verilir

### Adım 7 — Etiket Harmonizasyonu
SNOMED-CT eşleştirme ile 5 hedef sınıfa net eşleşemeyenler atılır. Çift etiketli kayıtlarda dominant seçilir, belirsizse kayıt atılır.

### Adım 8 — Çıktı
Her kayıt: `(12, 2500) float32 sinyal + label + SQI(12,) + wide_features(12,)` olarak `.npy` formatında disk'e yazılır.

---

## 4. Augmentasyon Kuralları (PDF Bölüm 3)

### 4.1. İZİN VERİLEN Augmentasyonlar
Eğitim setinde %80 olasılıkla uygulanır:

| Yöntem | Parametre | Neden |
|---|---|---|
| **Lead-Wise Amplitude Scale** | Her lead'e 0.9–1.1 arası bağımsız çarpan | Farklı cihaz kalibrasyonlarını simüle eder |
| **Gaussian Noise** | std=0.01 (SNR > 20 dB) | Elektrod gürültüsü simülasyonu |
| **Lead Dropout** | 1-2 rastgele lead sıfırlanır (%30 olasılık) | Model, bir lead düşük kaliteli olsa bile diğerlerine güvenmeyi öğrenir |
| **Baseline Wander** | Sinüzoidal drift (0.1-0.5 Hz, amp 0.05-0.3) | Nefes alma artefaktı simülasyonu |
| **Time Masking** | 50-250 sample'lık rastgele bölge sıfırlanır (%40) | SpecAugment'in EKG versiyonu — zaman kaydırma DEĞİL, maskeleme |
| **Signal Inversion** | Tüm sinyal ters çevrilir (%5) | Kablo ters bağlama simülasyonu |

### 4.2. YASAK Augmentasyonlar (PDF Bölüm 3 + Bölüm 6)

| Yöntem | Neden YASAK |
|---|---|
| **Zaman Kaydırma (np.roll)** | P-QRS-T temporal ilişkisini bozar. P dalgası QRS'ten önce gelmelidir. |
| **Rastgele Kırpma (Random Crop)** | Baştaki P dalgası veya sondaki T dalgası kaybolabilir. |
| **Global Amplitude Scale** | Tüm 12 lead'i aynı çarpanla çarpmak V1/V6 oranını bozar — RBBB/LBBB tanısı etkilenir. |
| **50 Hz Notch Filter** | T dalgasına zarar verebilir. 0.5-40 Hz bandpass yeterlidir. |
| **Mixup** | Yapay EKG'ler üreterek loss eğrisini zig-zag'a soktuğu deneylerde gözlemlenmiştir; tamamen kapatılmıştır. |

---

## 5. Model Mimarisi: CardioFusion-5 v2

PDF'te önerilen 3-Branch (CNN + MLP + Beat-Level BiGRU) Gating yapısının (Versiyon B) yerine, debug süresini minimize eden ve eğitim stabilitesini maksimize eden **Versiyon A (CardioFusion-Efficient)** temel alınmış ve Transformer Encoder ile güçlendirilmiştir. Toplam ~11.8 milyon parametre.

### 5.1. Multi-Scale SE-ResNet1D Backbone

**Multi-Scale CNN (PDF Versiyon A: 1D-CNN + SE):**
Sinyale tek bir pencereden bakmak yerine, 3 farklı evrişim çekirdeği paralel çalışır:
- **Kernel 3:** QRS spike'ları gibi çok hızlı değişimleri yakalar (morfoloji detayı)
- **Kernel 7:** P ve T dalgaları gibi orta vadeli yapıları inceler
- **Kernel 15:** RR aralıkları gibi uzun vadeli paternleri görür (ritim)

3 paralel dalın çıkışları birleştirilir → tek kanal boyutuna düşürülür.

**Squeeze-and-Excitation (SE) Blocks (PDF Versiyon A):**
Her ResNet bloğunun sonunda SE mekanizması, hangi kanalın/derivasyonun o an daha önemli olduğunu dinamik olarak ağırlıklandırır. Örneğin RBBB tespitinde V1'in ağırlığını otomatik artırır.

**ResNet Blokları:**
4 katmanlı ResNet yapısı (layer1→layer4), her katmanda:
- 2× Conv1D + BatchNorm + ReLU
- SE Block (channel attention)
- Spatial Dropout (0.1) — CNN overfitting önleme
- Residual (artık) bağlantı

### 5.2. Attention Pooling
Normal Average Pooling yerine, CNN'den çıkan zaman adımlarından hangilerinin sınıflandırma için en kritik olduğunu ağırlıklandıran zamansal dikkat mekanizması kullanılır. Model QRS kompleksine odaklanmayı kendi öğrenir.

### 5.3. Transformer Encoder (PDF Versiyon B'den Uyarlanmış)
CNN'den gelen sıkıştırılmış zaman serisi özelliklerini alır:
- **2 katman, 8 dikkat başı, dim=384**
- Sinyalin en başı ile en sonu arasındaki ritim değişikliklerini ilişkilendirir
- AFIB gibi düzensiz ritimleri yakalamada CNN'den çok daha etkili
- Positional Encoding ile zaman bilgisi korunur

### 5.4. Wide Feature Enjeksiyonu — Rhythm Branch (PDF Branch 2: Fizyolojik Feature MLP)
PDF'te önerilen "Sayı" dalı (fizyolojik feature MLP) doğrudan uygulanmıştır. NeuroKit2 ve sinyal analizi ile çıkarılan **12 adet** biyobelirteç:

| # | Özellik | Hedef Sınıf | Kaynak |
|---|---|---|---|
| 1 | Yaş | Genel | Metadata |
| 2 | Cinsiyet | Genel | Metadata |
| 3 | PR Mesafesi | AV Blok / Normal | R-peak tespiti |
| 4 | QRS Süresi | LBBB / RBBB (>120ms) | R-peak tespiti |
| 5 | QT Süresi | Genel | NeuroKit2 |
| 6 | QTc (Düzeltilmiş) | Genel | Bazett formülü |
| 7 | Kalp Hızı | Genel | RR aralıkları |
| 8 | Eksen (Axis) | LBBB/RBBB | Lead I / aVF |
| 9 | P-Dalgası Varlığı | AFIB (P yok) | Lead II analizi |
| 10 | RR Varyans Katsayısı (`rr_cv`) | AFIB (yüksek) vs AFL (düşük) | RR aralık std/mean |
| 11 | P-Dalgası Düzenliliği (`p_regularity`) | AFIB vs Normal | P-peak otokorelasyon |
| 12 | Atriyal Hız (`atrial_rate`) | AFL (250-350 bpm) | P-peak frekansı |

Bu özellikler CNN+Transformer'dan gelen gizli özelliklerle (latent features) "bottleneck" katmanında kaynaştırılır. **.npy cache dosyaları** halinde RAM'de tutularak I/O darboğazı %95 oranında çözülmüştür.

### 5.5. DANN — Domain-Adversarial Neural Network (PDF Bölüm 3.3)
Çoklu veri setinin en büyük riski, modelin "hangi hastaneden geldiğini" öğrenip kardiyak bilgiyi ikincil plana atmasıdır.

**Mimari:**
```
Feature Extractor (CNN+Transformer) → features
    ├── Classifier (5 sınıf) ─────→ L_class
    └── Domain Classifier (2 sınıf: TEKNOFEST / İnternet) ──→ L_domain (ters gradyan)
```

**Gradient Reversal Layer:** Domain sınıflandırıcıya giden gradyanlar tersine çevrilir. Bu, feature extractor'ı veri seti bilgisini "unutmaya" zorlar — sadece evrensel kardiyak bilgi kalır.

**Lambda (λ):** 0.1 olarak ayarlanmıştır. P2'de alpha 0'dan 1'e kademeli artar.

### 5.6. Multi-Task Learning (PDF Bölüm 8)
Ana 5-sınıf görevine ek olarak, 1. Aşamadaki 3-sınıf bilgisi yardımcı görev olarak korunur:

| Ana Sınıf | Yardımcı Sınıf |
|---|---|
| Normal → | Normal (0) |
| AFIB → | Ritim Bozukluğu (1) |
| AFL → | Ritim Bozukluğu (1) |
| LBBB → | İletim Bozukluğu (2) |
| RBBB → | İletim Bozukluğu (2) |

**Loss Formülü:** `L = L_main + 0.3 * L_aux + 0.1 * L_domain`

Bu sayede model, LBBB ve RBBB'nin aynı "İletim" ailesinden olduğunu loss seviyesinde öğrenir; AFIB ve AFL'nin "Ritim" ailesinden olduğunu kavrar.

---

## 6. Eğitim Stratejisi: 3 Aşamalı Curriculum Learning (PDF Bölüm 3.3)

PDF'te önerilen 3 aşamalı curriculum learning tam olarak uygulanmıştır. Model basitten zora, özelden genele, genelden özele doğru eğitilir.

### Phase 1 — Temel (Sadece TEKNOFEST / Epoch 1-20)
- **Veri:** ~3,400 TEKNOFEST kaydı (dengeli dağılım)
- **DANN:** Kapalı
- **LR:** Baz değer (1e-3)
- **Early Stopping Patience:** 25 epoch
- **Warmup:** İlk 5 epoch'ta LR kademeli artırılır
- **Amaç:** Model hedef yarışma verisinin temiz yapısını, gürültüsüz ortamda hızlıca kavrar. Doğru bir başlangıç ağırlığı (prior) oluşturulur.

### Phase 2 — Genişleme (KARMA + DANN / Epoch 21-100)
- **Veri:** 90,253 kaydın TAMAMI (TEKNOFEST + tüm İnternet verileri)
- **DANN:** Aktif (alpha 0'dan 1'e kademeli artar)
- **LR:** Baz değerin **%15'i** (0.00015) — P1 özelliklerini korumak için
- **Scheduler:** Cosine Annealing (T_max = 80 epoch, eta_min = 1e-6)
- **Early Stopping Patience:** 25 epoch
- **Class-Balanced Sampler:** Aktif (her batch'te 5 sınıf eşit)
- **Hard Example Mining:** P2'de AFIB/AFL kayıpları **1.5x** çarpılır
- **Amaç:** Model farklı cihaz gürültülerini, binlerce farklı morfolojiyi görerek kas yapar; overfit olmaktan kurtulur. DANN ile domain-agnostic özellikler çıkarılır.

### Phase 3 — İnce Ayar (FINE-TUNE + SWA / Epoch 101-130)
- **Veri:** Sadece TEKNOFEST kaydı
- **Kritik:** CNN Backbone katmanlarının tamamı **DONDURULUR** (~7M parametre kitlenir). Sadece Transformer ve Sınıflandırıcı eğitilir.
- **LR:** Baz değerin **%5'i** (0.00005)
- **Early Stopping Patience:** 15 epoch
- **SWA (Stochastic Weight Averaging):** Aktif — son epoch'lardaki ağırlıkların ortalaması alınır
- **Hard Example Mining:** AFIB/AFL kayıpları **2.5x** agresif çarpanla cezalandırılır
- **Amaç:** Devasa veriden öğrenilen mükemmel jenerik CNN özelliklerini bozmadan, sadece karar verme mekanizmasını TEKNOFEST'in jüri standartlarına göre kalibre etmek.

### Neden Bu 3 Aşama? (PDF Gerekçesi)
- **Aşama 1:** Model dengeli veride temel öğrenir
- **Aşama 2:** Farklı hasta popülasyonlarını öğrenir (domain genelleme)  
- **Aşama 3:** TEKNOFEST veri dağılımına geri dön, domain overfit'i kır

---

## 7. Kayıp Fonksiyonları ve Sınıf Dengesi

### 7.1. Focal Loss (PDF Bölüm 6)
```python
focal_term = (1 - pt) ** gamma  # gamma = 2.0
loss = focal_term * ce_loss * alpha_t  # alpha_t = class weight
```
- **gamma = 2.0:** Model kolay örnekleri (zaten doğru tahmin ettiği Normal sınıfı) "boşverip" zorlandığı örneklere (AFL, AFIB) odaklanır
- **Label Smoothing:** 0.03 (yumuşak hedef etiketler — aşırı özgüveni önler, AFL/AFIB ayrımı için yeterince keskin)

### 7.2. Class Weights — 4th-Root Smoothing (PDF Bölüm 6)
PDF'te `weight = 1 / sqrt(freq)` önerilmiştir. Biz daha hafif bir versiyonu kullandık:
```python
weights = power(1/freq, 0.25)  # 4. dereceden kök
weights = weights / weights.mean()  # Normalize
```
Normal sınıf %74 yer kapladığından sqrt çok agresif oluyordu. 4th-root, sınıflar arasında dengeli ama aşırı olmayan bir ağırlıklandırma sağladı.

### 7.3. Class-Balanced Sampler (PDF Bölüm 6)
Eğitimin tüm aşamalarında, her mini-batch'in içine **5 sınıftan eşit sayıda sinyal** girmesi garanti altına alınmıştır. Normal sınıfın batch bazında baskınlığı ortadan kaldırılmıştır.

### 7.4. Hard Example Mining (PDF Bölüm 3.4)
Validation setindeki en yüksek Cross-Entropy loss'a sahip ~200 örnek tespit edilip sonraki epoch'larda kullanılır. Ek olarak, AFIB ve AFL (en çok karışan iki sınıf) tahminlerinde çarpan uygulanır:
- **P2:** `[Normal:1.0, AFIB:1.5, AFL:1.5, LBBB:1.0, RBBB:1.0]`
- **P3:** `[Normal:1.0, AFIB:2.5, AFL:2.5, LBBB:1.0, RBBB:1.0]`

### 7.5. Multi-Task Loss (PDF Bölüm 8)
`L = L_main(Focal) + 0.3 * L_aux(CE) + 0.1 * L_domain(CE)`

---

## 8. Test Zamanı Stratejileri (Post-Processing)

### 8.1. SWA — Stochastic Weight Averaging
P3'te son epoch'lardaki model ağırlıklarının hareketli ortalaması alınır. Bu yöntem modeli keskin (ve muhtemelen testte başarısız olacak) bir "loss çukurundan" çıkarıp daha geniş ve güvenli bir optimum noktaya oturtur.

### 8.2. TTA — Test-Time Augmentation (PDF Bölüm 3)
Model, tahmin yaparken sinyalin **5 farklı varyasyonuna** bakar:
1. Orijinal sinyal
2. Zaman ekseninde ters çevrilmiş sinyal (`flip`)
3. Hafif Gaussian gürültü eklenmiş sinyal
4. %95 genlikli sinyal (`scale 0.95`)
5. %105 genlikli sinyal (`scale 1.05`)

5 olasılık matrisinin ortalaması alınarak son karar verilir.

### 8.3. Nelder-Mead Sınıf Bazlı Eşik Optimizasyonu
Sınıfların olasılık eşikleri standart 0.50'de bırakılmaz. Validation seti üzerinde:
1. **Nelder-Mead optimizasyon algoritması** ile her sınıf için F1 skorunu maksimize eden noktalar aranır (6 farklı başlangıç noktasından)
2. **0.01 hassasiyetinde ince Grid Search** ile Nelder-Mead sonucunun çevresinde rafine edilir

Bulunan optimal eşikler:
| Sınıf | Optimal Eşik | F1 Kazancı |
|---|---|---|
| Normal | 0.21 | 0.934 |
| AFIB | 0.26 | 0.756 |
| AFL | 0.23 | 0.604 |
| LBBB | 0.29 | 0.922 |
| RBBB | 0.33 | 0.908 |

AFL'nin eşiği 0.50 yerine 0.23'e çekilerek Recall artırılmış, bu tek başına Macro F1'i ~%1 yükseltmiştir.

---

## 9. Açıklanabilirlik (XAI): 1D-GradCAM

BirunAI sadece tahmin yapan bir kara kutu değildir. Sınıflandırma yaptıktan sonra son evrişim (Conv1D) katmanına doğru geriye dönük türev (gradient) hesabı yaparak, modele *"Bu kararı verirken EKG'nin hangi dalgasına odaklandın?"* sorusunu sorar.

**Yöntem:**
1. Son Conv1D katmanının (layer4) aktivasyonları ve gradyanları kaydedilir
2. Gradyanların kanal bazında ortalaması alınarak ağırlıklar hesaplanır
3. Aktivasyonlar × ağırlıklar → ısı haritası (heatmap) üretilir
4. Her 5 sınıftan 3'er örnek için GradCAM görselleri oluşturulur
5. Confusion matrix otomatik çizilir

---

## 10. Tekrar Üretilebilirlik (Reproducibility) (PDF Bölüm 9)

### 10.1. Seed Sabitlenmesi
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 10.2. Merkezi Konfigürasyon (config.py)
Tüm hiperparametreler tek dosyada (`config.py`) tutulur. PDF'te önerilen `config.yaml` yerine Python modülü olarak uygulanmıştır (IDE otomatik tamamlama ve tip güvenliği avantajı).

Güncel kritik parametreler:
```
seed: 42
sampling_rate: 250 Hz
filter_low: 0.5 Hz
filter_high: 40 Hz
filter_order: 4
n_classes: 5
n_aux_classes: 3
batch_size: 64
lr: 0.001
weight_decay: 5e-4
focal_gamma: 2.0
label_smoothing: 0.03
epochs: 130 (P1:20 + P2:80 + P3:30)
patience_p1: 25 | patience_p2: 25 | patience_p3: 15
dann_lambda: 0.1
aux_loss_weight: 0.3
```

### 10.3. Per-Patient Split (PDF Bölüm 2)
Aynı hastanın kaydı hem train hem test'te asla bulunmaz. TEKNOFEST'in verdiği resmi split'ler kullanılmış, ek olarak `patient_id` kümelerinin kesişiminin sıfır olduğu assert ile doğrulanmıştır.

---

## 11. Sakınılması Gereken Kritik Hususlar (PDF Bölüm 6)

PDF'te belirtilen "ölümcül hatalar" ve projemizde nasıl önlendiği:

| # | Kritik Husus | PDF Kuralı | Bizim Uygulamam |
|---|---|---|---|
| 1 | **Global Z-Score** | ASLA yapma | ✅ Lead-wise Z-Score, train stats |
| 2 | **Rastgele Hasta Bölmesi** | Per-patient split | ✅ Resmi split + assert kontrolü |
| 3 | **Zaman Kaydırma (np.roll)** | ASLA yapma | ✅ Kod tabanından tamamen silindi |
| 4 | **Rastgele Kırpma** | ASLA yapma | ✅ Ortadan kırpma (simetrik) |
| 5 | **Global Amplitude Scale** | ASLA yapma | ✅ Lead-wise scale (0.9-1.1) |
| 6 | **50 Hz Notch** | ASLA yapma | ✅ Sadece bandpass 0.5-40 Hz |
| 7 | **Derivasyon Atma** | ASLA tamamen çıkarma | ✅ Lead dropout augmentation |
| 8 | **Accuracy'ye Aldanma** | Macro F1 tek metrik | ✅ Sınıf bazında F1 takibi |
| 9 | **Data Leakage** | Train stats only | ✅ Sadece train'den hesaplanan μ/σ |
| 10 | **Aynı Hasta Train+Test** | Kontrol et | ✅ assert len(overlap) == 0 |

---

## 12. 5 Sınıfın Fizyolojik Özellikleri ve 12 Lead Anatomisi (PDF Bölüm 5 + 10)

### 12.1. Sınıf Morfolojileri

| Sınıf | RR Interval | P-Dalgası | QRS | Kritik Lead | En Sık Karıştığı |
|---|---|---|---|---|---|
| **Normal** | Düzenli (60-100 bpm) | Her QRS'ten önce mevcut | 80-120 ms | Tümü | — |
| **AFIB** | Düzensiz (irregularly irregular) | Yok, f-wave (kaotik) | Normal | Lead II | AFL |
| **AFL** | Düzenli | Testere dişi flutter wave (250-350/dk) | Normal | Lead II, V1 | AFIB |
| **LBBB** | Düzenli | Normal | >120 ms, V1: rS/QS, V6: Geniş R | V1, V6, I, aVL | RBBB |
| **RBBB** | Düzenli | Normal | >120 ms, V1: rsR' (tavşan kulağı) | V1 (en kritik) | LBBB |

### 12.2. 12 Lead Anatomisi

| Lead | Gördüğü Yer | Kritik Olduğu Sınıf |
|---|---|---|
| **V1** | Sağ ventrikül, septum | **RBBB** (rsR'), **LBBB** (rS) |
| **V2** | Septum | LBBB, RBBB |
| **V5-V6** | Sol ventrikül lateral | **LBBB** (geniş R) |
| **I, aVL** | Sol ventrikül yüksek lateral | LBBB |
| **II, III, aVF** | Alt duvar | **AFIB/AFL** (P dalgası görünürlüğü) |
| **aVR** | Sağ atriyum | Genelde negatif |

**Altın Kurallar:**
- RBBB tanısı **V1'de** yazılıdır
- LBBB tanısı **V1+V6'da** yazılıdır
- AFIB/AFL tanısı **Lead II'de** yazılıdır (P yokluğu / flutter wave)

---

## 13. Denenen Ancak Terk Edilen Yöntemler

Aşağıdaki yöntemler denenmiş ancak Macro F1'i iyileştirmediği veya kötüleştirdiği için terk edilmiştir:

| Yöntem | Denenen Parametreler | Sonuç | Neden Terk Edildi |
|---|---|---|---|
| **LDAM Loss** | s=30, s=15 | F1: 0.80 (kötüleşme) | Gradient patlaması + logit scaling Focal ile çelişiyor |
| **Mixup Augmentation** | alpha=0.2, P2'de | Loss zig-zag | Yapay EKG'ler gerçek morfolojiyi bozuyor |
| **Undersampling** | 90K → 30K | F1: 0.79 | Veri kaybı modelin genelleme yeteneğini düşürüyor |
| **Label Smoothing 0.05** | LABEL_SMOOTHING=0.05 | F1: 0.83 | 0.03 ile daha keskin kararlar → daha iyi AFL/AFIB ayrımı |
| **P2 LR 0.1x** | LEARNING_RATE * 0.1 | Yeterli öğrenme yok | 0.15x ile 90K veriyi daha iyi öğreniyor |

---

## 14. Sonuç ve Stratejik Değerlendirme

### 14.1. En İyi Sonuçlar

| Metrik | Skor |
|---|---|
| En İyi Val Macro F1 | **0.8317** |
| TTA Macro F1 (5 augmentation) | 0.8280 |
| Eşik Opt. Macro F1 (TTA üzerinden) | **0.8352** |

**Sınıf Bazında F1 (En İyi Model):**

| Sınıf | F1 | Durum |
|---|---|---|
| Normal | 0.9453 | ✅ Mükemmel |
| AFIB | 0.7529 | ⚠️ İyileştirilebilir |
| **AFL** | **0.5872** | ❌ Ana darboğaz |
| LBBB | 0.9360 | ✅ Mükemmel |
| RBBB | 0.9369 | ✅ Mükemmel |

### 14.2. Kritik Öğrenimler
1. **90K'lık tam veri kullanımı**, dengelenmiş 30K'lık alt kümeye göre çok daha iyi sonuç vermiştir (+3% F1)
2. **LDAM Loss bu mimariyle uyumsuzdur** — iki kez denenip iki kez başarısız olmuştur
3. **AFL en zor sınıftır** — AFIB ile karışması macro F1'in %5 kaybına neden olur
4. **Curriculum Learning** overfit'i önlemede en etkili yöntemdir
5. **Nelder-Mead eşik optimizasyonu** tek başına +%1 F1 kazandırmıştır

### 14.3. Altın Cümle
> *"Herkes daha derin CNN koyarken, biz EKG'nin fizyolojisini modelin içine gömdük. 12 derivasyonun hangisine güveneceğini, hangi nabızın anormal olduğunu, ne zaman emin olmadığını kendisi öğrendi."*

### 14.4. 3 Altın Kural (PDF Alt Satır)
1. **12 derivasyon, lead-wise Z-Score, hasta bazında bölme.** Bunlar olmazsa proje çöker.
2. **Macro F1 tek tanrıdır.** Accuracy aldatıcıdır.
3. **Görülmemiş veri mindset'i.** Her zaman *"bu model yarışmada hiç görmediğim bir hastanede çalışır mı?"* diye sorgula.

---

*Bu belge, BirunAI CardioFusion-5 projesinin TEKNOFEST 2026 PDR (Ön Tasarım Değerlendirmesi) aşaması için hazırlanmış kapsamlı teknik dokümanıdır.*
