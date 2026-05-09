# BirunAI — EKG Sınıflandırma Proje Durum Raporu
**Son Güncelleme:** 9 Mayıs 2026

---

## 1. Projenin Amacı

TEKNOFEST Sağlıkta Yapay Zeka yarışması kapsamında **12 derivasyonlu EKG sinyallerini** otomatik olarak 3 sınıfa ayıran bir derin öğrenme sistemi geliştirilmektedir.

| Sınıf | Tanım | Klinik Örnekler |
|-------|-------|-----------------|
| **0 — Normal** | Sağlıklı kalp ritmi | Sinüs ritmi, sinüs bradikardisi |
| **1 — Ritim Bozukluğu** | Kalbin düzensiz/hızlı atması | Atriyal fibrilasyon (AFIB), SVT, PVC |
| **2 — İletim Bozukluğu** | Elektrik iletim sistemi hasarı | Dal blokları (RBBB/LBBB), AV blok, ST değişiklikleri |

Çakışma durumunda öncelik sırası: **İletim (2) > Ritim (1) > Normal (0)**

---

## 2. Proje Gelişim Süreci — Kronoloji

### Faz 1: PTB-XL ile Alıştırma (Tamamlandı)

İlk aşamada sadece **PTB-XL** veri seti kullanılarak tüm pipeline kuruldu ve model eğitildi. Bu aşama bir "alıştırma" niteliğindeydi.

- **Veri:** 21,799 EKG kaydı (PhysioNet PTB-XL v1.0.3)
- **Format:** 12 kanal × 10 sn × 500 Hz → filtre sonrası (12, 2500)
- **Sınıf dağılımı:** Normal %37, Ritim %3.2, İletim %59.7
- **Temel sorun:** Ritim Bozukluğu yalnızca 687 kayıt (%3.2) — aşırı sınıf dengesizliği

**PTB-XL eğitiminde gözlemlenen sorunlar:**
- Ritim F1 skoru çok düşük (veri azlığı nedeniyle)
- Val Loss düzensiz seyretti
- Model overfit eğilimi gösterdi

### Faz 2: Multi-Dataset Entegrasyonu (Tamamlandı)

PTB-XL'in yetersiz Ritim verisi sorunundan dolayı **5 büyük açık kaynak veri seti** birleştirildi.

---

## 3. Kullanılan Veri Setleri

| Veri Seti | Kaynak | Kayıt Sayısı | Not |
|-----------|--------|--------------|-----|
| PTB-XL | PhysioNet (Challenge 2020) | 21,837 | Ana set |
| CPSC 2018 | Çin Fizyolojik Sinyal Yarışması | ~6,877 | — |
| CPSC 2018 Extra | CPSC ek seti | ~3,453 | — |
| Georgia | Georgia Elektrokardiografi DB | ~10,344 | — |
| ECG Arrhythmia | Chapman-Shaoxing/Ningbo | ~45,009 | — |

> **Önemli Not:** ECG Arrhythmia (Chapman-Shaoxing) verisi ile Challenge 2020 arasında veri örtüşmesi riski mevcuttur. Bu nedenle tüm kayıtlar MD5 hash ile taranarak mükerrerler tespit edilip temizlendi.

**Ham toplam:** 87,520 kayıt → **Temizlenmiş toplam: 86,539 kayıt**

### Temizlik Sonrası Sınıf Dağılımı

| Sınıf | Kayıt | Oran |
|-------|-------|------|
| Normal | 32,528 | %37.6 |
| Ritim Bozukluğu | 7,807 | %9.0 |
| İletim Bozukluğu | 46,204 | %53.4 |

Multi-dataset entegrasyonu sayesinde Ritim Bozukluğu verisi **687 → 7,807** adet oldu (**~21 kat artış**).

---

## 4. Veri İşleme Pipeline'ı

Tüm işlemler sıralı Python script'leri ile gerçekleştirilir. `baslat.py` bunları tek komutla çalıştırır.

### Adım 0 — Veri Birleştirme (`adim00_veri_birlestirme.py`)
- 5 veri setinden `.hea` başlık dosyaları okunur
- SNOMED-CT kodları → 3 sınıf etiketine dönüştürülür
- `unified_manifest.csv` üretilir (87,520 kayıt)

### Adım 1 — Genel Kalite Kontrol (`adim01_kalite_kontrol_genel.py`)
- **MD5 sinyal hash** ile çapraz veri seti mükerrer tespiti
- NaN/Inf/Flat-line sinyal kontrolü
- 12 derivasyon (lead) sayısı doğrulaması
- **Sonuç:** 981 kayıt elendi (106 bozuk + 875 mükerrer) → 86,539 temiz kayıt

### Adım 2 — Filtreleme (`adim02_filtreleme.py`)
- wfdb kütüphanesi ile `.mat`/`.hea` okuma
- **500 Hz → 250 Hz** alt örnekleme (VRAM tasarrufu, bilgi kaybı yok)
- **Butterworth Bandpass** 0.5–40 Hz (solunum artefaktı + EMG gürültüsü eleme)
- **Z-score normalizasyon** (kanal bazlı)
- `filtfilt` kullanımı (sıfır faz kayması — EKG zamanlaması kritik)
- **Sonuç:** 86,539/86,539 başarılı

### Adım 3 — Sinyal Kalite Kontrol (`adim03_kalite_kontrol.py`)
- Flat-line testi (std < 0.01)
- Clipping testi (%5'ten fazlası ±20 sınırı aşıyorsa)
- Uzunluk doğrulaması
- **Sonuç:** 81,006 geçti (%93.6), 5,533 elendi (uzunluk uyuşmazlığı)

### Adım 4 — Segmentasyon (`adim04_segmentasyon.py`)
- Tüm sinyaller tam **(12, 2500)** boyutuna getirilir
- Kısa sinyaller: sıfır doldurma (zero-padding)
- Uzun sinyaller: merkezi kırpma (center-crop)
- Her kayıt `{ecg_id}.npy` olarak kaydedilir
- **Sonuç:** 81,006 .npy dosyası oluşturuldu

### Adım 6 — Veri Bölme (`adim06_veri_bolme.py`)
- **Stratified %70/%15/%15** bölme (Train/Val/Test)
- Aynı hastanın kayıtları aynı sette kalır (veri sızıntısı önleme)

| Set | Kayıt | Normal | Ritim | İletim |
|-----|-------|--------|-------|--------|
| Train | 56,704 | 22,384 | 4,573 | 29,747 |
| Val | 12,151 | 4,797 | 980 | 6,374 |
| Test | 12,151 | 4,796 | 980 | 6,375 |

### Adım 6b — SMOTE Oversampling (`adim06b_oversampling.py`)
Ritim sınıfının dengelenmesi için **PCA-SMOTE** uygulandı:

- 30,000 boyutlu sinyal vektörleri (12×2500) PCA ile **256 bileşene** indirgendi
- SMOTE ile sentetik Ritim örnekleri üretildi
- Inverse transform ile orijinal sinyal uzayına döndürüldü
- Time-shift + Gaussian noise + amplitude scaling augmentasyonu eklendi
- **Hedef:** Ritim sayısını 2×'e çıkar, max 15,000 ile sınırla
- **Sonuç:** 4,573 → 9,146 Ritim örneği (4,573 sentetik)
- **SMOTE sonrası eğitim seti:** 61,277 kayıt

**Önemli:** SMOTE **yalnızca eğitim setine** uygulandı. Val/Test setleri gerçek verilerden oluşmaktadır.

---

## 5. Model Mimarisi

**Hibrit 1D-CNN + BiLSTM + Self-Attention**

```
Girdi: (batch, 12, 2500)
  │
  ▼
[CNN Blok 1] Conv1d(12→64, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  → (batch, 64, 1250)
[CNN Blok 2] Conv1d(64→128, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  → (batch, 128, 625)
[CNN Blok 3] Conv1d(128→256, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  → (batch, 256, 312)
  │
  ▼
[BiLSTM] 2 katman, 128 hidden, bidirectional → (batch, 312, 256)
  │
[Self-Attention] Softmax ağırlıklı toplam → (batch, 256)
  │
[FC] 256 → 128 → 3 (logits)
```

| Bileşen | Görev | Klinik Karşılık |
|---------|-------|-----------------|
| **CNN** | Yerel dalga şekli öğrenme | QRS genişliği, T dalgası morfolojisi |
| **BiLSTM** | Zamansal sıra öğrenme | RR aralığı düzenliliği, P-QRS ilişkisi |
| **Attention** | Önemli bölgeye odaklanma | XAI için temel — hangi saniyeye baktı? |

**Toplam parametre:** ~1,150,276

### Kayıp Fonksiyonu: Focal Loss
```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
```
- γ=1.5 → Kolay örneklerin etkisini düşürür, zor örneklere odaklanır
- α_t → Sınıf ağırlıkları (sınıf dengesizliği için)

---

## 6. Eğitim Konfigürasyonu (v5 — Multi-Dataset)

| Parametre | Değer | Gerekçe |
|-----------|-------|---------|
| Batch Size | 64 | 87K+ veri, GPU'da verimli |
| Learning Rate | 1e-4 | Büyük/çok-domainli veri için stabil |
| Weight Decay | 1e-3 | 5 farklı kaynak → güçlü regularizasyon |
| Epochs | 60 | Büyük veri daha az epoch gerektirir |
| Early Stopping | 15 epoch | Cosine LR yeniden artacak, sabret |
| Warmup | 3 epoch | Büyük batch ile kısa warmup |
| Focal Loss γ | 1.5 | SMOTE sonrası hafif dengesizlik kaldı |
| Label Smoothing | 0.1 | Multi-domain için aşırı güvenilirliği önle |
| Grad Clip | 1.0 | BiLSTM gradyan patlamasını önle |
| Optimizer | AdamW | Weight decay ile güvenli momentum |
| Scheduler | CosineAnnealingWarmRestarts | T₀=15, periyot 2× uzar |
| GPU | RTX 4060 Laptop (8.6 GB VRAM) | Mixed Precision (AMP) aktif |

---

## 7. Eğitim Sonuçları

### Son Eğitim — 87K Multi-Dataset Modeli

**Eğitim Süreci:**
- Toplam 60 epoch çalıştı (early stopping devreye girmedi)
- Epoch başına süre: ~6-8 dakika
- Toplam süre: ~6 saat (veri işleme dahil ~8 saat)

**Validation Metrikleri (En İyi Epoch):**
- En iyi Val Macro F1: **0.7856**

**Test Seti Sonuçları (12,151 kayıt üzerinde):**

| Sınıf | Precision | Recall | F1 | Support | ROC-AUC |
|-------|-----------|--------|----|---------|---------|
| Normal | 0.7815 | 0.9443 | **0.8553** | 4,796 | 0.9522 |
| Ritim Bozukluğu | 0.6109 | 0.7531 | **0.6746** | 980 | 0.9455 |
| İletim Bozukluğu | 0.9165 | 0.7401 | **0.8189** | 6,375 | 0.9171 |
| **Macro Ortalama** | — | — | **0.7829** | 12,151 | — |

**Cohen's Kappa:** 0.6934

### Sonuçların Yorumu

**Güçlü Yönler:**
- ROC-AUC değerleri tüm sınıflarda 0.91+ → model ayırt etme kapasitesi yüksek
- Normal sınıfı recall %94.4 → normal EKG'leri çok iyi buluyor
- İletim Bozukluğu precision %91.6 → İletim tespitinde çok az yanlış alarm

**Zayıf Yönler:**
- **Ritim F1 = 0.6746** → En düşük sınıf, hâlâ gelişmeye açık
  - Ritim Precision = 0.61: Ritim dediği 39 hasta aslında başka sınıf
  - Ritim Recall = 0.75: 100 gerçek Ritim hastasından 25'ini kaçırıyor
- İletim Recall = 0.74 → 100 İletim hastasından 26'sını kaçırıyor
- Val Loss train loss'tan yüksek seyrediyor → hafif underfitting/domain gap

**Neden Ritim Hâlâ Zayıf?**
1. SMOTE'a rağmen gerçek Ritim verisi oransal olarak hâlâ düşük
2. Ritim bozuklukları en çeşitli grup (AFib, PVC, SVT...) → model genellemeye çalışıyor
3. PCA-SMOTE %44.1 varyans açıklıyor → sentetik veriler gerçeği tam yansıtmıyor

---

## 8. Açıklanabilir YZ (XAI) — GradCAM

`adim10_gradcam.py` — Eğitim sonrası otomatik çalışır.

**Çalışma Prensibi:**
1. Son CNN katmanına forward + backward hook takılır
2. Model tahmin yapar → gradyanlar geriye akar
3. `cam = Σ(weights_i × activations_i)` → hangi zaman diliminin kararı etkilediği
4. EKG sinyali üzerine ısı haritası olarak çizilir

**Ne Gösteriyor:**
- Kırmızı bölge → modelin o kararı verirken en çok odaklandığı yer
- İletim bozukluğunda genellikle QRS kompleksinin bulunduğu bölge ısınır
- Ritim bozukluğunda atımlar arası boşluklar ve zamansal düzensizlikler ısınır

---

## 9. Dashboard

`dashboard.py` — Eğitim sırasında `http://localhost:5000` adresinden canlı izleme.

**Özellikler:**
- Train/Val Loss grafikleri (Chart.js, her 2 saniyede güncellenir)
- Train/Val Macro F1 grafikleri
- Sınıf bazlı F1 çubuk grafikleri
- Epoch ilerleme çubuğu + tahmini kalan süre
- Early Stopping sayacı (config'den otomatik patience değeri alır)
- Dark theme + glassmorphism tasarım

---

## 10. Proje Dosya Yapısı (Güncel)

```
birun/
│
├── config.py                        ← Tüm parametreler burada
│
├── adim00_veri_birlestirme.py       ← 5 dataset → unified_manifest.csv
├── adim01_kalite_kontrol_genel.py   ← Hash ile mükerrer + bozuk temizlik
├── adim01_veri_yukleme.py           ← Eski PTB-XL pipeline (referans)
├── adim02_filtreleme.py             ← Filtre + alt örnekleme + normalizasyon
├── adim03_kalite_kontrol.py         ← Sinyal QC (flat-line, clipping)
├── adim04_segmentasyon.py           ← (12, 2500) standart boyut
├── adim05_ozellik_cikarma.py        ← İstatistiksel özellikler (EDA)
├── adim06_veri_bolme.py             ← %70/%15/%15 stratified split
├── adim06b_oversampling.py          ← PCA-SMOTE (sadece train seti)
├── adim07_model_mimarisi.py         ← CNN + BiLSTM + Attention + Dataset
├── adim08_egitim.py                 ← Eğitim döngüsü (SWA, Focal Loss)
├── adim09_degerlendirme.py          ← Test metrikleri ve raporlar
├── adim10_gradcam.py                ← 1D-GradCAM (XAI görselleştirme)
│
├── baslat.py                        ← TEK KOMUT: tüm pipeline
├── dashboard.py                     ← Canlı eğitim izleme (Flask)
│
├── datasets/                        ← Ham veri setleri (.gitignore'da)
│   ├── classification-of-12-lead-ecgs-.../ (Challenge 2020 v1.0.2)
│   └── a-large-scale-12-lead-.../          (ECG Arrhythmia v1.0.0)
│
├── ptb-xl-.../                      ← PTB-XL ham verisi (.gitignore'da)
│
└── outputs/
    ├── processed_data/
    │   ├── unified_manifest.csv       (87,520 — ham birleşik)
    │   ├── unified_manifest_clean.csv (86,539 — kalite sonrası)
    │   ├── filtered_manifest.csv      (86,539 — filtreleme sonrası)
    │   ├── quality_manifest.csv       (81,006 — sinyal QC sonrası)
    │   ├── segmented_manifest.csv     (81,006 — segmentasyon sonrası)
    │   ├── train_manifest.csv         (56,704)
    │   ├── train_manifest_smote.csv   (61,277 — SMOTE sonrası)
    │   ├── val_manifest.csv           (12,151)
    │   ├── test_manifest.csv          (12,151)
    │   ├── class_weights.npy
    │   └── segmented_signals/         (81,006 .npy dosyası)
    ├── checkpoints/
    │   └── best_model.pth             ← Eğitilmiş model
    └── reports/
        ├── test_metrics.json          ← Son model test sonuçları
        ├── quality_report.txt
        ├── confusion_matrix.png
        └── roc_curves.png
```

---

## 11. Nasıl Çalıştırılır?

### Sıfırdan Tam Pipeline (Veri işlemeden eğitime)
```powershell
.\env\Scripts\python.exe baslat.py
```

### Sadece Eğitim (Veri işleme zaten tamamsa)
```powershell
.\env\Scripts\python.exe baslat.py --sadece-egitim
```

### Adımları Tek Tek Çalıştırma
```powershell
.\env\Scripts\python.exe adim00_veri_birlestirme.py
.\env\Scripts\python.exe adim01_kalite_kontrol_genel.py
.\env\Scripts\python.exe adim02_filtreleme.py
.\env\Scripts\python.exe adim03_kalite_kontrol.py
.\env\Scripts\python.exe adim04_segmentasyon.py
.\env\Scripts\python.exe adim06_veri_bolme.py
.\env\Scripts\python.exe adim06b_oversampling.py
.\env\Scripts\python.exe adim08_egitim.py
.\env\Scripts\python.exe adim09_degerlendirme.py
.\env\Scripts\python.exe adim10_gradcam.py
```

---

## 12. Sistem Bilgileri

| Bileşen | Değer |
|---------|-------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8.6 GB VRAM) |
| CUDA | 12.4 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.12.7 |
| İşletim Sistemi | Windows |
| Disk (Veri) | ~25 GB (datasets + processed) |

---

## 13. Mevcut Durum ve Sonraki Adımlar

### Tamamlanan ✅
- [x] 5 veri seti entegrasyonu (87K+ kayıt)
- [x] Çapraz veri seti kalite kontrol ve mükerrer temizliği
- [x] Dataset-agnostik filtreleme, QC, segmentasyon pipeline'ı
- [x] PCA-SMOTE ile Ritim oversampling
- [x] Stratified train/val/test bölme
- [x] 87K veri ile model eğitimi (Macro F1: 0.7829)
- [x] Test değerlendirmesi ve raporlama
- [x] GradCAM (XAI) desteği
- [x] Canlı eğitim dashboard'u

### Geliştirilmesi Gereken 🔄
- [ ] **Ritim F1'ini artırma** (şu an 0.6746 — en zayıf sınıf)
  - Daha fazla gerçek Ritim verisi aranabilir
  - Augmentasyon stratejisi iyileştirilebilir
- [ ] **Val Loss stabilitesi** — train/val arasındaki gap kapatılabilir
  - Daha güçlü dropout veya regularizasyon denenebilir
- [ ] **Hiperparametre optimizasyonu** — Optuna ile otomatik arama
- [ ] **Model ensemble** — Birden fazla modelin tahminini birleştirme

---

## 14. SNOMED-CT Etiket Eşleştirme Mantığı

Tüm veri setleri farklı tanı kodları kullanıyor. Bunlar `config.py` içindeki büyük eşleştirme tablosu ile 3 sınıfa indirgeniyor:

```python
# Örnekler:
SNOMED_TO_LABEL = {
    426783006: 0,  # Normal sinus rhythm → Normal
    164889003: 1,  # Atrial fibrillation → Ritim
    713422000: 1,  # Atrial flutter → Ritim
    164909002: 2,  # LBBB → İletim
    713427006: 2,  # RBBB → İletim
    270492004: 2,  # 1st degree AV block → İletim
    # ... toplam ~100+ kod
}
```

Birden fazla tanı kodu varsa **en yüksek öncelikli sınıf** atanır.
