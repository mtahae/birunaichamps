# BirunAI EKG Sınıflandırma — Proje Dokümantasyonu

## 1. Projenin Amacı

TEKNOFEST Sağlıkta Yapay Zeka yarışması kapsamında, **12 derivasyonlu EKG sinyallerinden** otomatik tanı koyabilen bir derin öğrenme sistemi geliştiriyoruz. Sistem, bir EKG kaydını 3 sınıftan birine atar:

| Sınıf | Açıklama | Klinik Örnek |
|-------|----------|--------------|
| 0 — Normal | Sağlıklı kalp ritmi | Sinüs ritmi, sinüs bradikardisi |
| 1 — Ritim Bozukluğu | Kalbin düzensiz atması | Atriyal fibrilasyon (AFIB), SVT, PVC |
| 2 — İletim Bozukluğu | Elektrik iletim sistemi hasarı | Dal blokları (RBBB/LBBB), AV blok, ST değişiklikleri |

**Neden 3 sınıf?** Yarışma formatı bu şekilde tanımlanmış. Klinik pratikte yüzlerce tanı var ama biz bunları 3 ana kategoriye indirgiyoruz.

---

## 2. Veri Seti: PTB-XL

**Kaynak:** PhysioNet PTB-XL v1.0.3  
**Toplam kayıt:** 21,799 EKG  
**Hasta sayısı:** 18,869 benzersiz hasta  
**Her kayıt:** 12 kanal × 10 saniye × 500 Hz = 12 × 5000 sayısal değer  

### Neden PTB-XL?
- Dünyanın en büyük açık kaynaklı 12 derivasyonlu EKG veri setidir
- Her kayıt **kardiyolog tarafından etiketlenmiş** (SCP kodları ile)
- Hasta bazlı `strat_fold` sütunu var → veri sızıntısız bölme yapabiliyoruz
- TEKNOFEST'in önerdiği veri setlerinden biri

### Dosya Yapısı (Ham Veri)
```
ptb-xl-.../
├── ptbxl_database.csv      ← 21,799 satırlık metadata (hasta ID, tanı kodları, fold)
├── scp_statements.csv      ← SCP tanı kodlarının açıklamaları
└── records500/             ← Ham EKG dosyaları (.dat + .hea çiftleri)
    ├── 00000/
    │   ├── 00001_hr.dat    ← İkili sinyal verisi
    │   ├── 00001_hr.hea    ← Başlık dosyası (kanal isimleri, örnekleme frekansı)
    │   └── ...
    └── 21000/
```

---

## 3. Proje Dosya Yapısı

```
d:\python\birunai\
│
├── config.py                    ← TÜM parametreler burada (tek kaynak)
│
├── adim01_veri_yukleme.py       ← Ham veriyi okur, etiketler
├── adim02_filtreleme.py         ← Sinyal işleme (filtre + normalizasyon)
├── adim03_kalite_kontrol.py     ← Bozuk kayıtları eler
├── adim04_segmentasyon.py       ← Sabit pencereye getirir
├── adim05_ozellik_cikarma.py    ← İstatistiksel özellikler çıkarır
├── adim06_veri_bolme.py         ← Train/Val/Test ayırır
├── adim07_model_mimarisi.py     ← CNN + LSTM + Attention modeli
├── adim08_egitim.py             ← Eğitim döngüsü
├── adim09_degerlendirme.py      ← Test metrikleri
├── adim10_gradcam.py            ← Açıklanabilir YZ (XAI)
│
├── ana_pipeline.py              ← Tüm adımları sırayla çalıştırır
├── baslat.py                    ← TEK TIKLA eğitim başlatıcı
├── dashboard.py                 ← Canlı eğitim izleme arayüzü
│
└── outputs/
    ├── processed_data/
    │   ├── raw_manifest.csv
    │   ├── filtered_manifest.csv
    │   ├── quality_manifest.csv
    │   ├── segmented_manifest.csv
    │   ├── train_manifest.csv / val_manifest.csv / test_manifest.csv
    │   ├── class_weights.npy
    │   ├── features.csv
    │   └── segmented_signals/   ← 21,769 adet .npy dosyası
    ├── checkpoints/
    │   └── best_model.pth       ← (eğitim sonrası oluşacak)
    └── reports/                 ← (değerlendirme sonrası oluşacak)
```

---

## 4. config.py — Merkezi Konfigürasyon

**Neden tek config?** Bir parametreyi değiştirmek istediğinde (ör. learning rate) 10 dosyayı aramak yerine tek dosyadan yaparsın. Jüri de "bu proje nasıl yönetilmiş?" diye baktığında profesyonel bulur.

### Kritik Parametreler

| Parametre | Değer | Neden Bu Değer? |
|-----------|-------|-----------------|
| `TARGET_FS` | 250 Hz | Klinik EKG bandı 0.5-40 Hz. Nyquist'e göre 80 Hz yeter, 250 Hz güvenli. VRAM %50 azalır |
| `WINDOW_SEC` | 10 sn | Klinik standart EKG çekim süresi. 2-3 tam P-QRS-T döngüsü içerir |
| `BANDPASS_LOW/HIGH` | 0.5 / 40 Hz | 0.5 Hz: taban çizgisi kaymasını eler. 40 Hz: EMG + şebeke gürültüsünü eler |
| `CNN_FILTERS` | [64, 128, 256] | Artan filtre = artan soyutlama seviyesi |
| `LSTM_HIDDEN_SIZE` | 128 | BiLSTM → çıkış 256. Yeterli kapasite, VRAM dostu |
| `FOCAL_LOSS_GAMMA` | 2.0 | Kolay örneklerin etkisini 0.25x'e düşürür, zor örneklere odaklanır |
| `BATCH_SIZE` | 32 | 8 GB VRAM sınırı ve gradyan kararlılığı dengesi |
| `EARLY_STOPPING_PATIENCE` | 10 | 10 epoch iyileşme yoksa eğitimi durdur (overfitting önleme) |

### Etiket Eşleştirme Mantığı

PTB-XL'de her kayıt SCP kodları ile etiketlenmiş (ör: `{"AFIB": 100.0, "SR": 0.0}`). Biz bunları 3 sınıfa eşliyoruz:

```
SCP Kodu → Sınıf Eşleştirme Örnekleri:
  NORM, SR, SBRAD, STACH  → 0 (Normal)
  AFIB, AFLT, SVTAC, PVC  → 1 (Ritim Bozukluğu)
  CRBBB, CLBBB, 1AVB, IMI → 2 (İletim Bozukluğu)
```

**Çakışma durumunda öncelik:** İletim (2) > Ritim (1) > Normal (0)  
**Gerekçe:** Patolojik bulgu klinik olarak daha kritiktir.

---

## 5. Adım 1 — Veri Yükleme (`adim01_veri_yukleme.py`)

### Ne Yapıyor?
1. `ptbxl_database.csv` dosyasını okur (21,799 satır)
2. Her satırdaki `scp_codes` sütununu parse eder (string → Python dict)
3. Dict'teki kodları `SCP_TO_LABEL` tablosuna göre 3 sınıfa eşler
4. `.dat` + `.hea` dosya çiftlerinin varlığını kontrol eder
5. `raw_manifest.csv` üretir

### Çıktı: `raw_manifest.csv`
| Sütun | Açıklama |
|-------|----------|
| ecg_id | Kayıt ID'si (1-21799) |
| filename_hr | Dosya yolu (records500/00000/00001_hr) |
| label | 0, 1 veya 2 |
| patient_id | Hasta ID'si |
| strat_fold | PTB-XL fold numarası (1-10) |

### Elde Edilen Sonuç
```
Toplam kayıt    : 21,799
Etiketlenemeyen : 0
Dosyası eksik   : 0
Sınıf 0 (Normal): 8,089 (%37.1)
Sınıf 1 (Ritim) :   687 (%3.2)
Sınıf 2 (İletim): 13,023 (%59.7)
```

**Gözlem:** Ritim Bozukluğu sadece %3.2 — ciddi sınıf dengesizliği var. Bu sorun Adım 6'da (Focal Loss + Weighted Sampling) ele alınıyor.

---

## 6. Adım 2 — Filtreleme (`adim02_filtreleme.py`)

### Ne Yapıyor?
1. **wfdb kütüphanesi** ile ham .dat dosyalarını okur → (12, 5000) numpy array
2. **Alt örnekleme:** 500 Hz → 250 Hz (`scipy.signal.resample`) → (12, 2500)
3. **Butterworth Bandpass Filtre:** 0.5-40 Hz, 4. derece, `filtfilt` (sıfır faz kayması)
4. **Z-score normalizasyon:** Her kanal ayrı ayrı → ortalam=0, std=1
5. `.npy` dosyası olarak kaydeder, `filtered_manifest.csv` üretir

### Neden Bu İşlemler?

| İşlem | Problem | Çözüm |
|-------|---------|-------|
| Alt örnekleme | 5000 örnek/kayıt → VRAM israfı | 2500 örneğe düşür, bilgi kaybı yok |
| Bandpass 0.5 Hz | Solunum + hareket → taban çizgisi kayması | Düşük frekans gürültüsünü eler |
| Bandpass 40 Hz | Kas + elektronik → yüksek frekans gürültüsü | 50 Hz şebeke gürültüsünü de eler |
| Z-score | Farklı cihazların farklı voltaj aralıkları | Tüm kanalları aynı ölçeğe getirir |
| filtfilt | Standart filtre sinyal zamanlamasını bozar | İleri-geri filtre → faz kayması SIFIR |

### Neden `filtfilt`?
EKG'de P-QRS-T dalgalarının **zamanlaması** kritiktir (PR aralığı, QT süresi). Normal filtre bu zamanlamayı kaydırır. `filtfilt` sinyali iki kez filtreler (ileri + geri) → net faz kayması sıfır olur.

### Çıktı
```
Başarılı: 21,799 | Hatalı: 0
Her sinyal: (12, 2500) float32
Disk: ~2.5 GB (outputs/processed_data/filtered_signals/)
```

---

## 7. Adım 3 — Kalite Kontrol (`adim03_kalite_kontrol.py`)

### Ne Yapıyor?
Her filtrelenmiş sinyali 3 teste tabi tutar:

| Test | Kriter | Gerekçe |
|------|--------|---------|
| Flat-line | Kanalın std < 0.01 ise düz sinyal | Elektrot teması yok demek |
| Clipping | Sinyalin >%5'i ±20'yi aşıyorsa | ADC doygunluğu, sinyal bozuk |
| Elektrot | `electrodes_problems` sütunu boş değilse | Metadata'da bilinen problem |

### Çıktı
```
QC geçen : 21,769 (%99.9)
QC kalan :     30 (hepsi elektrot problemi)
```

**Yorum:** %99.9 geçiş oranı, PTB-XL'in kaliteli bir veri seti olduğunu doğrular. Elenen 30 kayıt, elektrot bağlantı problemi olan kayıtlardır.

---

## 8. Adım 4 — Segmentasyon (`adim04_segmentasyon.py`)

### Ne Yapıyor?
Tüm sinyalleri **tam olarak (12, 2500)** boyutuna getirir:
- Kısa sinyaller → sıfır ile doldurulur (zero-padding)
- Uzun sinyaller → ortadan kırpılır (center-crop)
- Tam boyutlu → olduğu gibi bırakılır

### Neden Sabit Boyut?
PyTorch'ta bir batch içindeki tüm tensorlerin **aynı boyutta** olması zorunludur. CNN'nin Conv1d katmanları da sabit girdi boyutu bekler.

### Çıktı
```
Başarılı  : 21,769
Tam boyut : 21,769 (hepsı zaten 10 saniye)
Padding   : 0
Cropping  : 0
Disk      : 2,493.9 MB (outputs/processed_data/segmented_signals/)
```

Her dosya: `{ecg_id}.npy` → (12, 2500) float32 numpy array

---

## 9. Adım 5 — Özellik Çıkarma (`adim05_ozellik_cikarma.py`)

### Ne Yapıyor?
Her kayıttan **123 sayısal özellik** çıkarır. CNN kendi özelliklerini öğrenir ama bu özellikler EDA (keşifsel analiz) ve potansiyel feature-augmented model denemeleri içindir.

### Özellik Grupları

| Grup | Sayı | Örnekler |
|------|------|----------|
| İstatistiksel | 108 (12 kanal × 9) | mean, std, min, max, skewness, kurtosis, ptp, rms, zcr |
| Morfolojik | 8 | R-pike sayısı, RR aralık mean/std, kalp hızı (BPM), RMSSD (HRV) |
| Frekans Alanı | 7 | VLF/LF/HF/QRS band gücü, baskın frekans, LF/HF oranı |

### Klinik Doğrulama
```
Sınıfa Göre Ortalama Kalp Hızı (BPM):
  Normal            : 79.2 BPM  ← Sağlıklı yetişkin normu (60-100)
  Ritim Bozukluğu   : 97.3 BPM  ← Beklenen: aritmi → hızlı/düzensiz
  İletim Bozukluğu  : 93.2 BPM  ← Beklenen: kompansatuar taşikardi
```

Bu sonuçlar modelimizin öğrenmesi gereken paternlerin **gerçekten veride var olduğunu** doğrular.

### Çıktı
- `features.csv` — 21,769 × 124 matris (ecg_id + 123 özellik)
- `feature_stats.csv` — Her özelliğin describe() istatistikleri

---

## 10. Adım 6 — Veri Bölme (`adim06_veri_bolme.py`)

### Ne Yapıyor?
PTB-XL'in kendi `strat_fold` sütununu kullanarak **hasta bazlı** bölme yapar:

| Set | Fold | Kayıt | Oran |
|-----|------|-------|------|
| Train | 1-7 | 15,224 | %69.9 |
| Validation | 8 | 2,171 | %10.0 |
| Test | 9-10 | 4,374 | %20.1 |

### Neden Hasta Bazlı?
Aynı hastanın birden fazla EKG kaydı olabilir. Eğer bir hastanın bir kaydı train'de, diğeri test'te olursa → model **hastayı ezberler**, hastalığı öğrenmez. Buna **veri sızıntısı (data leakage)** denir.

### Veri Sızıntısı Kontrolü
```
Train-Val hasta çakışması  : 0  ✅
Train-Test hasta çakışması : 0  ✅
Val-Test hasta çakışması   : 0  ✅
```

### Sınıf Ağırlıkları
Dengesizliği ele almak için **inverse frequency** ağırlıkları hesaplandı:

| Sınıf | Train'de | Ağırlık | Yorum |
|-------|----------|---------|-------|
| Normal | 5,721 | 0.887 | Çoğunluk → düşük ağırlık |
| Ritim Boz. | 476 | **10.661** | Azınlık → 10x yüksek ağırlık! |
| İletim Boz. | 9,027 | 0.562 | En çok → en düşük ağırlık |

Bu ağırlıklar `class_weights.npy` olarak kaydedildi → eğitimde Focal Loss ve WeightedRandomSampler tarafından kullanılacak.

### Çıktı
- `train_manifest.csv` (15,224 kayıt)
- `val_manifest.csv` (2,171 kayıt)
- `test_manifest.csv` (4,374 kayıt)
- `class_weights.npy` — [0.887, 10.661, 0.562]

---

## 11. Adım 7 — Model Mimarisi (`adim07_model_mimarisi.py`)

### Hibrit Mimari: 1D-CNN + BiLSTM + Attention

```
Girdi: (batch, 12, 2500) — 12 derivasyon, 10 saniye, 250 Hz
  │
  ▼
[CNN Blok 1] Conv1d(12→64, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  → (batch, 64, 1250)
  │
[CNN Blok 2] Conv1d(64→128, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  → (batch, 128, 625)
  │
[CNN Blok 3] Conv1d(128→256, k=7) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  → (batch, 256, 312)
  │
  ▼ permute(0,2,1)
[BiLSTM] 2 katman, 128 hidden, bidirectional
  → (batch, 312, 256)
  │
[Self-Attention] Linear→Tanh→Linear→Softmax→Weighted Sum
  → (batch, 256) + attention weights
  │
[FC] Linear(256→128) + ReLU + Dropout + Linear(128→3)
  → (batch, 3) logits
```

### Neden Bu Mimari?

| Bileşen | Ne Öğreniyor? | Klinik Karşılığı |
|---------|---------------|------------------|
| **1D-CNN** | Yerel morfoloji (dalga şekilleri) | P dalgası genişliği, QRS süresi, T dalgası yüksekliği |
| **BiLSTM** | Zamansal bağımlılık (ritim düzeni) | RR aralığı düzenliliği, P-QRS ilişkisi |
| **Attention** | Hangi zaman dilimine odaklanmalı? | "Bu tanı için QRS kompleksine baktım" → XAI |

### Model İstatistikleri
```
Toplam parametre   : 1,150,276
Eğitilebilir param : 1,150,276
Model boyutu (GPU) : ~4.4 MB
```

### Focal Loss
Standart Cross-Entropy yerine **Focal Loss** kullanıyoruz:

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

γ = 0 → Standart CE (kolay+zor eşit)
γ = 2 → Kolay örneğin katkısı ~0.25x'e düşer, zor örneğe odaklanır
α_t   → Sınıf ağırlığı (Ritim Boz. = 10.66x)
```

---

## 12. Adım 8-10 — Eğitim, Değerlendirme, GradCAM

Bu adımlar **henüz çalıştırılmadı** — yarın çalıştırılacak.

### Adım 8: Eğitim Stratejisi
- **Focal Loss** (γ=2.0) + sınıf ağırlıkları
- **WeightedRandomSampler**: Her batch'te sınıflar dengelenir
- **AdamW** optimizer (lr=0.001, weight_decay=1e-5)
- **CosineAnnealingLR**: LR'yi kosinüs eğrisiyle azaltır
- **Gradient Clipping** (max_norm=1.0): BiLSTM gradyan patlamasını önler
- **Mixed Precision (AMP)**: RTX 3060 Ti'de VRAM + hız kazanımı
- **Early Stopping**: Val Macro F1, 10 epoch iyileşmezse dur

### Adım 9: Test Metrikleri
- Macro F1 Score (birincil metrik)
- Per-class Precision, Recall, F1
- Confusion Matrix (normalize)
- ROC-AUC (One-vs-Rest)
- Cohen's Kappa

### Adım 10: 1D-GradCAM (XAI)
Son CNN katmanının gradyanlarını kullanarak EKG üzerinde **ısı haritası** üretir. Jüri savunmasında "modelimiz bu tanıyı koyarken sinyalin şu bölgesine odaklandı" diyebiliriz.

---

## 13. Dashboard (`dashboard.py`)

Eğitim sırasında `http://localhost:5000` adresinden canlı izleme:
- Loss & F1 grafikleri (Chart.js, her 2 saniyede güncellenir)
- Sınıf bazlı F1 metrikleri
- İlerleme çubuğu + tahmini kalan süre
- Early Stopping sayacı

---

## 14. Sistem Gereksinimleri

| Bileşen | Mevcut | Durum |
|---------|--------|-------|
| GPU | RTX 3060 Ti (8 GB) | ✅ |
| PyTorch | 2.5.1+cu121 | ✅ (yeni kuruldu) |
| Python | 3.10 + 3.12 | ✅ |
| Disk | ~5 GB (veri + çıktılar) | ✅ |
| RAM | 16+ GB önerilir | - |

---

## 15. Yarın İçin Çalıştırma Komutu

```bash
python baslat.py
```

Bu tek komut: GPU kontrol → Dashboard aç → Eğit → Değerlendir → GradCAM üret.
Tahmini süre: **RTX 3060 Ti ile ~20-30 dakika.**
