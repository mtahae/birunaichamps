# ⚠️ TEKNOFEST 2026 — Kritik Teknik Notlar ve Dikkat Edilmesi Gerekenler

**CardioFusion-Net Projesi — Takım İçi Referans Dokümanı**

---

## 🔴 BÖLÜM 1: Z-SCORE NORMALİZASYONU (En Kritik Madde)

### ❌ YANLIŞ: Global Z-Score

```python
# BUNU ASLA YAPMAYIN
mean = np.mean(signal)  # Tüm 12 lead'i birlikte ortalama
std = np.std(signal)
normalized = (signal - mean) / std
```

**Neden yanlış?**
V1'deki QRS kompleksi 0.5 mV, V5'teki 2.5 mV olabilir. Global ortalama alırsan V1'in sinyali "yok" olur, V5 baskın hale gelir. RBBB tanısı V1'e bağlıdır, V1'i yok edersen model RBBB öğrenemez. LBBB tanısı V6'ya bağlıdır — V6'yı baskılarsan LBBB öğrenemez.

### ✅ DOĞRU: Lead-Wise (Derivasyon Bazında) Z-Score

```python
# BUNU YAPIN
normalized = np.zeros_like(signal)  # (5000, 12)
for lead in range(12):
    mean = np.mean(signal[:, lead])
    std = np.std(signal[:, lead])
    if std > 1e-6:  # Bölü sıfır kontrolü
        normalized[:, lead] = (signal[:, lead] - mean) / std
    else:
        normalized[:, lead] = signal[:, lead]  # Flatline ise dokunma
```

**Kural:** 12 lead'in her biri kendi içinde normalize edilir. Lead 1'in ortalaması Lead 2'yi etkilemez.

### ⚠️ Ek Kural: Z-Score İstatistikleri SADECE Train Setinden Hesaplanır

```python
# Train setinden hesapla
train_mean = np.mean(train_signals, axis=0)  # (12,)
train_std = np.std(train_signals, axis=0)    # (12,)

# Val ve Test'e uygula (kendi istatistiklerini hesaplama!)
val_normalized = (val_signals - train_mean) / train_std
test_normalized = (test_signals - train_mean) / train_std
```

**Neden?** Validation/test setinin kendi ortalamasını kullanırsan data leakage olur. Model test hakkında bilgi sızmış olur.

**Kritik Kurallar:**
1. Her lead kendi içinde normalize edilir
2. Lead 1'in ortalaması Lead 2'yi etkilemez
3. Train istatistikleri val/test'e uygulanır
4. std = 0 olan lead için bölme yapma

---

## 🔴 BÖLÜM 2: PER-PATIENT SPLIT (Hasta Bazında Bölme)

### ❌ YANLIŞ: Rastgele Bölme

```python
# BUNU ASLA YAPMAYIN
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2)  # HASTA SIZINTISI!
```

**Neden yanlış?** Aynı hastanın 5 farklı EKG kaydı varsa, 4'ü train 1'i test'e gidebilir. Model o hastayı "ezberlemiş" olur. Yarışma finalinde yeni hasta gördüğünde patlar.

### ✅ DOĞRU: Hasta ID'ye Göre Bölme

```python
# BUNU YAPIN
import pandas as pd
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('validation.csv')
test_df = pd.read_csv('test_public.csv')

# KONTROL ET: Aynı hasta farklı CSV'lerde mi?
if 'patient_id' in train_df.columns:
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    overlap = train_patients & val_patients
    assert len(overlap) == 0, f"Train-Val overlap: {len(overlap)} hasta"
    print("Split kontrolü BAŞARILI")
```

Alternatif (manuel bölme):

```python
unique_patients = df['patient_id'].unique()
np.random.shuffle(unique_patients)

train_patients = unique_patients[:int(0.7 * len(unique_patients))]
val_patients = unique_patients[int(0.7 * len(unique_patients)):int(0.85 * len(unique_patients))]
test_patients = unique_patients[int(0.85 * len(unique_patients)):]

train_df = df[df['patient_id'].isin(train_patients)]
val_df = df[df['patient_id'].isin(val_patients)]
test_df = df[df['patient_id'].isin(test_patients)]
```

**Kural:** Bir hasta ya train'tedir, ya val'dedir, ya test'tedir. Asla ikisinde birden olmaz.

---

## 🔴 BÖLÜM 3: AUGMENTASYON KURALLARI (Ne Yapılır, Ne Yapılmaz)

### ✅ YAPILACAKLAR

**1. Amplitude Scale (Lead-Wise):** Her lead'i 0.9–1.1 arası rastgele çarp.

```python
def augment_amplitude(signal):
    scale = np.random.uniform(0.9, 1.1, size=12)
    return signal * scale  # (5000, 12) * (12,) → broadcasting
```

**2. Gaussian Noise (Lead-Wise):** SNR > 20 dB olacak şekilde.

```python
def augment_noise(signal, snr_db=20):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise
```

**3. Lead Dropout:** Rastgele 1-2 lead'i sıfırla (SQI=0 yap).

```python
def augment_lead_dropout(signal, n_dropout=1):
    augmented = signal.copy()
    dropout_leads = np.random.choice(12, size=n_dropout, replace=False)
    augmented[:, dropout_leads] = 0
    return augmented
```

**Neden?** Model bazen V1 düşük kaliteli gelirse bile diğer lead'lere güvenmeyi öğrenir.

### ❌ ASLA YAPILMAYACAKLAR

**1. Zaman Kaydırma (Time-Shift / Circular Shift):**

```python
# BUNU ASLA YAPMAYIN
shifted = np.roll(signal, shift=100, axis=0)
```

**Neden?** P-QRS-T temporal ilişkisini bozar. P dalgası QRS'ten önce gelmelidir. Kaydırırsan bu fizyolojik sıra bozulur.

**2. Rastgele Crop:** EKG'nin başını veya sonunu rastgele kesmek.

```python
# BUNU ASLA YAPMAYIN
cropped = signal[random_start:random_start+4000, :]
```

**Neden?** Baştaki P dalgası veya sondaki T dalgası kaybolabilir. Ritim analizi bozulur.

**3. Global Amplitude Scale:** Tüm 12 lead'i aynı çarpanla çarpmak.

```python
# BUNU ASLA YAPMAYIN
augmented = signal * 1.1  # Tüm lead'ler aynı
```

**Neden?** Lead'ler arası oran bozulur. V1/V6 oranı RBBB/LBBB tanısı için kritiktir.

---

## 🔴 BÖLÜM 4: FİLTRELEME KURALLARI

### ✅ Bandpass 0.5 – 40 Hz

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=500, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered = np.zeros_like(signal)
    for lead in range(12):
        filtered[:, lead] = filtfilt(b, a, signal[:, lead])
    return filtered
```

**Neden 0.5 Hz?** Baseline wander (nefes/konuşma artefaktı) kaldırır.
**Neden 40 Hz?** EMG gürültüsünü ve güç hattı gürültüsünü kaldırır. 50 Hz notch filter kullanmayın (T dalgasına zarar verebilir).
**Neden 4. derece?** Düşük geçiş bantlı, faz bozulması minimal.

### ⚠️ Filtreleme Sonrası Kontrol

Her zaman 1 örnek çizip kontrol edin:
- P dalgası hâlâ görünür mü? (Eğer kayboldıysa filtre çok agresif)
- QRS genişliği bozulmadı mı?
- ST segment düz mü? (Baseline wander kalktı mı?)

---

## 🔴 BÖLÜM 5: 5 SINIFIN FİZYOLOJİK ÖZELLİKLERİ

### AFIB (Atrial Fibrillation)
- **RR interval:** Düzensiz (irregularly irregular)
- **P-wave:** Yok, f-wave (kaotik)
- **Kritik lead:** Lead II
- **Ayırt edici:** RR variance YÜKSEK
- **Karıştığı:** AFL (en sık)

### AFL (Atrial Flutter)
- **RR interval:** Düzenli
- **P-wave:** Testere dişi flutter wave
- **Atrial rate:** 250-350 (düzenli)
- **Kritik lead:** Lead II, V1
- **Ayırt edici:** RR variance DÜŞÜK
- **Karıştığı:** AFIB (en sık)

### LBBB (Left Bundle Branch Block)
- **QRS width:** >120 ms
- **V1:** rS veya QS
- **V6:** Geniş monofazik R
- **I, aVL:** Geniş R
- **Kritik lead:** V1, V6, I, aVL
- **Karıştığı:** RBBB

### RBBB (Right Bundle Branch Block)
- **QRS width:** >120 ms
- **V1:** rsR' (tavşan kulağı) — EN KRİTİK
- **V6:** Geniş S dalgası
- **I, aVL:** Geniş S
- **Kritik lead:** V1 (en önemlisi)
- **Karıştığı:** LBBB

### Normal
- **RR interval:** Düzenli, 60-100 bpm
- **P-wave:** Her QRS'ten önce, düzenli
- **QRS width:** 80-120 ms
- **PR interval:** 120-200 ms

---

## 🔴 BÖLÜM 6: MACRO F1 MAKSİMİZASYONU (Tek Önemli Metrik)

### ❌ Accuracy'ye Aldanmayın

```
Accuracy: 95%  ← Aldanmayın! Normal sınıfını ezberlemiş olabilir.
Macro F1: 72%  ← Gerçek skor bu. APB/VPB 0.30 F1 ise buraya yansır.
```

### ✅ Sınıf Bazında F1 Takibi

```python
from sklearn.metrics import f1_score

macro_f1 = f1_score(y_true, y_pred, average='macro')
per_class_f1 = f1_score(y_true, y_pred, average=None)

class_names = ['Normal', 'AFIB', 'AFL', 'LBBB', 'RBBB']
for name, f1 in zip(class_names, per_class_f1):
    print(f"{name}: {f1:.3f}")
```

**Hedef:** Her sınıf F1'i > 0.85 olmalı.

### Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        return loss.mean()
```

---

## 🔴 BÖLÜM 7: SQI (SİNYAL KALİTESİ) KURALLARI

### Lead-Wise Kalite Skoru

Her lead için 0-1 arası skor:

```python
def compute_sqi(signal_12lead, fs=500):
    from scipy.stats import kurtosis
    from scipy.signal import welch
    sqis = []
    for lead in range(12):
        s = signal_12lead[:, lead]

        # 1. kSQI: Kurtosis (QRS varsa yüksek)
        k = kurtosis(s)
        k_norm = min(abs(k) / 10, 1.0)  # Normalize

        # 2. bSQI: Baseline wander (düşük olmalı)
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 0.5/(0.5*fs), btype='low')
        baseline = filtfilt(b, a, s)
        b_ratio = np.std(baseline) / (np.std(s) + 1e-6)
        b_norm = max(0, 1 - b_ratio)

        # 3. pSQI: QRS band gücü (5-15 Hz)
        f, psd = welch(s, fs=fs, nperseg=256)
        qrs_mask = (f >= 5) & (f <= 15)
        qrs_power = np.sum(psd[qrs_mask])
        total_power = np.sum(psd[(f >= 0.5) & (f <= 40)]) + 1e-6
        p_norm = qrs_power / total_power

        # 4. rSQI: R-peak detection başarısı
        try:
            import neurokit2 as nk
            _, info = nk.ecg_process(s, sampling_rate=fs)
            r_peaks = info["ECG_R_Peaks"]
            hr = len(r_peaks) / (len(s)/fs/60)
            r_norm = 1.0 if 40 <= hr <= 150 else 0.3
        except:
            r_norm = 0.0

        sqi = 0.3*k_norm + 0.3*b_norm + 0.2*p_norm + 0.2*r_norm
        sqis.append(sqi)

    return np.array(sqis)  # (12,)
```

### SQI Kullanımı
- **Eğitimde:** SQI < 0.3 olan lead'leri dropout yap (augmentation olarak).
- **Inference'da:** SQI ağırlıklı fusion. Düşük kaliteli lead'in embedding'ini baskıla.
- **Asla atma:** Lead'i tamamen çıkarma. Sadece ağırlığını düşür. 12 lead her zaman orada.

---

## 🔴 BÖLÜM 8: MULTI-TASK LOSS

```python
class HierarchicalLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits_main, logits_aux, labels_main, labels_aux):
        L_main = self.ce(logits_main, labels_main)
        L_aux = self.ce(logits_aux, labels_aux)
        return L_main + self.alpha * L_aux

def get_aux_label(main_label):
    mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
    return mapping[main_label]
```

---

## 🔴 BÖLÜM 9: REPRODUCIBILITY (Tekrar Üretilebilirlik)

### Seed Fix

```python
def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Config YAML

```yaml
seed: 42
sampling_rate: 500
filter_low: 0.5
filter_high: 40
filter_order: 4
n_classes: 5
n_aux_classes: 3
batch_size: 32
lr: 0.001
weight_decay: 1e-4
focal_gamma: 2.0
epochs: 100
patience: 15
```

---

## 🔴 BÖLÜM 10: 12 LEAD ANATOMİ

| Lead | Gördüğü Yer | Kritik Olduğu Sınıf |
|---|---|---|
| V1 | Sağ ventrikül, septum | RBBB (rsR'), LBBB (rS) |
| V2 | Septum | LBBB, RBBB |
| V3-V4 | Anterior duvar | (Yarışmada yok) |
| V5-V6 | Sol ventrikül lateral | LBBB (geniş R) |
| I, aVL | Sol ventrikül yüksek lateral | LBBB |
| II, III, aVF | Alt duvar | AFIB/AFL (P dalgası) |
| aVR | Sağ atriyum | Genelde negatif |

**Kural:**
- RBBB tanısı V1'de yazılıdır
- LBBB tanısı V1+V6'da yazılıdır
- AFIB/AFL tanısı Lead II'de yazılıdır

---

## 🔴 BÖLÜM 11: SON KONTROL LİSTESİ

**Preprocessing**
- [ ] WFDB dosyaları doğru okunuyor mu? (5000, 12)
- [ ] Lead-wise z-score yapıldı mı?
- [ ] Z-score istatistikleri sadece train setinden mi?
- [ ] Verilen split'ler kullanıldı mı?
- [ ] Patient overlap kontrolü yapıldı mı?
- [ ] Bandpass 0.5-40 Hz uygulandı mı?
- [ ] Filtre sonrası P dalgası kontrol edildi mi?

**Model**
- [ ] Multi-task loss çalışıyor mu?
- [ ] Macro F1 hesaplanıyor mu?
- [ ] Sınıf başına F1 izleniyor mu?
- [ ] Cross-lead attention + SQI-gating çalışıyor mu?
- [ ] Ensemble (3 seed) yapıldı mı?
- [ ] TTA var mı?

**Reproducibility**
- [ ] Seed fixlendi mi?
- [ ] config.yaml ve requirements.txt var mı?
- [ ] Dockerfile test edildi mi?
- [ ] Inference script tek komutla çalışıyor mu?

**Dokümantasyon**
- [ ] Teknik rapor yazıldı mı?
- [ ] Sunum slaytları hazır mı?

---

## 🎯 ALT SATIR

**3 Altın Kural:**
1. 12 lead, lead-wise z-score, per-patient split.
2. Macro F1 tek tanrı.
3. Görülmemiş veri mindset'i.

**Başarı formülü:** *"Herkes daha derin CNN koyarken, biz EKG'nin fizyolojisini modelin içine gömdük."*

*Bu dokümanı her kod yazmadan önce okuyun. Bu kurallar ihlal edilirse, model başarısız olur.*

---
---

# TEKNOFEST 2026 — ÇOKLU VERİ SETİ STRATEJİSİ

## CardioFusion-5 Multi-Dataset: 118.000+ Kayıt, 5 Sınıf

**Hedef: Macro F1'de 97+ Puanı Geçmek**

## VERİ SETİ PORTFÖYÜ

| Veri Seti | Kaynak | Toplam Kayıt | Kullanılan | Hz | Format | SNOMED |
|---|---|---|---|---|---|---|
| PhysioNet ECG Arrhythmia | Ana veri | 45.152 | ~35.000 | 500->250 | .mat/.hea | Evet |
| PTB-XL | Almanya | 21.837 | ~18.000 | 500->250 | .mat/.hea | Evet |
| CPSC 2018 + Extra | Çin | 10.330 | ~8.000 | 500->250 | .mat/.hea | Evet |
| Georgia 12-Lead | ABD | 10.344 | ~8.000 | 500->250 | .mat/.hea | Evet |
| Chapman-Shaoxing/Ningbo | Çin | 45.182 | ~35.000 | 500->250 | .mat/.hea | Evet |
| **TOPLAM (Ham)** | | **132.845** | | | | |
| **TOPLAM (Filtreli)** | | **~104.000** | | | | |
| **TOPLAM (5 Sınıf Alt Kümesi)** | | **~85.000** | | | | |
| TEKNOFEST Dengeli Alt Küme | | 5.000 | 5.000 | 500 | .mat/.hea | Evet |

**Not:** 500 Hz -> 250 Hz alt örnekleme yapılmış. Bu durum mimariyi etkiler.

## STRATEJİ DEĞİŞİKLİKLERİ (Çoklu Veri Seti İçin)

### 1. FREKANS UYUMSUZLUĞU — KRİTİK PROBLEM

**Problem:**
- Internet verileri: 250 Hz (siz alt örneklemişsiniz)
- TEKNOFEST verisi: 500 Hz (orijinal)

**Çözüm — İki Yol:**

**Yol A: TEKNOFEST'e Uyum (Önerilen)**
- Internet verilerini 250 Hz -> 500 Hz'e interpolasyon ile yükselt
- Veya TEKNOFEST verisini 500 Hz -> 250 Hz'e düşür
- **Tavsiye:** 250 Hz'e düşürün. Neden?
  - Daha hızlı eğitim
  - 250 Hz'de P-QRS-T hâlâ korunur (Nyquist: 125 Hz, P dalgası ~10 Hz)
  - Daha az RAM kullanımı
  - Internet verileri zaten 250 Hz

**Yol B: Hz-Agnostic Mimari**
- Input boyutu sabit değil, zamansal olarak
- Resample katmanı mimariye dahil
- Her veri seti kendi Hz'siyle girer, model içinde 250 Hz'e düşer

```python
class HzAdaptiveInput(nn.Module):
    def __init__(self, target_fs=250):
        self.target_fs = target_fs

    def forward(self, x, fs):
        # x: (batch, samples, 12)
        # fs: orijinal örnekleme frekansı
        if fs != self.target_fs:
            # Resample
            from scipy.signal import resample_poly
            # ...
        return x
```

**Karar: Yol A — 250 Hz'e düşür.** Daha basit, daha robust.

### 2. VERİ DENGESİZLİĞİ — YENİ STRATEJİ

Internet verilerinde 5 sınıf dengesiz olacak:

| Sınıf | Tahmini Dağılım | Dengesizlik |
|---|---|---|
| Normal | ~40.000 (%47) | Çoğunluk |
| AFIB | ~15.000 (%18) | Orta |
| AFL | ~3.000 (%3.5) | Azınlık |
| LBBB | ~12.000 (%14) | Orta |
| RBBB | ~15.000 (%18) | Orta |

**Çözüm: Aşamalı Eğitim (Curriculum Learning)**

```
Aşama 1 (Epoch 1-20): SADECE TEKNOFEST verisi (5.000, dengeli)
-> Model temel öğrenir

Aşama 2 (Epoch 21-60): TEKNOFEST + Internet verisi (85.000, dengesiz)
-> Class-balanced sampler
-> Focal Loss
-> Hard Example Mining

Aşama 3 (Epoch 61-100): SADECE TEKNOFEST verisi (fine-tune)
-> Domain adaptation
-> Overfit'i engelle
```

**Neden?**
- Aşama 1: Model dengeli veride temel öğrenir
- Aşama 2: Farklı hasta popülasyonlarını öğrenir (domain genelleme)
- Aşama 3: TEKNOFEST veri dağılımına geri dön, overfit'i kır

### 3. DOMAIN ADAPTATION — ÇOKLU HASTANE

Farklı veri setleri = farklı cihazlar + farklı popülasyonlar

| Veri Seti | Cihaz | Popülasyon | Domain Farkı |
|---|---|---|---|
| PhysioNet | Çeşitli | Karma | Referans |
| PTB-XL | Schiller | Alman | Cihaz farkı |
| CPSC | Mindray | Çin | Cihaz + etnik |
| Georgia | GE | ABD | Cihaz + etnik |
| Chapman | Philips | Çin | Cihaz farkı |

**Çözüm: Domain-Adversarial Training (DANN)**

```
Input (250 Hz, 12 lead)
        |
        v
Feature Extractor (CNN)
        |
        +----> Classifier (5 sınıf) -----> L_class
        |
        +----> Domain Classifier (5 veri seti) -----> L_domain (ters gradient)
```

```python
class DANN(nn.Module):
    def __init__(self):
        self.feature_extractor = CNN()
        self.classifier = nn.Linear(128, 5)
        self.domain_classifier = nn.Linear(128, 5)  # 5 veri seti

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        # Sınıf tahmini
        class_logits = self.classifier(features)
        # Domain tahmini (ters gradient)
        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return class_logits, domain_logits
```

**Loss:** `L = L_class + lambda * L_domain`

**Neden?** Feature extractor veri seti bilgisini "unutmaya" zorlanır. Sadece kardiyak bilgi kalır.

### 4. ETİKET HARMONİZASYONU — SNOMED CT

Farklı veri setlerinde farklı SNOMED kodları olabilir:

```python
SNOMED_UNIFIED_MAP = {
    # Ritim Bozuklukları
    '164889003': 'AFIB',  # Atrial Fibrillation
    '164890007': 'AFL',   # Atrial Flutter
    '713422000': 'AFL',   # Atrial Tachycardia -> AFL ile birleştir
    '426761007': 'AFIB',  # SVT -> AFIB ile birleştir (veya ayrı)
    '284470004': 'AFIB',  # APB -> AFIB ile birleştir (veya ayrı)
    '17338001': 'AFIB',   # VPB -> AFIB ile birleştir (veya ayrı)

    # İletim Bozuklukları
    '164909002': 'LBBB',  # Left Bundle Branch Block
    '59118001': 'RBBB',   # Right Bundle Branch Block
    '270492004': 'LBBB',  # 1AVB -> LBBB ile birleştir
    '195042002': 'LBBB',  # 2AVB -> LBBB ile birleştir
    '27885002': 'LBBB',   # 3AVB -> LBBB ile birleştir

    # Normal
    '426783006': 'Normal',  # Sinus Rhythm
    '426177001': 'Normal',  # Sinus Bradycardia -> Normal
    '427084000': 'Normal',  # Sinus Tachycardia -> Normal
}
```

**NOT:** Yukarıdaki birleştirmeler ÖRNEKTİR. Gerçek birleştirmeleri veri seti analizinden sonra yapmalısınız.

**Karar Ağacı:**
```
SNOMED kodu -> Üst sınıf (Ritim/İletim/Normal)
            -> Alt sınıf (AFIB/AFL/LBBB/RBBB/Normal)
            -> Eğer uyuşmazlık var -> En sık görülen etiketi al
            -> Eğer hâlâ belirsiz -> Kaydı at
```

## REVİZE MİMARİ: CardioFusion-5 Multi-Dataset

```
GİRDİ: 12 Lead x 10sn @ 250 Hz veya 500 Hz
(2500 veya 5000, 12)
        |
        v
+---------------------+
|   Hz ADAPTATION     |
|   (500->250 Hz      |
|    resample)        |
+----------+----------+
        |
        +----------+----------+
        |                     |
        v                     v
+-------------+       +-------------+
|  PREPROC    |       |   DOMAIN    |
|             |       |  EMBEDDING  |
|  Bandpass   |       |  (5 veri    |
|  0.5-40Hz   |       |   seti ID)  |
|  Lead-wise  |       |             |
|  Z-Score    |       | Output: (5,)|
+------+------+       +------+------+
        |                     |
        +---------+---------+
                  |
                  v
        +---------------------+
        |  FEATURE EXTRACTOR  |
        |   (1D-CNN + SE)     |
        |                     |
        |  Output: (128,)     |
        +----------+----------+
                  |
        +----------+----------+
        |                     |
        v                     v
+-------------+       +-------------+
| CLASSIFIER  |       |   DOMAIN    |
| (5 sınıf)   |       | CLASSIFIER  |
|             |       | (5 veri     |
| Softmax     |       |  seti)      |
| L_class     |       | L_domain    |
+-------------+       +-------------+
```

**Loss:** `L = L_class + 0.1 * L_domain + 0.3 * L_aux`

## 8 HAFTALIK REVİZE PLAN (Çoklu Veri Seti)

### HAFTA 1: Çoklu Veri Seti Entegrasyonu

**Hedef:** Tüm veri setlerini indir, harmonize et, 250 Hz'e düşür.

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | Tüm veri setlerini indir (PhysioNet, PTB-XL, CPSC, Georgia, Chapman) | Veri Mühendisi | İndirme tamam |
| Sal | SNOMED harmonizasyonu: Tüm veri setlerinden 5 sınıf için SNOMED kodlarını bul | Veri Mühendisi | snomed_map.py |
| Çar | Hz uyumu: 500 Hz -> 250 Hz resample (scipy.signal.resample_poly) | Sinyal Mühendisi | resample_250hz.py |
| Per | Veri kalite filtresi: SQI < 0.3 olan kayıtları at | Sinyal Mühendisi | quality_filter.py |
| Cum | Sınıf dağılımı analizi: Her veri setinde 5 sınıf kaç kayıt? | Veri Mühendisi | class_distribution.csv |
| Cmt | Quiz 1: Çoklu veri seti challenge'ları | Herkes | Kahoot |
| Paz | Review: Veri seti harmoni raporu | Takım Lideri | data_harmony_report.md |

**Çıktılar:** `snomed_map.py`, `resample_250hz.py`, `quality_filter.py`, `class_distribution.csv`

### HAFTA 2: Curriculum Learning Başlangıcı

**Hedef:** Aşamalı eğitim pipeline'ı kur. Macro F1 hedefi: 0.78

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | Aşama 1: SADECE TEKNOFEST (5.000) ile baseline eğit | Model Geliştirici 2 | baseline_teknofest.py |
| Sal | Domain embedding: 5 veri seti için learnable embedding (5, 16) | Model Geliştirici 1 | domain_embedding.py |
| Çar | DANN: Domain adversarial training implementasyonu | Model Geliştirici 1 | dann_module.py |
| Per | Aşama 2: TEKNOFEST + Internet (85.000) ile eğitim | Model Geliştirici 2 | train_curriculum.py |
| Cum | Class-balanced sampler: Azınlık sınıfları (AFL) zorla dahil et | Veri Mühendisi | balanced_sampler_v2.py |
| Cmt | Quiz 2: Curriculum learning + DANN | Herkes | Kahoot |
| Paz | F1 karşılaştırması: Sadece TEKNOFEST vs Çoklu veri | XAI | f1_comparison.md |

**Çıktılar:** `baseline_teknofest.py`, `domain_embedding.py`, `dann_module.py`, `train_curriculum.py`, F1: 0.78+

### HAFTA 3: F1 Maksimizasyon + Hard Example Mining

**Hedef:** 0.84+ Macro F1. Azınlık sınıfı (AFL) kurtar.

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | Hard Example Mining: Validation'daki en yüksek loss'lu 200 örnek | Model Geliştirici 2 | hard_pool.py |
| Sal | AFL spesifik: SMOTE veya kopyalama ile AFL verisini 3x artır | Veri Mühendisi | afl_oversample.py |
| Çar | Focal Loss gamma tuning: gamma=1.5, 2.0, 2.5 dene | Model Geliştirici 2 | focal_tuning.py |
| Per | Cross-dataset evaluation: Her veri setini ayrı test et | Takım Lideri | cross_dataset_eval.py |
| Cum | Aşama 3: SADECE TEKNOFEST ile fine-tune (son 20 epoch) | Model Geliştirici 2 | finetune_teknofest.py |
| Cmt | Quiz 3: Hard example mining + Oversampling | Herkes | Kahoot |
| Paz | Hata defteri: Hangi veri setinden gelen örnekler daha çok hata? | XAI | 03_error_log.md |

**Çıktılar:** `hard_pool.py`, `afl_oversample.py`, `cross_dataset_eval.py`, F1: 0.84+

### HAFTA 4: Cross-Lead Attention + SQI (Multi-Dataset)

**Hedef:** 0.87+ Macro F1. Farklı cihazlardan gelen lead kalitesini yönet.

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | SQI modülü v2: Her veri seti için ayrı SQI threshold (cihaz farkı) | Sinyal Mühendisi | sqi_multidataset.py |
| Sal | Cross-Lead Attention: V1↔V6, I↔aVL ilişkisi | Model Geliştirici 1 | cross_lead_attn.py |
| Çar | SQI-Gated: Düşük kaliteli lead'i baskıla (ama atma!) | Model Geliştirici 1 | sqi_gating.py |
| Per | Branch A + B + Cross-Lead entegrasyonu | Model Geliştirici 1 | branch_fusion_v2.py |
| Cum | Eğitim. Confusion matrix: LBBB↔RBBB, AFIB↔AFL analizi | Herkes | Conf matrix |
| Cmt | Quiz 4: Cihaz farkı ve SQI | Herkes | Kahoot |
| Paz | Hata defteri: Hangi veri setinde LBBB↔RBBB daha çok? | XAI | 03_error_log.md |

**Çıktılar:** `sqi_multidataset.py`, `cross_lead_attn.py`, `sqi_gating.py`, F1: 0.87+

### HAFTA 5: Ensemble + Domain Robustluk

**Hedef:** 0.89+ Macro F1. Overfit kontrolü.

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | 3 Seed Ensemble (42, 123, 2026) | Model Geliştirici 2 | ensemble.py |
| Sal | Model Soup: 3 checkpoint ortalaması | Model Geliştirici 2 | model_soup.py |
| Çar | Mixup: alpha=0.2 ile interpolasyon | Model Geliştirici 2 | mixup.py |
| Per | Leave-one-dataset-out: Her veri setini ayrı test et | Takım Lideri | leave_one_out.py |
| Cum | Domain generalization raporu: Her veri seti için F1 | XAI | domain_gen_report.md |
| Cmt | Quiz 5: Ensemble + Domain genelleme | Herkes | Kahoot |
| Paz | F1 kazanç raporu #2 | Herkes | 05_f1_gains.md |

**Çıktılar:** `ensemble.py`, `model_soup.py`, `leave_one_out.py`, F1: 0.89+

### HAFTA 6: Hata Avı + Fine-Tuning

**Hedef:** 0.91+ Macro F1.

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | Confusion matrix derin analizi (veri seti bazında) | XAI | error_analysis_v2.py |
| Sal | AFIB↔AFL fix: P-wave regularity feature ekle | Sinyal Mühendisi | fix_afib_afl.py |
| Çar | LBBB↔RBBB fix: V1 SQI threshold'unu veri setine göre ayarla | Model Geliştirici 1 | fix_lbbb_rbbb.py |
| Per | Hyperparameter sweep | Model Geliştirici 2 | Sweep raporu |
| Cum | Final ensemble seçimi | Takım Lideri | Final ensemble |
| Cmt | Quiz 6: Hata analizi | Herkes | Kahoot |
| Paz | Full pipeline test | Herkes | Pipeline test |

**Çıktılar:** `error_analysis_v2.py`, `fix_afib_afl.py`, `fix_lbbb_rbbb.py`, F1: 0.91+

### HAFTA 7-8: Submission + Rapor

Aynı: Docker, reproducibility, teknik rapor, sunum.

## KRİTİK FARKLAR (Tek Veri Seti vs Çoklu Veri Seti)

| Konu | Tek Veri Seti (5.000) | Çoklu Veri Seti (85.000) |
|---|---|---|
| Hz | 500 Hz (sabit) | 250 Hz ve 500 Hz (uyum gerekli) |
| Sınıf dağılımı | Dengeli (1000/sınıf) | Dengesiz (AFL az) |
| Domain | Tek | Çok (5 farklı hastane/ülke) |
| Etiket | Tek SNOMED seti | Harmonizasyon gerekli |
| Eğitim | Tek aşama | Curriculum learning (3 aşama) |
| Overfit riski | Düşük | Yüksek (domain overfit) |
| DANN | Gerekmez | Gerekli |
| AFL sorunu | Yok (dengeli) | Var (azınlık) — oversample gerekli |

## 🎯 ALT SATIR

Çoklu veri seti = daha fazla veri ama daha fazla challenge.

**Kazanma formülü:** *"Herkes TEKNOFEST'in 5.000 kaydını kullanırken, biz 85.000 kayıtlık çoklu hastane verisini fizyolojik olarak harmanladık. Modelimiz sadece TEKNOFEST verisini değil, dünyanın farklı köşelerinden gelen EKG'leri de gördü."*

**Risk:** Domain overfit. **Çözüm:** DANN + Curriculum Learning + Leave-one-out.

**Hedef:** Macro F1 = 0.92-0.95 (çoklu veri avantajıyla)

---
---

# 🏆 TEKNOFEST 2026 2. Aşama — Takım Görev Planı

**Hedef: Macro F1'de 97+ Puanı Geçmek**

## 1. Yarışma Nedir? (30 saniyede anlat)

**TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — Lise Kategorisi 2. Aşama**

- **Girdi:** 12 derivasyonlu (lead) EKG kaydı, 500 Hz, 10 saniye
- **Çıktı:** 9 sınıf (Normal + AFIB, AFL, SVT, APB, VPB, LBBB, RBBB, AV Blok)
- **Metrik:** Tek şey önemli — Macro F1-Score
  - Her sınıfın F1 skorunu alıp ortalamasını alıyor.
  - "Normal" sınıfı %95 doğru bilmek işe yaramaz. Az örnekli sınıf (örneğin VPB) %40 kalırsa, Macro F1 çöker.
- **Test:** Yarışma finalinde bizim görmediğimiz, farklı bir hastaneden gelen EKG'lerle test edilecek. Ezberleyen model patlar.
- **Rakip:** PulseNet takımı 1. aşamada 97 aldı. Biz onları geçmeliyiz.

**Altın Kural:** Bu yarışmada kazanan en "derin" model değil, azınlık sınıfları (APB, VPB, AV Blok) en iyi ayıran takımdır.

## 2. EKG Nedir? (Hiç Bilmeyen İçin)

Kalbin elektrik aktivitesini kaydeden bir grafik.

- **P dalgası:** Atriyum (kulakçık) kasılması
- **QRS kompleksi:** Ventrikül (karıncık) kasılması (en büyük sivri şey)
- **T dalgası:** Ventrikül gevşemesi
- **12 Lead:** Vücudun 12 farklı yerinden aynı anda kayıt
  - **Frontal:** I, II, III, aVR, aVL, aVF (kol/bacaklardan)
  - **Prekordiyal:** V1, V2, V3, V4, V5, V6 (göğüs üzerinden)
  - Her lead kalbin farklı açıdan görür. Örneğin V1 sağ ventrikülü, V5 sol ventrikülü gösterir.

**Sınıfların Kısa Tarifi:**

| Sınıf | Türkçe | Nasıl Anlaşılır? |
|---|---|---|
| AFIB | Atriyal Fibrilasyon | Kalp düzensiz çarpar, P dalgası yok, kaotik |
| AFL | Atriyal Flutter | Düzenli ama çok hızlı atriyal aktivite (testere dişi) |
| SVT | Supraventriküler Taşikardi | Aniden hızlanan dar QRS ritim |
| APB | Atriyal Prematüre Beat | Erken gelen bir P dalgası, sonrası kısa duraklama |
| VPB | Ventriküler Prematüre Beat | Erken gelen, geniş ve çirkin QRS, sonrası uzun duraklama |
| LBBB | Sol Dal Bloğu | QRS geniş, V1'de küçük rS, V6'da geniş R |
| RBBB | Sağ Dal Bloğu | QRS geniş, V1'de rsR' (tavşan kulağı), V6'da geniş S |
| AV Blok | AV Blok | P dalgası ile QRS arasındaki mesafe (PR) uzar veya QRS düşer |
| Normal | Normal | Her şey düzenli ve kurallı |

## 3. Bizim Modelimiz: "CardioFusion-Net" (3 Branch)

Herkes "tek büyük CNN koyalım" diyor. Biz 3 farklı bakış açısını birleştiriyoruz. Hepsi aynı 12 lead'den besleniyor.

### Branch 1 — "GÖRÜŞ" (1D-CNN)

**Görev:** 12 lead'in ham sinyalinden şekilleri öğrenmek.
- V1'deki rsR' şekli (RBBB)
- V6'daki monofazik R (LBBB)
- Geniş QRS (VPB)
- **Input:** (5000, 12) → 10 saniye × 12 lead

### Branch 2 — "SAYI" (Fizyolojik Feature MLP)

**Görev:** NeuroKit2 ile otomatik çıkarılan sayıları kullanmak.
- Ortalama kalp hızı, RR aralığı varyansı
- QRS genişliği, PR aralığı
- "P dalgası var mı?" (0 veya 1)
- V1/V6 QRS oranı
- **Input:** (10,) sayısal vektör
- **Neden:** Az örnekli sınıflarda (VPB, AVB) modelin "kendisi öğrenmesini" beklemek verimsiz. Direkt veriyoruz.

### Branch 3 — "ZAMAN" (Beat-Level BiGRU + Attention)

**Görev:** Nabızların zaman dizisini öğrenmek.
- Her 10 saniyelik kaydı ~10 nabıza bölüyoruz.
- AFIB: Nabızlar düzensiz aralıklarla geliyor.
- APB: 3. nabız erken geldi, sonrası duraklama var.
- **Input:** (n_beats, 400, 12) → nabız × zaman × lead

### Gating (Kontrol Merkezi)

Model her kayıt için kendisi karar veriyor:
- "Bu kayıt ritim bozukluğu gibi görünüyor" → Branch 2+3'ü ağırlaştır.
- "Bu kayıtta morfoloji şüpheli" → Branch 1'i ağırlaştır.

**Neden bu F1 artırır?** Çünkü azınlık sınıfın gradient'i kendi expert'ine gider, büyük CNN'in gürültüsüne kaybolmaz.

## 4. Takım Rollerimiz (5 Kişi Varsayımı)

| Rol | Görev | Haftalık Zaman | Gereksinim |
|---|---|---|---|
| Takım Lideri (Sen) | Mimari tasarım, Gating, Eğitim loop, Cross-dataset strateji, Kod review | 20+ saat | Her şeyi koordine eder, en zor kodları yazar |
| Sinyal Mühendisi | Preprocessing (filtre, z-score), NeuroKit2 entegrasyonu, Fizyolojik feature çıkarımı (Branch 2) | 10-15 saat | Python bilse yeter, EKG'yi burada öğrenir |
| Model Geliştirici 1 | Branch 1 (1D-CNN + SE blocks), Branch 3 (Beat extraction + BiGRU) | 10-15 saat | PyTorch bilgisi |
| Model Geliştirici 2 | Training loop, Focal Loss, Class-balanced sampler, Ensemble, TTA | 10-15 saat | PyTorch + optimizasyon bilgisi |
| Veri Mühendisi | 5 veri setini indirme, SNOMED/ICD eşleme, Per-patient split, Augmentation | 10-15 saat | Pandas, dosya yönetimi |
| XAI & Dokümantasyon | Confusion matrix analizi, Hata raporları, Teknik rapor, Sunum slaytları | 8-12 saat | Yazı + matplotlib, kod bilgisi az da olur |

**Not:** Eğer 4 kişiyseniz, XAI rolü Sinyal Mühendisi'ne veya Takım Lideri'ne devredilir.

## 5. 8 Haftalık Detaylı Plan

### 🔴 HAFTA 1: Fizyoloji + Veri Altyapısı

**Hedef:** Herkes kendi sınıfının morfolojisini öğrensin. Veri setleri hazır olsun.

**Pazartesi:**
- Takım toplantısı (1 saat): Bu dokümanı konuş.
- Herkese 2 sınıf atanır (örnek: Ali=AFIB+AFL, Ayşe=LBBB+RBBB, Mehmet=APB+VPB, Zeynep=SVT+AVB, Sen=Normal).

**Salı–Perşembe:**
- Herkes ECGPedia'dan kendi 2 sınıfını okur.
- Herkes PhysioNet'ten kendi sınıflarından 5'er örnek indirip Lead II'yi çizer.
- Veri Mühendisi: Tüm veri setlerini (PhysioNet, PTB-XL, CPSC, Chapman, Georgia) indirir.

**Cuma:**
- Veri Mühendisi: SNOMED kod eşleme tablosunu + per-patient split CSV'lerini teslim eder.
- Sinyal Mühendisi: `preprocess_12lead()` fonksiyonunu teslim eder (filtre + z-score).

**Cumartesi:**
- **Quiz 1:** 12 lead anatomisi + sınıf tanımları (10 soru, Kahoot/Quizizz).
- En yüksek alan takıma kahve ısmarlar (motivasyon).

**Pazar:**
- Haftalık review: Herkes kendi sınıfının 3 EKG'sini takıma anlatır (5 dk).
- Fizyoloji defteri kontrolü.

**Çıktılar:** `dataset_registry.json`, `preprocess_12lead.py`, Herkesin fizyoloji notu (2 sayfa)

### 🔴 HAFTA 2: Baseline + SQI + Balanced Sampling

**Hedef:** Çalışan bir model olsun. Macro F1 hedefi: 0.75

**Pazartesi:**
- Sinyal Mühendisi: NeuroKit2 ile 12 lead'de R-peak detection (`rpeak_12lead.py`).

**Salı:**
- Model Geliştirici 1: Baseline 1D-CNN kurar. Input (batch, 5000, 12). 3 Conv block + GAP + Dense.
- Model Geliştirici 2: Training loop + WandB loglama kurar.

**Çarşamba:**
- Model Geliştirici 2: Class-balanced batch sampler yazar. Her batch'te azınlık sınıfları (VPB, APB) zorla dahil eder.
- Sinyal Mühendisi: SQI modülü v1 (12 lead için kurtosis + baseline wander).

**Perşembe:**
- Model Geliştirici 2: Focal Loss implementasyonu (gamma=2.0).
- İlk eğitim denemesi (20 epoch).

**Cuma:**
- Validation sonuçları. Confusion matrix çizilir. Hangi sınıf düşük?

**Cumartesi:**
- **Quiz 2:** Sınıf morfolojisi + preprocessing kuralları.
- **Hands-on:** Herkes kendi sınıfından 1 kayıtta R-peaks işaretler.

**Pazar:**
- Hata defteri #1: "VPB F1 neden 0.35? Batch'te VPB yoktu."

**Çıktılar:** `baseline_cnn.py`, `balanced_sampler.py`, `train.py` (v1), İlk F1 raporu

### 🔴 HAFTA 3: F1 Maksimizasyon I (Data-Centric)

**Hedef:** Azınlık sınıflarını kurtar. Macro F1 hedefi: 0.82

**Pazartesi–Salı:**
- Model Geliştirici 2: Hard Example Mining. Validation'daki en yüksek loss'lu 100 örneği bul, "hard pool" oluştur. Batch'lerin %30'u buradan.

**Çarşamba–Perşembe:**
- Sinyal Mühendisi: 12 lead augmentation. Lead-wise amplitude scale (0.9–1.1) + Gaussian noise. **Zaman kaydırma YOK** (P-QRS-T bozulur).
- Model Geliştirici 2: Class weight tuning. 1/sqrt(freq) ve Focal Loss gamma=1.5/2.0/2.5 dene.

**Cuma:**
- Model Geliştirici 1: TTA (Test-Time Augmentation). Inference'da amplitude ×0.95, ×1.0, ×1.05 ortalaması.

**Cumartesi:**
- **Quiz 3:** Augmentation kuralları + Focal Loss mantığı.
- **F1 Kazanç Raporu #1:** Herkes "bu hafta ne denedim, F1 etkisi ne oldu" sunar.

**Pazar:**
- Leaderboard #1: Haftanın en büyük F1 artışını yapan kazanır.

**Çıktılar:** `hard_pool.py`, `augment_12lead.py`, `inference_tta.py`, F1: 0.82+

### 🔴 HAFTA 4: Hibrit Mimari (3 Branch)

**Hedef:** 3 branch'li model çalışsın. Macro F1 hedefi: 0.86

**Pazartesi:**
- Takım Lideri: Gating network + fusion katmanını yazar.
- Sinyal Mühendisi: Branch 2 (fizyolojik feature) tamamlar. `extract_physio_features()` teslim eder.

**Salı–Çarşamba:**
- Model Geliştirici 1: Branch 1 (1D-CNN) + SE blocks ekler.
- Model Geliştirici 1: Branch 3 (Beat extraction) başlar. R-peak etrafında ±200 sample (800ms), 12 lead.

**Perşembe:**
- Model Geliştirici 2: Multi-task loss. Ana loss (9 sınıf Focal) + auxiliary loss (Ritim/İletim/Normal 3 sınıf).

**Cuma:**
- İlk 3-branch eğitim. Debug et.

**Cumartesi:**
- **Quiz 4:** Branch mantığı + Gating nedir?
- **Hands-on:** Beat segmentasyonu görselleştirme. 12 lead × 10 beat plot.

**Pazar:**
- Hata defteri #2: "Branch 3 beat detection hatası veriyor mu?"

**Çıktılar:** `cardiofusion_net.py` (v1), `branch_cnn.py`, `branch_sequence.py`, `features.py`

### 🔴 HAFTA 5: Beat-Level + Sınıf Spesifik Optimizasyon

**Hedef:** APB/VPB gibi "erken beat" sınıfları yakalamak. Macro F1 hedefi: 0.88

**Pazartesi–Salı:**
- Model Geliştirici 1: Beat-Level Transformer. Self-attention over beats (not raw samples).
- Input: (n_beats, 400, 12) → CNN per beat → (n_beats, 128) → Transformer → (128,)

**Çarşamba:**
- Sinyal Mühendisi: APB/VPB spesifik feature'ler. Prev-RR, post-RR, QRS width explicit olarak modele sokulur.

**Perşembe:**
- Model Geliştirici 2: LBBB/RBBB spesifik cross-lead attention. V1, V6, I, aVL lead'lerine başlangıçta daha yüksek ağırlık.

**Cuma:**
- Model Geliştirici 2: AFIB/AFL spesifik. RR variance + P-wave absence detection.

**Cumartesi:**
- **Quiz 5:** Beat-level vs sample-level attention farkı.
- **F1 Kazanç Raporu #2**

**Pazar:**
- Sınıf başına F1 analizi. En düşük F1'li sınıf hangisi? Aksiyon planı.

**Çıktılar:** `beat_transformer.py`, Sınıf spesifik F1 tablosu

### 🔴 HAFTA 6: Ensemble + Domain Robustness

**Hedef:** Overfit'i kır, external validation'a hazırlan. Macro F1 hedefi: 0.90

**Pazartesi–Salı:**
- Model Geliştirici 2: Cross-Dataset Protocol. Leave-one-dataset-out:
  - Exp1: Train PTB+CPSC → Test Georgia
  - Exp2: Train PTB+Georgia → Test CPSC
  - 5 farklı kombinasyon.

**Çarşamba:**
- Model Geliştirici 1: Instance Normalization (early layers) + Layer Normalization (late layers).

**Perşembe:**
- Model Geliştirici 2: Model Ensemble. 3 farklı seed ile aynı mimariyi eğit, inference'da logit ortalaması.

**Cuma:**
- Model Geliştirici 2: Model Soup. 3 checkpoint'in ağırlıklarını ortalama al (test seti görmeden!).

**Cumartesi:**
- **Quiz 6:** Cross-dataset neden önemli? External validation nedir?
- **Hands-on:** 1 modeli farklı seed'lerle eğitip sonuç karşılaştırma.

**Pazar:**
- Cross-dataset rapor: Macro F1 mean / std. Std > 0.05 ise overfit var demektir.

**Çıktılar:** `cross_dataset_eval.py`, `ensemble_3seed.py`, `model_soup.py`

### 🔴 HAFTA 7: Hata Avı + Fine-Tuning

**Hedef:** Son %2'lik kazancı al. Macro F1 hedefi: 0.91+

**Pazartesi–Salı:**
- XAI & Dokümantasyon: Confusion matrix derin analizi. En sık karışan çiftler:
  - AFIB↔AFL?
  - LBBB↔RBBB?
  - VPB↔Normal?

**Çarşamba:**
- Takım Lideri + Model Geliştiriciler: Karışan çiftler için fizyolojik düzeltme.
- Örn: AFIB↔AFL karışıyorsa → Rhythm branch'e P-wave autocorrelation ekle.

**Perşembe:**
- Model Geliştirici 2: Hyperparameter sweep (LR, dropout, attention heads). WandB ile.

**Cuma:**
- Final ensemble seçimi. En iyi 3 checkpoint (cross-dataset F1'e göre).

**Cumartesi:**
- **Quiz 7:** Hata analizi fizyolojisi. "Neden model AFIB'yi AFL sanıyor?"
- **F1 Kazanç Raporu #3 (Final)**

**Pazar:**
- Full pipeline test. Görülmemiş hasta simülasyonu.

**Çıktılar:** `error_analysis.py`, `final_ensemble.py`, Final F1 raporu

### 🔴 HAFTA 8: Submission + Teknik Rapor + Sunum

**Hedef:** Teslim edilebilir, tekrar üretilebilir paket.

**Pazartesi–Salı:**
- Takım Lideri: Docker container. `docker build` ile çalışan inference.
- Model Geliştirici 2: Reproducibility. Seed fix (42, 123, 2026). `config.yaml` + `requirements.txt`.

**Çarşamba–Perşembe:**
- XAI & Dokümantasyon: Teknik rapor yazımı.
- Bölümler: Problem, 12 Lead Fizyolojisi, Mimari, Preprocessing, F1 Sonuçları, Hata Analizi, Limitler.

**Cuma:**
- XAI & Dokümantasyon: Sunum slaytları (max 10 slide).
  1. Problem + Macro F1 nedir?
  2. 12 Lead anatomisi (görsel)
  3. Sınıf morfolojileri (örnek EKG'ler)
  4. Mimari diyagram (3 Branch)
  5. Preprocessing pipeline
  6. Macro F1 sonuçları (tablo)
  7. Cross-dataset robustluk
  8. Hata analizi (confusion matrix)
  9. Açıklanabilirlik (attention görseli)
  10. Limitler ve gelecek

**Cumartesi:**
- Final submission format kontrolü. Dosya yapısı, model ağırlıkları, inference script.
- Takım dışından birinin (örn. danışman) test etmesi.

**Pazar:**
- Son review. Teslim.

**Çıktılar:** `Dockerfile`, `technical_report.pdf`, `presentation.pptx`, Submission paketi ✅

## 6. Eğitim Sistemi: "F1 Akademisi"

Hiçbir şey bilmeyen takım arkadaşlarının domain bilgisi F1'i dolaylı olarak artırır. Çünkü hata analizi yapıp modeli düzeltirler.

**Haftalık Ritüel (Cumartesi Sabah 10:00)**

| Süre | Aktivite | Amaç |
|---|---|---|
| 10 dk | Quiz (Kahoot/Quizizz) | O haftanın 2 sınıfı + 12 lead anatomisi |
| 20 dk | Hands-on Lab | Herkes kendi sınıfından 1 EKG kaydı çizip R-peaks işaretler |
| 30 dk | F1 Kazanç Raporu | Herkes "bu hafta ne denedim, F1 etkisi ne oldu" sunar |

**Quiz Örneği (Hafta 2)**
1. V1 lead'inde RBBB için hangi morfoloji görülür? (rsR' / tavşan kulağı)
2. 12 lead'de z-score normalizasyonu neden lead-wise yapılır? (Her lead farklı genlikte)
3. VPB F1'si düşükse hangi sampling yöntemi kullanılır? (Class-balanced / hard example mining)
4. AFIB'de RR interval nasıldır? (Düzensiz / irregularly irregular)
5. Focal Loss gamma=2.0 ne yapar? (Azınlık sınıfına daha fazla odaklanır)

**F1 Kazanç Defteri (Herkes Yazar)**

Her hafta sonu 1 sayfa:

```
## [İsim] — Hafta X F1 Kazanç Raporu

Bu hafta denediğim şey: Class-balanced sampler ekledim.
Neden denedim: VPB F1 0.35'ti, batch'te hiç VPB yoktu.
Sonuç: VPB F1 0.35 → 0.52. Macro F1 0.78 → 0.81.
Kanıt: [WandB screenshot]
Sonraki hipotez: QRS width explicit feature eklemeliyim.
```

**En büyük F1 artışını yapan haftanın kazananı olur.** (Motivasyon)

## 7. Kurallar ve Ritüeller

1. **Per-Patient Split:** Aynı hastanın kaydı hem train hem test'te olamaz. Yarışma bunu kontrol eder.
2. **Lead-Wise Normalizasyon:** 12 lead'in her biri kendi z-score'u ile normalize edilir. Global yapma.
3. **Zaman Kaydırma Yok:** Augmentation'da sinyali kaydırma (time-shift) yasak. P-QRS-T ilişkisi bozulur.
4. **Macro F1 Tek Tanrı:** Her karar "bu Macro F1'i artırır mı?" sorusuna göre verilir.
5. **Görülmemiş Veri Simülasyonu:** Hafta 6'dan itibaren her hafta en az 1 kez leave-one-dataset-out test yapılır.
6. **Kod Review:** Çarşamba akşamları 1 saat. Herkes bir başkasının kodunu okur.
7. **Hata Defteri:** Her hafta sonu confusion matrix'teki en sık 3 hata çifti yazılır ve fizyolojik olarak yorumlanır.

## 8. Hemen Bugün/Yarın Yapılacaklar

1. Takım toplantısı (1 saat): Bu dokümanı ekranda göster, herkes okusun.
2. Rol ataması: Herkese bir rol ve 2 sınıf ata.
3. GitHub repo açın: `teknofest-2026-cardiofusion`
4. `/docs` klasörü açın: 5 boş dosya oluşturun:
   - `01_morphology_atlas.md`
   - `02_adr.md` (teknik karar defteri)
   - `03_error_log.md`
   - `04_dataset_registry.md`
   - `05_f1_gains.md`
5. Quiz 1 hazırla: Kahoot'ta 10 soru yaz (12 lead anatomisi + sınıf tanımları).
6. Veri seti indirme başlasın: PhysioNet 12-Lead ECG Arrhythmia Dataset.
7. Herkes ECGPedia'ya girsin: ecgpedia.org → Basics bölümünü okusun.

## 9. Özet: Neden Bu Plan PulseNet'i Geçer?

| PulseNet (97 puan) | Bizim Planımız | F1 Etkisi |
|---|---|---|
| Tek branch 1D-CNN | 3 Branch + Gating | +2-3% |
| Sadece raw signal | Raw + Fizyolojik Sayılar + Beat Sequence | +1-2% |
| Global average pool | Beat-level Attention | +1-2% (APB/VPB) |
| Tek veri seti odaklı | Cross-Dataset + Ensemble + TTA | +1-2% |
| Fixed fusion | Learned Gating | +0.5-1% |
| | **Toplam** | **+5-8% F1** |

**Altın Cümle:**
*"Herkes daha derin CNN koyarken, biz EKG'nin fizyolojisini modelin içine gömdük. 12 lead'in hangisine güveneceğini, hangi nabızın anormal olduğunu, ne zaman emin olmadığını kendisi öğrendi."*

**Hazır mısınız?**

---
---

# TEKNOFEST 2026 — 2. Aşama Teknik Rapor

## CardioFusion-5: 5-Sınıf EKG Sınıflandırma — Çoklu Veri Seti Stratejisi

**1. Aşama Sonucu:** Macro F1 = 96.7 (3 Sınıf: Sağlıklı / Ritim / İletim)
**2. Aşama Hedefi:** Macro F1 ≥ 97.0 (5 Sınıf: Normal / AFIB / AFL / LBBB / RBBB)

**Ekip:** 3 Kişi + Mentor
**Zaman Çizelgesi:** 8 Hafta
**Rapor Tarihi:** Mayıs 2026

## 1. Giriş ve Bağlam

### 1.1 1. Aşama Deneyimi ve Çıkarımlar

1. Aşamada 3 sınıf (Sağlıklı / Ritim Bozukluğu / İletim Bozukluğu) sınıflandırması yapılmış ve Macro F1 = 96.7 skoru elde edilmiştir. Bu skor, 97.0 alan birinci takıma çok yakındır. 1. Aşamadaki başarının temelinde şunlar yatmaktadır:
- Sağlam ön işleme hattı (lead-wise Z-Score, per-patient split)
- Fizyolojik bilginin modele yansıtılması
- Dengeli veri seti üzerinde stabil eğitim

Ancak 2. Aşama, 3 sınıf yerine 5 ayrı sınıf (Normal, AFIB, AFL, LBBB, RBBB) içermektedir. Bu, problemi önemli ölçüde karmaşıklaştırır çünkü:
- Ritim Bozukluğu artık AFIB ve AFL olarak ayrılmaktadır (fizyolojik olarak yakın, karışma riski yüksek)
- İletim Bozukluğu artık LBBB ve RBBB olarak ayrılmaktadır (morfolojik olarak yakın, V1/V6 bağımlılığı kritik)
- Sınıf sayısı arttıkça azınlık sınıf sorunu derinleşmektedir

### 1.2 2. Aşama Problem Tanımı

- **Girdi:** 12 derivasyonlu EKG kaydı, 10 saniye, 500Hz (TEKNOFEST) veya 250Hz (İnternet)
- **Çıktı:** 5 sınıf (0:Normal, 1:AFIB, 2:AFL, 3:LBBB, 4:RBBB)
- **Metrik:** Macro F1-Score (her sınıf eşit ağırlıklı)
- **Test:** Yarışma finalinde görülmemiş, farklı bir hastaneden EKG'ler
- **Veri:** TEKNOFEST (5.000, dengeli) + İnternet (~100.000, dengesiz)

### 1.3 Stratejik Farkındalık

0.3 puanlık marjı kapatmak için "daha derin katman" eklemek yeterli olmayacaktır. Kazanç, fizyolojik bilginin daha derin gömülmesi ve domain robustluğun artırılmasıyla gelecektir. 1. Aşamadaki 96.7 skoru, temel altyapının sağlam olduğunu göstermektedir; şimdi bu altyapı 5 sınıf ve çoklu veri seti için genişletilecektir.

## 2. Teknik Mimari

### 2.1 Mimari Felsefe: İki Versiyon

Ekip 3 kişi olduğundan, mimari karmaşıklığı ile debug süresi arasında denge kurulmalıdır. İki paralel versiyon geliştirilecek, Hafta 4'te stabil olana karar verilecektir.

### 2.2 Versiyon A: CardioFusion-Efficient (Pragmatik)

Tek ana dal (branch) üzerine kurulmuş, ancak fizyolojik bilgiyi doğrudan enjekte eden yapı. Debug süresi minimum, eğitim stabilitesi maksimumdur.

```
Girdi: (2500, 12) @ 250Hz
Ön İşleme: Bandpass 0.5-40Hz + Lead-wise Z-Score + SQI(12,)
Ana Dal: 1D-CNN + SE Blocks + Cross-Lead Attention → (128,)
Yardımcı Girdi: Precomputed fizyolojik özellikler (RR varyansı, QRS genişliği, P-dalga varlığı,
V1/V6 oranı) → (8,)
Birleştirme: (128 + 8) = (136,)
Sınıflandırıcı: Dense(128) → Dense(5) + Softmax
Yardımcı Başlık: 3-sınıf (Normal / Ritim / İletim) Multi-Task Loss
```

**Cross-Lead Attention:** V1↔V6 ve I↔aVL arasındaki fizyolojik ilişkiyi explicit olarak öğrenir. RBBB tanısı V1'de (rsR'), LBBB tanısı V1+V6'da yazılıdır.

**Precomputed Özellikler (1. Aşama Bilgisinin Aktarımı):** 1. Aşamada öğrenilen ritim/iletim ayrımı bilgisi, bu skaler özellikler aracılığıyla 2. Aşamaya taşınır. Beat segmentasyonu model içinde yapılmaz; eğitim hızı 10x artar.

### 2.3 Versiyon B: CardioFusion-Pro (Agresif)

Tam hibrit mimari. Morfoloji ve ritim bilgisini ayrı dallarda işler, SQI-Gated adaptif birleştirme ile karar verir.

```
Dal A - Morfoloji: 1D-CNN + SE + Cross-Lead Attention + Instance Norm
→ V1 rsR' (RBBB), V6 geniş R (LBBB), QRS genişliği öğrenir

Dal B - Ritim: Beat-Level BiGRU + Self-Attention + RR-Variance explicit
→ AFIB düzensiz RR, AFL düzenli testere, beat-to-beat değişim öğrenir
→ R-tepesi preprocessing'de tespit edilir, model içinde değil
```

**SQI-Gated Fusion:** 12 derivasyonun kalite skoruna göre ağırlıklandırma. V1 kalitesi düşükse V6'ya güven. Derivasyon asla atılmaz; sadece ağırlığı düşürülür.

**Domain Embedding:** 5 veri seti için öğrenilebilir vektör (16-dim). Cihaz/popülasyon farkını normalize eder.

**DANN (Domain-Adversarial):** Ters gradyan ile domain sınıflandırıcı. Özellik çıkarıcı veri seti bilgisini "unutmaya" zorlanır.

**Birleşik Başlık:** 5-sınıf (Focal Loss) + 3-sınıf (Aux Loss - 1. Aşama bilgisini korur)

## 3. Veri Stratejisi

### 3.1 Ön İşleme Pipeline (Ortak)

Tüm ham kayıtlar aşağıdaki sırayla işlenir. Seed=42 ile sabitlenmiştir.

- **Adım 1 - Okuma:** WFDB formatından (N, 12) sinyal çıkarımı
- **Adım 2 - Hz Uyumu:** 500Hz → 250Hz polifazik resampling (`scipy.signal.resample_poly`). Nyquist 125Hz, EKG'deki en yüksek anlamlı frekans ~40Hz. P-QRS-T morfolojisi korunur.
- **Adım 3 - Filtreleme:** Butterworth bandpass 0.5-40Hz, 4. derece, filtfilt. Baseline wander ve EMG gürültüsü kaldırılır. 50Hz notch kullanılmaz (T dalgasına zarar).
- **Adım 4 - Normalizasyon:** Lead-wise Z-Score. μ ve σ SADECE train setinden hesaplanır, val/test'e uygulanır. Global Z-Score ASLA yapılmaz.
- **Adım 5 - Kalite Taraması:** SQI hesaplama (lead-wise): kurtosis, baseline wander, QRS band gücü (5-15Hz), R-tepesi tespiti. SQI < 0.2 olan kayıtlar eğitimden çıkarılır.
- **Adım 6 - Etiket Harmoni:** SNOMED-CT eşleştirme. 5 hedef sınıfa net eşleşemeyenler atılır. Çift etiketli kayıtlarda dominant seçilir, belirsizse atılır.
- **Adım 7 - Çıktı:** (2500, 12) + label + SQI(12,)

### 3.2 SNOMED-CT Harmonizasyonu

Farklı veri setlerinde farklı kodlar bulunur. Harmonizasyon iki seviyededir:

**Seviye 1 (Üst Sınıf):** Ritim bozukluğu / İletim bozukluğu / Normal
**Seviye 2 (Alt Sınıf):** AFIB / AFL / LBBB / RBBB / Normal

Örnek eşleştirmeler:
- 164889003 → AFIB (Atrial Fibrillation)
- 164890007 → AFL (Atrial Flutter)
- 164909002 → LBBB (Left Bundle Branch Block)
- 59118001 → RBBB (Right Bundle Branch Block)
- 426783006 → Normal (Sinus Rhythm)

Ara kodlar (SVT, AT, APB, VPB, AVB çeşitleri) en yakın hedef sınıfa atanır. Hâlâ belirsizse kayıt atılır.

### 3.3 Eğitim Stratejisi: Curriculum Learning

Çoklu veri seti kullanımında domain overfit riski yüksektir. Üç aşamalı eğitim uygulanır:

- **Aşama 1 - Temel (Epoch 1-20):** Sadece TEKNOFEST (5.000 kayıt, dengeli). DANN kapalı. Model temel kardiyak öğrenmeyi edinsin. Hedef: Stabil başlangıç.
- **Aşama 2 - Genişleme (Epoch 21-80):** TEKNOFEST + İnternet (~100k, dengesiz). DANN açık (λ=0.1). Focal Loss (γ=2.0). Hard Example Mining. Class-Balanced Sampler. AFL ~%4 olduğundan 3x artırma (kopyalama + gürültü, SMOTE yok). Hedef: Farklı popülasyonları gör.
- **Aşama 3 - İnce Ayar (Epoch 81-100):** Sadece TEKNOFEST (5.000). DANN kapalı. LR 10x düşük. Yarışma domain'ine geri dön, overfit'i kır. Hedef: Yarışma verisine uyum.

### 3.4 Sınıf Dengesizliği Çözümleri

İnternet verisindeki tahmini dağılım:
- Normal: ~47% | AFIB: ~18% | AFL: ~4% | LBBB: ~14% | RBBB: ~17%

Çözümler:
- **Class-Balanced Sampler:** Her mini-batch'te tüm sınıflar eşit temsil edilir.
- **Focal Loss (γ=2.0):** Azınlık sınıflarının gradient'ini şişirir.
- **AFL Artırma:** SMOTE kullanılmaz (zaman serisini bozar). Kopyalama + amplitüd ölçekleme (0.9-1.1, lead-wise) + hafif Gauss gürültüsü (SNR>20dB) ile 3 kat artırma.
- **Hard Example Mining:** Validation'daki en yüksek loss'lu 200 örnek havuzlanır. Her epoch'un %30'u bu zor örneklerden oluşturulur.

## 4. Sekiz Haftalık Yol Haritası

**Ekip:** 2 Model Geliştirici + 1 Veri Mühendisi + Mentor

### Hafta 1: Veri Altyapısı

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Tüm veri setleri indirme (PhysioNet, PTB-XL, CPSC, Georgia, Chapman) | Veri Müh. | dataset_registry.json |
| Çar-Per | SNOMED kod eşleştirme + 5 sınıf harmoni | Veri Müh. | snomed_map.py |
| Cum | Hz uyumu (500→250) + bandpass filtreleme | Geliştirici 1 | preprocess.py |
| Cmt | Lead-wise Z-Score + SQI v1 | Geliştirici 1 | sqi_module.py |
| Paz | Veri kalite raporu + per-patient split kontrolü | Mentor + Ekip | quality_report.md |

**Hedef:** 5.000 TEKNOFEST kaydı + ~100k internet kaydı harmonize ve temiz.

### Hafta 2: Baseline ve Dengeli Örnekleme

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | Baseline 1D-CNN (Versiyon A) | Geliştirici 1 | baseline_cnn.py |
| Sal | Eğitim döngüsü + WandB loglama | Geliştirici 2 | train_loop.py |
| Çar | Class-Balanced Sampler | Geliştirici 2 | balanced_sampler.py |
| Per | Focal Loss (γ=2.0) + ilk eğitim | Geliştirici 2 | focal_loss.py |
| Cum | Validation + Confusion matrix | Ekip | f1_report_v1.md |
| Cmt | Quiz 1: 12 derivasyon anatomisi | Mentor | Kahoot |
| Paz | Hata defteri 1 | Ekip | 01_error_log.md |

**Hedef:** Macro F1 ≥ 0.85 (sadece TEKNOFEST). Eğer tutmazsa preprocessing'te hata vardır.

### Hafta 3: F1 Maksimizasyonu

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Hard Example Mining: Zor örnek havuzu | Geliştirici 2 | hard_pool.py |
| Çar | AFL artırma (kopyalama + gürültü) | Veri Müh. | afl_augment.py |
| Per | Cross-Lead Attention v1 (V1↔V6) | Geliştirici 1 | cross_lead_attn.py |
| Cum | TTA prototip | Geliştirici 2 | tta_inference.py |
| Cmt | Quiz 2: Focal Loss + augmentation | Mentor | Kahoot |
| Paz | F1 Kazanç Raporu 1 | Ekip | 02_f1_gains.md |

**Hedef:** Macro F1 ≥ 0.88. AFL F1'i 0.50+ seviyesine çekilmeli.

### Hafta 4: Mimari Geliştirme ve Versiyon Ayrımı

| Versiyon | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| A | Cross-Lead + Precomputed fizyolojik özellikler | Geliştirici 1 | cardiofusion_efficient.py |
| A | Multi-Task Loss (5+3 sınıf) | Geliştirici 2 | multi_task_loss.py |
| B | Dal B: Beat extraction + BiGRU | Geliştirici 1 | branch_rhythm.py |
| B | DANN modülü | Geliştirici 2 | dann_module.py |
| B | SQI-Gated Fusion | Geliştirici 1 | sqi_gating.py |
| Ortak | Domain Embedding (5 veri seti) | Veri Müh. | domain_embed.py |
| Ortak | Hafta sonu karar: A veya B | Mentor + Ekip | mimari_karar.md |

**Karar Kriteri:** Versiyon B debug > 3 gün veya beat hatası > %5 ise A'ya dönülür.

### Hafta 5: Sınıf-Spesifik Optimizasyon

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | AFIB/AFL ayrımı: P-dalga otokorelasyonu + RR varyansı | Geliştirici 1 | fix_afib_afl.py |
| Çar | LBBB/RBBB ayrımı: V1 SQI threshold + V1/V6 cross-attention | Geliştirici 1 | fix_lbbb_rbbb.py |
| Per | Hyperparameter sweep (LR, dropout) | Geliştirici 2 | sweep_report.md |
| Cum | Sınıf başına F1 analizi | Ekip | per_class_f1.csv |
| Cmt | Quiz 3: Hata analizi fizyolojisi | Mentor | Kahoot |
| Paz | F1 Kazanç Raporu 2 | Ekip | 03_f1_gains.md |

**Hedef:** Macro F1 ≥ 0.91. AFIB↔AFL ve LBBB↔RBBB karışımı %50 azaltılmalı.

### Hafta 6: Ensemble ve Domain Doğrulama

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | 3-Seed Ensemble (42, 123, 2026) | Geliştirici 2 | ensemble_3seed.py |
| Çar | Model Soup: 3 checkpoint ortalaması | Geliştirici 2 | model_soup.py |
| Per | Leave-One-Dataset-Out protokolü | Veri Müh. | leave_one_out.py |
| Cum | Cross-dataset rapor (5 kombinasyon F1 mean/std) | Ekip | cross_dataset_report.md |
| Cmt | Quiz 4: Domain genelleme | Mentor | Kahoot |
| Paz | Karar: Overfit var mı? (Std > 0.05 ise alarm) | Mentor + Ekip | robustluk_karari.md |

**Kabul Kriteri:** 5 kombinasyonun Macro F1 std'si < 0.05 olmalı.

### Hafta 7: Hata Avı ve İnce Ayar

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Confusion matrix derin analizi (veri seti bazında) | Ekip | error_analysis_v2.py |
| Çar | Final ensemble seçimi (cross-dataset F1 en yüksek 3) | Geliştirici 2 | final_ensemble.py |
| Per | Inference pipeline (tek komut) | Geliştirici 1 | inference_final.py |
| Cum | Full pipeline test: Görülmemiş hasta simülasyonu | Ekip | pipeline_test.md |
| Cmt | Quiz 5: Final sistem testi | Mentor | Kahoot |
| Paz | Hata defteri 2: Son aksiyonlar | Ekip | 04_final_actions.md |

**Hedef:** Macro F1 ≥ 0.93 (A) veya ≥ 0.95 (B). Son %2'lik kazanç.

### Hafta 8: Teslimat

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Docker container oluşturma ve test | Geliştirici 1 | Dockerfile |
| Çar-Per | Teknik rapor yazımı | Mentor + Ekip | technical_report.pdf |
| Cum | Sunum slaytları (max 10 slide) | Mentor | presentation.pptx |
| Cmt | Final submission format kontrolü | Ekip | submission_package/ |
| Paz | Son review ve teslimat | Mentor + Ekip | TEKNOFEST Submission |

**Önemli:** Seed fix (42), config.yaml, requirements.txt kesin versiyonlarla teslim.

## 5. Riskler ve Önlemler

| Risk | Etki | Önlem | Öncelik |
|---|---|---|---|
| Domain Overfit | İnternet verisine aşırı uyum, yarışmada düşük F1 | DANN + Curriculum + Leave-one-out | Yüksek |
| Etiket Hatası | Yanlış fizyoloji öğrenimi | SNOMED harmoni + kalite filtre + belirsiz atma | Yüksek |
| Sınıf Dengesizliği | AFL F1 çöker, Macro F1 düşer | Balanced sampler + Focal Loss + AFL artırma | Yüksek |
| Mimari Karmaşıklığı | Debug süresi uzar | Versiyon A yedek patikası | Orta |
| V1 Kalitesi | RBBB/LBBB karışımı artar | SQI-Gating veya V1 threshold | Orta |
| Overfit (Teknofest) | Aşama 3 ihmal edilirse domain bozulur | Aşama 3 zorunlu fine-tune | Yüksek |

**Domain Overfit Detayı:** Çoklu veri setinin en büyük riski, modelin "hangi hastaneden geldiğini" öğrenip kardiyak bilgiyi ikincil plana atmasıdır. DANN modülü özellik çıkarıcıyı veri seti bilgisini "unutmaya" zorlar. Leave-one-out testinde std > 0.05 çıkarsa, DANN lambda artırılır (0.1 → 0.3) veya Aşama 2 epoch'ları azaltılır.

## 6. Sakınılması Gereken Kritik Hususlar

Aşağıdaki hatalar, Macro F1 skorunu %5-%10 düşürebilen "ölümcül" hatalardır. Bu kuralların ihlali, projenin başarısız olması için yeterlidir.

1. **Global Z-Score Normalizasyonu:** ASLA tüm 12 derivasyonu birlikte normalize etmeyin. V1 QRS genliği 0.5mV, V5 QRS genliği 2.5mV olabilir. Global ortalama alındığında V1 sinyali "yok olur", V5 baskın hale gelir. RBBB tanısı V1'e bağlıdır; V1'i yok ederseniz model RBBB öğrenemez. LBBB tanısı V6'ya bağlıdır; V6'yı baskılarsanız LBBB öğrenemez. Her derivasyon kendi içinde normalize edilmelidir.

2. **Rastgele Hasta Bölmesi:** ASLA kayıt bazında rastgele bölme yapmayın. Aynı hastanın 5 farklı EKG kaydı varsa, 4'ü eğitim 1'i teste gidebilir. Model o hastayı ezberlemiş olur. Yarışma finalinde yeni hasta görüldüğünde model patlar. Bir hasta ya eğitimdedir, ya doğrulamada, ya testtedir. Asla ikisinde birden olmamalıdır.

3. **Zaman Kaydırma (Time-Shift) Artırma:** ASLA sinyali döngüsel olarak kaydırmayın (`np.roll`). P-QRS-T temporal ilişkisini bozar. P dalgası QRS'ten önce gelmelidir; kaydırırsanız bu fizyolojik sıra bozulur. Model ritim bozukluklarını öğrenemez.

4. **Rastgele Kırpma (Random Crop):** ASLA sinyalin başını veya sonunu rastgele kırpmayın. Baştaki P dalgası veya sondaki T dalgası kaybolabilir. Ritim analizi için bu dalgalar kritiktir.

5. **Genel Amplitüd Ölçeklendirme:** ASLA tüm 12 derivasyonu aynı çarpanla ölçeklendirmeyin. Derivasyonlar arası oran bozulur. V1/V6 oranı RBBB/LBBB tanısı için kritiktir; global scale bu oranı bozar. Amplitüd artırma her derivasyon için ayrı (0.9-1.1 arası) yapılmalıdır.

6. **50Hz Notch Filtresi:** 50Hz notch filtresi T dalgasına zarar verebilir. 0.5-40Hz bandpass filtresi yeterlidir. Güç hattı gürültüsü zaten 40Hz üst kesim frekansı ile elimine edilir.

7. **Derivasyon Tamamen Çıkarma:** ASLA düşük kaliteli derivasyonu tamamen çıkarmayın. Model 12 derivasyonun tamamını görmeye alıştırılmalıdır. SQI-Gating ile ağırlığını düşürün, ama fiziksel olarak çıkarmayın. Yarışmada düşük kaliteli kayıtlar çıkacaktır.

8. **Accuracy Metriğine Aldanma:** Accuracy %95 olabilir ama bu aldatıcıdır. Normal sınıfı ezberlemiş olabilirsiniz. Bu yarışmada tek önemli metrik Macro F1'dir. Her karar bu soruya göre verilmelidir: Bu Macro F1'i artırır mı? Sınıf başına F1 takibi zorunludur.

9. **Veri Sızma (Data Leakage):** Validation ve test setinin normalizasyon istatistiklerini kendi içlerinde hesaplamayın. SADECE eğitim setinden hesaplanan istatistikler val/test'e uygulanmalıdır. Aksi halde model test hakkında bilgi sızmış olur.

10. **Aynı Hasta Hem Eğitim Hem Test:** Train, validation ve test CSV'lerindeki patient_id kümelerinin kesişimi sıfır olmalıdır. Her hafta bu kontrol otomatik olarak yapılmalıdır: `assert len(train_patients & val_patients) == 0`.

## 7. Sonuç ve Stratejik Öneri

| Senaryo | Versiyon | Tahmini Macro F1 | Gerekçe |
|---|---|---|---|
| Sadece TEKNOFEST (5.000) | Baseline | 0.88 - 0.90 | Dengeli ama az veri |
| Çoklu veri + Curriculum (A) | Versiyon A | 0.93 - 0.95 | Fizyolojik bilgi + robust preprocessing |
| Çoklu veri + DANN + Ensemble (B) | Versiyon B | 0.94 - 0.96 | Tam hibrit mimari + domain bağımsızlık |
| Ensemble + TTA + Model Soup | Her ikisi | +0.5 - 1.0 puan | Çeşitlendirme ve test zamanı artırma |

**Stratejik Öneri:**

1. Aşamadaki 96.7 skoru, temel altyapının (lead-wise Z-Score, per-patient split, fizyolojik bilgi) sağlam olduğunu kanıtlamıştır. 2. Aşamada bu altyapı 5 sınıf ve çoklu veri seti için genişletilecektir.

0.3 puanlık marjı kapatmak için:

1. **Hafta 1-3:** Sağlam veri altyapısı ve baseline. Preprocessing hatası varsa hiçbir mimari kurtaramaz.
2. **Hafta 4:** İki versiyonu paralel deneyin, Versiyon A'ya dönüş patikasını açık tutun.
3. **Hafta 5-6:** Sınıf-spesifik fizyolojik fix'ler (P-dalga, V1/V6 oranı) ve ensemble.
4. **Hafta 7-8:** Hata avı ve reproducibility. Docker + seed fix + tek komut inference.

**Altın Cümle:** *"Herkes daha derin CNN koyarken, biz EKG'nin fizyolojisini modelin içine gömdük. V1 kalitesi düşükse V6'ya güvenmeyi, AFIB'de RR düzensizliğini, RBBB'de rsR' morfolojisini kendisi öğrendi."*

**Üç Altın Kural:**
1. 12 derivasyon, lead-wise Z-Score, hasta bazında bölme. Bunlar olmazsa proje çöker.
2. Macro F1 tek tanrıdır. Accuracy aldatıcıdır.
3. Görülmemiş veri mindset'i. Her zaman "bu model yarışmada hiç görmediğim bir hastanede çalışır mı?" diye sorgulayın.

---
---

# TEKNOFEST 2026 — 2. Aşama Teknik Rapor

## CardioFusion-5: Çoklu Veri Seti ile 5-Sınıf EKG Sınıflandırma

### Versiyon A (Pragmatik) & Versiyon B (Agresif) Strateji ve Yol Haritası

**Öğrenci Takımı:** CardioFusion-5
**Mevcut Durum:** Macro F1 = 96.7 (2. Sıra)
**Hedef:** Macro F1 ≥ 97.0 (1. Sıra)
**Fark:** 0.3 puanlık marjın kapatılması
**Ekip Büyüklüğü:** 3 Kişi + Mentor
**Zaman Çizelgesi:** 8 Hafta
**Rapor Tarihi:** Mayıs 2026

## Özet ve Stratejik Karar Çerçevesi

Mevcut durumda 96.7 Macro F1 skoru ile 2. sırada yer alınmaktadır. Birinci takım ile aradaki fark yalnızca 0.3 puan olup, bu marjın kapatılması için teknik derinlik ve veri stratejisinin birlikte optimize edilmesi gerekmektedir. Bu raporda, 3 kişilik ekip kapasitesine göre uyarlanmış iki paralel strateji sunulmaktadır:

| Özellik | Versiyon A: Efficient | Versiyon B: Pro |
|---|---|---|
| Mimari | Tek-Branch Enhanced (1D-CNN + SE + Cross-Lead) | Dual-Branch Hibrit (CNN + Beat BiGRU + SQI Gating) |
| Domain Adaptasyonu | Curriculum Learning | Curriculum + DANN |
| Veri Stratejisi | Class-Balanced + Focal Loss | Hard Example Mining + Model Soup |
| Tahmini F1 | 97.0 - 97.2 | 97.3 - 97.6 |
| Risk Seviyesi | Düşük (Stabil) | Orta (Debug karmaşıklığı) |
| Ekip İhtiyacı | 2 Geliştirici + 1 Veri | 2 Geliştirici + 1 Veri (Yoğun) |

**Tavsiye:** İki versiyon da aynı veri altyapısını ve ön işleme hattını paylaşmaktadır. Hafta 1-3 arası ortak ilerlenmeli, Hafta 4'te versiyon ayrımı yapılmalıdır. Eğer Hafta 4 sonunda Versiyon B'nin mimarisi stabil çalışmazsa, Versiyon A'ya anında dönüş patika açık tutulmalıdır.

## 1. Giriş ve Problem Tanımı

### 1.1 Yarışma Kapsamı

TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması 2. Aşama kapsamında, 12 derivasyonlu EKG kayıtlarından 5 kardiyak sınıfın (Normal, AFIB, AFL, LBBB, RBBB) otomatik sınıflandırılması hedeflenmektedir. Değerlendirme metriği Macro F1-Score olup, her sınıfın F1 skoru eşit ağırlıklı olarak ortalanmaktadır.

### 1.2 Mevcut Durum ve Hedef

- **Mevcut skor:** Macro F1 = 96.7 (2. Sıra)
- **Hedef skor:** Macro F1 ≥ 97.0 (1. Sıra)
- **Kapatılması gereken fark:** 0.3 puan
- **Kritik gözlem:** Bu fark, azınlık sınıflarda (özellikle AFL ve LBBB/RBBB ayrımında) yapılacak iyileştirmelerle kapatılabilir düzeydedir.

### 1.3 Veri Portföyü

Eğitim havuzu iki kaynaktan oluşmaktadır:
- **TEKNOFEST Verisi:** 5.000 kayıt, sınıf başına 1.000 örnek, dengeli dağılım, 500Hz örnekleme frekansı, yarışma domain'ini temsil eder.
- **Açık Kaynak Veri Setleri:** ~100.000 kayıt (PhysioNet, PTB-XL, CPSC, Georgia, Chapman), dengesiz dağılım (AFL ~%4), çoğunlukla 250Hz veya 500Hz, farklı cihazlar ve popülasyonlar.

### 1.4 Stratejik Varsayımlar

- 0.3 puanlık kazanç, "daha derin katman" eklemekten ziyade fizyolojik bilginin modele gömülmesi ve domain robustluğun artırılmasıyla elde edilecektir.
- Çoklu veri seti kullanımı, domain overfit riskini beraberinde getirir; bu nedenle Domain-Adversarial Eğitim (DANN) veya Curriculum Learning zorunludur.
- Ekip 3 kişi olduğundan, mimari karmaşıklığı ile debug süresi arasında denge kurulmalıdır.

## 2. Teknik Mimari

### 2.1 Versiyon A: CardioFusion-Efficient (Pragmatik)

Bu versiyon, tek bir ana dal (branch) üzerine kurulmuş, ancak fizyolojik bilgiyi doğrudan modele enjekte eden bir yapıdır. Debug süresi minimum, eğitim stabilitesi maksimum düzeydedir.

| Katman | Açıklama | Çıktı Boyutu |
|---|---|---|
| Girdi | 2500 örnek × 12 derivasyon @ 250Hz | (2500, 12) |
| Ön İşleme | Bandpass 0.5-40Hz + Lead-wise Z-Score + SQI | (2500, 12) + (12,) |
| Ana Dal | 1D-CNN + SE Blocks + Cross-Lead Attention | (128,) |
| Yardımcı Girdi | Precomputed fizyolojik özellikler (RR varyansı, QRS genişliği, P-dalga varlığı) | (8,) |
| Birleştirme | Ana dal çıktısı + Yardımcı özellikler | (136,) |
| Sınıflandırıcı | Dense(128) → Dense(5) + Softmax | (5,) |
| Yardımcı Başlık | 3-sınıf (Normal / Ritim / İletim) Multi-Task Loss | (3,) |

**Cross-Lead Attention Mekanizması:** V1↔V6 ve I↔aVL arasındaki fizyolojik ilişkiyi explicit olarak modelin öğrenmesini sağlar. RBBB tanısı V1'de, LBBB tanısı V1+V6'da yazılıdır; bu mekanizma bu morfolojik bağımlılıkları katman içinde kodlar.

**Yardımcı Özellikler (Branch 2'nin Hafifletilmiş Hali):** NeuroKit2 ile ön hesaplanan ve modele skaler olarak verilen özelliklerdir. Beat segmentasyonu model içinde yapılmaz, bu sayede eğitim hızı 10 kat artar.
- Ortalama kalp hızı, RR aralığı varyansı (AFIB/AFL ayrımı için kritik)
- QRS genişliği, PR aralığı (LBBB/RBBB için kritik)
- P-dalga varlığı ikili göstergesi (0 veya 1)
- V1/V6 QRS genlik oranı

### 2.2 Versiyon B: CardioFusion-Pro (Agresif)

Bu versiyon, tam hibrit mimariyi hedefler. Morfoloji ve ritim bilgisini ayrı dallarda işler, SQI-Gated adaptif birleştirme ile hangi dalın ağırlıklı olacağına karar verir.

| Katman | Versiyon B Özellikleri | Fizyolojik Karşılığı |
|---|---|---|
| Dal A: Morfoloji | 1D-CNN + SE + Cross-Lead Attention + Instance Norm | V1 rsR' (RBBB), V6 geniş R (LBBB) |
| Dal B: Ritim | Beat-Level BiGRU + Self-Attention + RR-Variance | AFIB düzensiz RR, AFL düzenli testere |
| SQI Gating | 12 derivasyonun kalite skoruna göre ağırlıklandırma | V1 kalitesi düşükse V6'ya güven |
| Domain Embedding | 5 veri seti için öğrenilebilir vektör (16-dim) | Cihaz/popülasyon farkını normalize et |
| DANN | Ters gradyan ile domain sınıflandırıcı | Model veri seti bilgisini "unutsun" |
| Birleşik Başlık | 5-sınıf (Focal Loss) + 3-sınıf (Aux Loss) | Ana görev + yardımcı görev |

**Dal B Detayı:** R-tepesi tespiti preprocessing aşamasında yapılır. Her kayıt ~10 nabıza bölünür. Her beat 400 örnek (800ms) olacak şekilde R-tepesi merkeze alınır. Padding + maskeleme ile sabit boyuta getirilir. BiGRU, nabızlar arası zamansal bağımlılığı öğrenir; Attention ise "hangi nabız anormal?" sorusuna cevap verir (örneğin APB/VPB için).

**SQI-Gated Fusion:** Inference sırasında her derivasyonun kalite skoru (kurtosis, baseline wander, QRS band gücü, R-tepesi tespiti) hesaplanır. Düşük kaliteli derivasyonun embedding'i baskılanır, yüksek kaliteli olanınki güçlendirilir. Derivasyon asla tamamen çıkarılmaz; sadece ağırlığı düşürülür. Bu, yarışmada karşılaşılabilecek düşük kaliteli kayıtlara karşı robustluk sağlar.

## 3. Veri Stratejisi ve İşleme Hattı

### 3.1 Ön İşleme Pipeline (Her İki Versiyon İçin Ortak)

Tüm ham kayıtlar aşağıdaki sırayla işlenir. Bu hattın her adımı reproducibility için sabitlenmiştir (seed=42).

| Adım | İşlem | Parametreler | Amaç |
|---|---|---|---|
| 1. Okuma | WFDB formatından sinyal çıkarımı | (N, 12), fs | Ham veri erişimi |
| 2. Hz Uyumu | Polifazik yeniden örnekleme | 500Hz → 250Hz (up=1, down=2) | Hız ve bellek optimizasyonu |
| 3. Filtreleme | Butterworth bandpass | 0.5-40Hz, 4. derece, filtfilt | Baseline wander ve EMG gürültüsü kaldır |
| 4. Normalizasyon | Lead-wise Z-Score | μ ve σ sadece train setinden | Amplitüd farklılıklarını giderme |
| 5. Kalite Taraması | SQI hesaplama (lead-wise) | kSQI + bSQI + pSQI + rSQI | Kalitesiz kayıtların tespiti |
| 6. Etiket Harmoni | SNOMED-CT eşleştirme | 5 hedef sınıf | Farklı veri setlerindeki kod farklılıklarını çözme |
| 7. Çıktı | İşlenmiş tensör | (2500, 12) + label + SQI(12,) | Model girdisi |

**Kritik Notlar:**
- **Hz Uyumu:** Nyquist frekansı 125Hz olup, EKG'deki en yüksek anlamlı frekans ~40Hz (QRS kompleksi) olduğundan 250Hz tamamen yeterlidir. P-QRS-T morfolojisi korunur.
- **Lead-wise Z-Score:** V1 QRS genliği 0.5mV, V5 QRS genliği 2.5mV olabilir. Global ortalama aldığında V1 sinyali "yok olur" ve RBBB tanısı öğrenilemez. Her derivasyon kendi ortalama ve standart sapması ile normalize edilir.
- **50Hz Notch Filtre Kullanılmaz:** T dalgasına zarar verebilir. 0.5-40Hz bandpass yeterlidir.

### 3.2 SNOMED-CT Harmonizasyonu

Farklı veri setlerinde farklı SNOMED kodları bulunmaktadır. Harmonizasyon iki seviyede yapılır:

**Seviye 1 (Üst Sınıf):** Ritim bozukluğu / İletim bozukluğu / Normal
**Seviye 2 (Alt Sınıf):** AFIB / AFL / LBBB / RBBB / Normal

Eğer bir kayıt Seviye 2'ye net olarak eşleşemiyorsa (örneğin SVT gibi ara kodlar), en yakın sınıfa atanır. Hâlâ belirsizlik varsa kayıt eğitim havuzundan çıkarılır. Çift etiketli kayıtlarda (örneğin AFIB+LBBB) dominant etiket seçilir; eşit ağırlıklıysa atılır.

### 3.3 Veri Temizliği ve Kalite Kontrolü

Kalite düşüklüğü gösteren kayıtların eğitim setine sızması, modelin "gürültüyü öğrenmesine" neden olur. Aşağıdaki kriterler uygulanır:
- SQI ortalaması < 0.2 olan kayıtlar eğitim havuzundan çıkarılır.
- Flatline (standart sapma ≈ 0) gösteren derivasyonlar maskeleme ile işaretlenir.
- Süresi < 8 saniye olan kayıtlar atılır (ritim analizi için yetersiz).
- Çoklu etiket çatışması olan kayıtlar atılır.

### 3.4 Eğitim Stratejisi: Curriculum Learning

Çoklu veri seti kullanımında domain overfit riski yüksektir. Bu riski yönetmek için üç aşamalı eğitim (Curriculum Learning) uygulanır:

| Aşama | Epoch | Veri | Strateji | Amaç |
|---|---|---|---|---|
| 1. Temel | 1-20 | Sadece TEKNOFEST (5.000) | Dengeli, DANN kapalı | Model temel kardiyak öğrenmeyi edinir |
| 2. Genişleme | 21-80 | TEKNOFEST + İnternet (~100k) | DANN açık (λ=0.1), Focal Loss, Hard Mining | Farklı popülasyonları gör, domain genelleme |
| 3. İnce Ayar | 81-100 | Sadece TEKNOFEST (5.000) | DANN kapalı, LR 10x düşük | Yarışma domain'ine geri dön, overfit kırılır |

**Sınıf Dengesizliği Çözümleri (Aşama 2):**
- **Class-Balanced Sampler:** Her mini-batch'te tüm sınıflar eşit temsil edilir. AFL ~%4 olduğundan, bu sınıf zorla dahil edilir.
- **Focal Loss (γ=2.0):** Azınlık sınıflarının gradient'ini şişirir, çoğunluk sınıfının (Normal) baskısını azaltır.
- **AFL Artırma:** SMOTE kullanılmaz (zaman serisini bozar). Kopyalama + amplitüd ölçekleme (0.9-1.1) + hafif Gauss gürültüsü ile 3 kat artırma yapılır.
- **Hard Example Mining:** Validation'daki en yüksek kayıp (loss) veren 200 örnek havuzlanır. Her epoch'un %30'u bu "zor örneklerden" oluşturulur.

## 4. Sekiz Haftalık Yol Haritası

### Hafta 1: Veri Altyapısı

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Tüm veri setleri indirme | Veri Mühendisi | dataset_registry.json |
| Çar-Per | SNOMED kod eşleştirme + 5 sınıf harmoni | Veri Mühendisi | snomed_map.py |
| Cum | Hz uyumu (500→250) + bandpass filtreleme | Geliştirici 1 | preprocess.py |
| Cmt | Lead-wise Z-Score + SQI v1 | Geliştirici 1 | sqi_module.py |
| Paz | Veri kalite raporu + per-patient split kontrolü | Mentor + Ekip | quality_report.md |

**Hedef:** 5.000 TEKNOFEST + ~100k internet kaydı harmonize ve temiz.

### Hafta 2: Baseline ve Dengeli Örnekleme

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt | Baseline 1D-CNN (Versiyon A) kurulumu | Geliştirici 1 | baseline_cnn.py |
| Sal | Eğitim döngüsü + WandB loglama | Geliştirici 2 | train_loop.py |
| Çar | Class-Balanced Sampler implementasyonu | Geliştirici 2 | balanced_sampler.py |
| Per | Focal Loss (γ=2.0) + ilk eğitim denemesi | Geliştirici 2 | focal_loss.py |
| Cum | Validation sonuçları + Confusion matrix | Ekip | f1_report_v1.md |
| Cmt | Quiz 1: 12 derivasyon anatomisi + sınıf morfolojileri | Mentor | Kahoot/Quizizz |
| Paz | Hata defteri 1: En düşük F1'li sınıf analizi | Ekip | 01_error_log.md |

**Hedef:** Macro F1 ≥ 0.85 (sadece TEKNOFEST verisi ile). Eğer bu hedef tutmazsa, preprocessing hattında hata vardır (muhtemelen Z-Score veya per-patient split).

### Hafta 3: F1 Maksimizasyonu (Data-Centric)

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Hard Example Mining: Zor örnek havuzu oluşturma | Geliştirici 2 | hard_pool.py |
| Çar | AFL artırma (kopyalama + gürültü, SMOTE yok) | Veri Mühendisi | afl_augment.py |
| Per | Cross-Lead Attention v1 (V1↔V6, I↔aVL) | Geliştirici 1 | cross_lead_attn.py |
| Cum | TTA (Test-Time Augmentation) prototip | Geliştirici 2 | tta_inference.py |
| Cmt | Quiz 2: Focal Loss + augmentation kuralları | Mentor | Kahoot |
| Paz | F1 Kazanç Raporu 1 | Ekip | 02_f1_gains.md |

**Hedef:** Macro F1 ≥ 0.88. AFL F1'i 0.50+ seviyesine çekilmeli.

### Hafta 4: Mimari Geliştirme ve Versiyon Ayrımı

Bu hafta iki paralel patika açılır. Ekip, hafta sonuna kadar hangi versiyonun daha stabil olduğuna karar verir.

| Versiyon | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| A | Cross-Lead Attention + Precomputed fizyolojik özellikler entegrasyonu | Geliştirici 1 | cardiofusion_efficient.py |
| A | Multi-Task Loss (5-sınıf + 3-sınıf yardımcı) | Geliştirici 2 | multi_task_loss.py |
| B | Dal B: Beat extraction + BiGRU + Attention | Geliştirici 1 | branch_rhythm.py |
| B | DANN modülü (Gradient Reversal Layer) | Geliştirici 2 | dann_module.py |
| B | SQI-Gated Fusion katmanı | Geliştirici 1 | sqi_gating.py |
| Ortak | Domain Embedding (5 veri seti ID) | Veri Mühendisi | domain_embed.py |
| Ortak | Hafta sonu karar: Versiyon A veya B seçimi | Mentor + Ekip | mimari_karar.md |

**Karar Kriteri:** Versiyon B'nin debug süresi > 3 gün sürerse veya beat extraction hata oranı > %5 ise, Versiyon A'ya dönülür. Versiyon A, 97.0+ hedefini tek başına karşılayabilir düzeydedir.

### Hafta 5: Sınıf-Spesifik Optimizasyon

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | AFIB/AFL ayrımı: P-dalga otokorelasyonu + RR varyansı | Geliştirici 1 | fix_afib_afl.py |
| Çar | LBBB/RBBB ayrımı: V1 SQI threshold ayarı + V1/V6 cross-attention | Geliştirici 1 | fix_lbbb_rbbb.py |
| Per | Hyperparameter sweep (LR, dropout, attention heads) | Geliştirici 2 | sweep_report.md |
| Cum | Sınıf başına F1 analizi + Confusion matrix derinlemesine | Ekip | per_class_f1.csv |
| Cmt | Quiz 3: Hata analizi fizyolojisi | Mentor | Kahoot |
| Paz | F1 Kazanç Raporu 2 | Ekip | 03_f1_gains.md |

**Hedef:** Macro F1 ≥ 0.91. AFIB↔AFL ve LBBB↔RBBB karışım oranı %50 azaltılmalı.

### Hafta 6: Ensemble ve Çapraz-Veri-Seti Doğrulama

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | 3-Seed Ensemble (42, 123, 2026) eğitimi | Geliştirici 2 | ensemble_3seed.py |
| Çar | Model Soup: 3 checkpoint ağırlık ortalaması | Geliştirici 2 | model_soup.py |
| Per | Leave-One-Dataset-Out protokolü | Veri Mühendisi | leave_one_out.py |
| Cum | Cross-dataset rapor: 5 kombinasyonun F1 ortalaması ve std | Ekip | cross_dataset_report.md |
| Cmt | Quiz 4: Domain genelleme neden önemli? | Mentor | Kahoot |
| Paz | Karar: Overfit var mı? (Std > 0.05 ise alarm) | Mentor + Ekip | robustluk_karari.md |

**Kabul Kriteri:** 5 farklı leave-one-out kombinasyonunun Macro F1 standart sapması < 0.05 olmalıdır. Daha yüksekse domain overfit vardır ve Aşama 2 (Curriculum) parametreleri revize edilmelidir.

### Hafta 7: Hata Avı ve İnce Ayar

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Confusion matrix derin analizi (veri seti bazında) | Ekip | error_analysis_v2.py |
| Çar | Final ensemble seçimi (cross-dataset F1 en yüksek 3 checkpoint) | Geliştirici 2 | final_ensemble.py |
| Per | Inference pipeline optimizasyonu (tek komut çalıştırma) | Geliştirici 1 | inference_final.py |
| Cum | Full pipeline test: Görülmemiş hasta simülasyonu | Ekip | pipeline_test.md |
| Cmt | Quiz 5: Final sistem testi | Mentor | Kahoot |
| Paz | Hata defteri 2: Son aksiyonlar | Ekip | 04_final_actions.md |

**Hedef:** Macro F1 ≥ 0.93 (Versiyon A) veya ≥ 0.95 (Versiyon B). Submission öncesi son %2'lik kazancın alınması.

### Hafta 8: Teslimat ve Raporlama

| Gün | Görev | Sorumlu | Çıktı |
|---|---|---|---|
| Pzt-Sal | Docker container oluşturma ve test | Geliştirici 1 | Dockerfile + docker_test.md |
| Çar-Per | Teknik rapor yazımı (problem, mimari, sonuçlar, hata analizi) | Mentor + Ekip | technical_report.pdf |
| Cum | Sunum slaytları (max 10 slide) | Mentor | presentation.pptx |
| Cmt | Final submission format kontrolü | Ekip | submission_package/ |
| Paz | Son review ve teslimat | Mentor + Ekip | TEKNOFEST Submission |

**Önemli:** Docker container, inference script'in tek komutla çalıştığını doğrulamalıdır. Reproducibility için seed fix (42), config.yaml ve requirements.txt kesin versiyonlarla birlikte teslim edilmelidir.

## 5. Riskler ve Önlemler

| Risk | Olası Etki | Önlem | Tedbir Seviyesi |
|---|---|---|---|
| Domain Overfit | İnternet verisine aşırı uyum, yarışma verisinde düşük F1 | DANN + Curriculum + Leave-one-out test | Yüksek |
| Etiket Hatası | Model yanlış fizyoloji öğrenir, karışıklık artar | SNOMED harmoni + kalite filtre + belirsiz atma | Yüksek |
| Sınıf Dengesizliği | AFL F1 çöker, Macro F1 düşer | Balanced sampler + Focal Loss + AFL artırma | Yüksek |
| Mimari Karmaşıklığı | Debug süresi uzar, haftalık hedefler kaçırılır | Versiyon A yedek patikası, Hafta 4 karar noktası | Orta |
| V1 Kalitesi Düşüklüğü | RBBB/LBBB karışımı artar | SQI-Gating (Versiyon B) veya V1 threshold (Versiyon A) | Orta |
| Overfit (Teknofest) | Aşama 3 ihmal edilirse internet verisi yarışma domainini bozar | Aşama 3 zorunlu fine-tune, LR düşürme | Yüksek |

**Domain Overfit Riski Detayı:** Çoklu veri seti kullanımının en büyük riski, modelin "hangi hastaneden geldiğini" öğrenip kardiyak bilgiyi ikincil plana atmasıdır. DANN modülü, özellik çıkarıcıyı (feature extractor) veri seti bilgisini "unutmaya" zorlar. Eğer leave-one-out testinde standart sapma > 0.05 çıkarsa, DANN lambda parametresi artırılır (0.1 → 0.3) veya Aşama 2 epoch sayısı azaltılır.

## 6. Sakınılması Gereken Kritik Hususlar

Aşağıdaki hatalar, dokümanlarda ve literatürde tekrar tekrar karşılaşılan ve F1 skorunu %5-%10 düşürebilen "ölümcül" hatalardır. Bu kuralların ihlali, projenin başarısız olması için yeterlidir.

1. **Global Z-Score Normalizasyonu:** ASLA tüm 12 derivasyonu birlikte normalize etmeyin. V1 QRS genliği 0.5mV, V5 QRS genliği 2.5mV olabilir. Global ortalama alındığında V1 sinyali "yok olur", V5 baskın hale gelir. RBBB tanısı V1'e bağlıdır; V1'i yok ederseniz model RBBB öğrenemez. LBBB tanısı V6'ya bağlıdır; V6'yı baskılarsanız LBBB öğrenemez. Her derivasyon kendi içinde normalize edilmelidir.

2. **Rastgele Hasta Bölmesi (Per-Patient Split İhlali):** ASLA kayıt bazında rastgele bölme yapmayın. Aynı hastanın 5 farklı EKG kaydı varsa, 4'ü eğitim 1'i teste gidebilir. Model o hastayı "ezberlemiş" olur. Yarışma finalinde yeni hasta görüldüğünde model patlar. Bir hasta ya eğitimdedir, ya doğrulamada, ya testtedir. Asla ikisinde birden olmamalıdır.

3. **Zaman Kaydırma (Time-Shift) Artırma Tekniği:** ASLA sinyali döngüsel olarak kaydırmayın (`np.roll`). P-QRS-T temporal ilişkisini bozar. P dalgası QRS'ten önce gelmelidir; kaydırırsanız bu fizyolojik sıra bozulur. Model ritim bozukluklarını öğrenemez.

4. **Rastgele Kırpma (Random Crop):** ASLA sinyalin başını veya sonunu rastgele kırpmayın. Baştaki P dalgası veya sondaki T dalgası kaybolabilir. Ritim analizi için bu dalgalar kritiktir.

5. **Genel Amplitüd Ölçeklendirme (Global Scale):** ASLA tüm 12 derivasyonu aynı çarpanla ölçeklendirmeyin. Derivasyonlar arası oran bozulur. V1/V6 oranı RBBB/LBBB tanısı için kritiktir; global scale bu oranı bozar. Amplitüd artırma her derivasyon için ayrı (0.9-1.1 arası) yapılmalıdır.

6. **50Hz Notch Filtresi Kullanımı:** 50Hz notch filtresi T dalgasına zarar verebilir. 0.5-40Hz bandpass filtresi yeterlidir. Güç hattı gürültüsü (50Hz) zaten 40Hz üst kesim frekansı ile elimine edilir.

7. **Derivasyon Tamamen Çıkarma (Atma):** ASLA düşük kaliteli derivasyonu tamamen çıkarmayın. Model 12 derivasyonun tamamını görmeye alıştırılmalıdır. SQI-Gating ile ağırlığını düşürün, ama fiziksel olarak çıkarmayın. Yarışmada karşınıza düşük kaliteli kayıtlar çıkacaktır.

8. **Accuracy Metriğine Aldanma:** Accuracy %95 olabilir ama bu aldatıcıdır. Normal sınıfı ezberlemiş olabilirsiniz. Bu yarışmada tek önemli metrik Macro F1'dir. Her karar "bu Macro F1'i artırır mı?" sorusuna göre verilmelidir. Sınıf başına F1 takibi zorunludur.

9. **Veri Sızma (Data Leakage):** Validation ve test setinin normalizasyon istatistiklerini (mean, std) kendi içlerinde hesaplamayın. SADECE eğitim setinden hesaplanan istatistikler val/test'e uygulanmalıdır. Aksi halde model test hakkında bilgi sızmış olur.

10. **Aynı Hasta Hem Eğitim Hem Test:** Train, validation ve test CSV'lerindeki patient_id kümelerinin kesişimi sıfır olmalıdır. Her hafta bu kontrol otomatik olarak yapılmalıdır: `assert len(train_patients & val_patients) == 0`.

## 7. Sonuç ve Stratejik Öneri

### 7.1 Beklenen Performans

| Senaryo | Versiyon | Tahmini Macro F1 | Gerekçe |
|---|---|---|---|
| Sadece TEKNOFEST (5.000) | Baseline | 0.88 - 0.90 | Dengeli ama az veri |
| Çoklu veri + Curriculum (A) | Versiyon A | 0.93 - 0.95 | Fizyolojik bilgi + robust preprocessing |
| Çoklu veri + DANN + Ensemble (B) | Versiyon B | 0.94 - 0.96 | Tam hibrit mimari + domain bağımsızlık |
| Ensemble + TTA + Model Soup | Her ikisi | +0.5 - 1.0 puan | Çeşitlendirme ve test zamanı artırma |

### 7.2 Stratejik Öneri

Mevcut 96.7 skorundan 97.0+ hedefine ulaşmak için "daha derin katman" eklemek yerine fizyolojik bilginin modele gömülmesi ve domain robustluğun artırılması gerekmektedir. Önerilen yol haritası şudur:

1. **Hafta 1-3:** Sağlam bir veri altyapısı ve baseline ile başlayın. Preprocessing hatası varsa hiçbir mimari kurtaramaz.
2. **Hafta 4:** İki versiyonu paralel deneyin, ancak Versiyon A'ya dönüş patikasını açık tutun.
3. **Hafta 5-6:** Sınıf-spesifik fizyolojik fix'ler (P-dalga, V1/V6 oranı) ve ensemble ile kazancı alın.
4. **Hafta 7-8:** Hata avı ve reproducibility. Docker + seed fix + tek komut inference olmadan submission yapılmamalıdır.

**Altın Cümle:** *"Herkes daha derin CNN koyarken, biz EKG'nin fizyolojisini modelin içine gömdük. V1 kalitesi düşükse V6'ya güvenmeyi, AFIB'de RR düzensizliğini, RBBB'de rsR' morfolojisini kendisi öğrendi."*

**Üç Altın Kural:**
1. 12 derivasyon, lead-wise Z-Score, hasta bazında bölme. Bunlar olmazsa proje çöker.
2. Macro F1 tek tanrıdır. Accuracy aldatıcıdır.
3. Görülmemiş veri mindset'i. Her zaman "bu model yarışmada hiç görmediğim bir hastanede çalışır mı?" diye sorgulayın.

---
---

# TEKNOFEST 2026 — TEKNİK RAPOR (TASLAK)

## CardioFusion-5 Multi-Dataset: Çoklu Veri Seti ile 5-Sınıf EKG Sınıflandırma

## 1. GİRİŞ

### 1.1 Problemin Tanımı

TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması 2. Aşama kapsamında, 12 derivasyonlu EKG kayıtlarından 5 kardiyak sınıfın (Normal, AFIB, AFL, LBBB, RBBB) otomatik sınıflandırılması hedeflenmektedir. Yarışmada kullanılan değerlendirme metriği Macro F1-Score'dur.

### 1.2 Motivasyon

Mevcut yarışmacıların büyük çoğunluğu TEKNOFEST'in sağladığı 5.000 kayıtlık dengeli veri seti ile sınırlı kalmaktadır. Ancak EKG sınıflandırma problemlerinde, modelin farklı hasta popülasyonlarını, cihazları ve klinik koşulları görmesi genelleme yeteneğini kritik ölçüde artırır. Bu nedenle biz, 5 farklı açık kaynaklı veri setini harmanlayarak toplam 85.000+ kayıtlık bir eğitim havuzu oluşturduk.

## 2. KULLANILAN VERİ SETLERİ

### 2.1 Veri Seti Portföyü

| # | Veri Seti | Kaynak | Toplam Kayıt | Kullanılan | Hz | Format |
|---|---|---|---|---|---|---|
| 1 | PhysioNet ECG Arrhythmia | Ana veri | 45.152 | ~35.000 | 500 | .mat/.hea |
| 2 | PTB-XL | Almanya | 21.837 | ~18.000 | 500 | .mat/.hea |
| 3 | CPSC 2018 + Extra | Çin | 10.330 | ~8.000 | 500 | .mat/.hea |
| 4 | Georgia 12-Lead | ABD | 10.344 | ~8.000 | 500 | .mat/.hea |
| 5 | Chapman-Shaoxing/Ningbo | Çin | 45.182 | ~35.000 | 500 | .mat/.hea |
| | **TOPLAM (Ham)** | | **132.845** | | | |
| | **TOPLAM (Filtreli)** | | **~104.000** | | | |
| | **TOPLAM (5 Sınıf)** | | **~85.000** | | | |
| | TEKNOFEST | | 5.000 | 5.000 | 500 | .mat/.hea |

### 2.2 Veri Seti Seçim Kriterleri

Tüm veri setleri aşağıdaki kriterlere göre seçilmiştir:
- 12 derivasyonlu EKG kayıtları içermeli
- SNOMED-CT etiket standardı kullanmalı
- Açık erişimli olmalı
- WFDB formatında (.mat/.hea) olmalı
- Etik güvenilirliği literatürde doğrulanmış olmalı

### 2.3 Veri Kalite Filtresi

Ham 132.845 kayıttan:
1. Sinyal kalitesi düşük kayıtlar atıldı (SQI < 0.3)
2. Aşırı gürültülü kayıtlar atıldı (SNR < 10 dB)
3. Etiket güvenilirliği zayıf kayıtlar atıldı (çoklu etiket çatışması)
4. Süresi yetersiz kayıtlar atıldı (< 8 saniye)

**Sonuç:** ~104.000 yüksek kaliteli kayıt kaldı.

### 2.4 SNOMED-CT Harmonizasyonu

Farklı veri setlerinde farklı SNOMED kodları bulunmaktadır. Örneğin:
- PTB-XL: 164889003 (AFIB)
- CPSC: 164889003 (AFIB) — aynı
- Chapman: 164889003 (AFIB) — aynı

Ancak bazı farklılıklar:
- PTB-XL: 713422000 (AT — Atrial Tachycardia)
- CPSC: 426761007 (SVT)

Bu kodlar yarışmadaki 5 sınıf ile doğrudan eşleşmeyebilir. Bu nedenle aşağıdaki harmoni stratejisi uygulanmıştır:

```
SNOMED Kodu -> Üst Sınıf (Ritim/İletim/Normal)
            -> Alt Sınıf (AFIB/AFL/LBBB/RBBB/Normal)
            -> Eğer direkt eşleşme yok -> En yakın sınıfa ata
            -> Eğer hâlâ belirsiz -> Kaydı at
```

### 2.5 Frekans Uyumu

Tüm veri setleri orijinalde 500 Hz'dir. Ancak donanım kısıtları nedeniyle:
- **Alt örnekleme:** 500 Hz -> 250 Hz
- **Yöntem:** `scipy.signal.resample_poly` (polifazik resampling)
- **Neden:** Nyquist frekansı 125 Hz, EKG'deki en yüksek önemli frekans ~40 Hz (QRS). 250 Hz yeterli.
- **Klinik bilgi kaybı:** Yok. P-QRS-T morfolojisi korunur.

```python
from scipy.signal import resample_poly

def resample_to_250hz(signal, original_fs=500):
    if original_fs == 250:
        return signal
    elif original_fs == 500:
        return resample_poly(signal, up=1, down=2, axis=0)
    else:
        raise ValueError(f"Desteklenmeyen frekans: {original_fs}")
```

## 3. ÖN İŞLEME PIPELINE

### 3.1 Pipeline Akışı

```
Ham WFDB Kaydı (.mat/.hea)
        |
        v
[1] WFDB Okuma -> signal: (N, 12), fs: int
        |
        v
[2] Hz Uyumu -> 250 Hz'e resample
        |
        v
[3] Bandpass Filtre (0.5-40 Hz, 4. derece Butterworth)
        |
        v
[4] Lead-wise Z-Score Normalizasyon
        |
        v
[5] SQI Hesaplama -> (12,) skor
        |
        v
[6] Kalite Filtresi -> SQI < 0.3 olan kayıtları at
        |
        v
[7] Etiket Harmonizasyonu -> 5 sınıf
        |
        v
İşlenmiş Kayıt: (2500, 12), label: int, sqi: (12,)
```

### 3.2 Bandpass Filtreleme

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=250, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    filtered = np.zeros_like(signal)
    for lead in range(12):
        filtered[:, lead] = filtfilt(b, a, signal[:, lead])
    return filtered
```

**Filtreleme sonrası kontrol:** Her batch'te rastgele 1 örnek seçilerek P dalgası görünürlüğü kontrol edilir.

### 3.3 Lead-Wise Z-Score Normalizasyonu

```python
def normalize_leadwise(signal, train_stats=None):
    if train_stats is not None:
        mean = train_stats['mean']  # (12,)
        std = train_stats['std']    # (12,)
    else:
        mean = np.mean(signal, axis=0)
        std = np.std(signal, axis=0)
    return (signal - mean) / (std + 1e-6), {'mean': mean, 'std': std}
```

**Kural:** Train istatistikleri val/test'e uygulanır. Asla val/test kendi istatistiklerini kullanmaz.

## 4. MİMARİ

### 4.1 Genel Mimari: CardioFusion-5 Multi-Dataset

```
GİRDİ: (2500, 12) @ 250 Hz
        |
        v
+---------------------+
|   PREPROCESSING     |
|   (Filtre + Z-Score)|
+---------------------+
        |
        v
+---------------------+
|  DOMAIN EMBEDDING    |
|  (5 veri seti ID)    |
|  Output: (5,)        |
+---------------------+
        |
        v
+---------------------+
|  FEATURE EXTRACTOR   |
|  (1D-CNN + SE)       |
|  12 lead -> (128,)   |
+---------------------+
        |
        +----> [Domain Classifier] -> L_domain
        |
        v
+---------------------+
|  FUSION              |
|  [128 | 32] = 160    |
|  -> Dense(128)        |
+---------------------+
        |
        v
+---------------------+
|  MULTI-TASK HEADS    |
|                       |
|  HEAD 1: 5 sınıf      |
|  (Normal/AFIB/AFL/    |
|   LBBB/RBBB)          |
|                       |
|  HEAD 2: 3 sınıf      |
|  (Normal/Ritim/       |
|   İletim)             |
+---------------------+
```

### 4.2 Domain-Adversarial Training (DANN)

Farklı veri setlerinden gelen kayıtların dağılımı farklıdır (domain shift). Modelin veri seti bilgisini "unutmasını" sağlamak için DANN kullanılmıştır.

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = nn.Linear(128, 5)
        self.domain_classifier = nn.Linear(128, 5)  # 5 veri seti

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        # Ters gradient
        reversed = GradientReversalFunction.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed)
        return class_logits, domain_logits
```

**Loss:** `L = L_class + lambda * L_domain`

### 4.3 Curriculum Learning

Aşamalı eğitim stratejisi:

```
Aşama 1 (Epoch 1-20): SADECE TEKNOFEST verisi (5.000, dengeli)
• Model temel kardiyak öğrenmeyi öğrenir
• Dengeli veride stabil başlangıç

Aşama 2 (Epoch 21-80): TEKNOFEST + Internet (85.000, dengesiz)
• Farklı hasta popülasyonlarını öğrenir
• Class-balanced sampler + Focal Loss + Hard Example Mining
• DANN aktif

Aşama 3 (Epoch 81-100): SADECE TEKNOFEST verisi (fine-tune)
• Yarışmaya özgü dağılıma geri dön
• Overfit'i engelle
• Domain classifier devre dışı
```

## 5. EĞİTİM STRATEJİSİ

### 5.1 Sınıf Dengesizliği Çözümleri

Internet verilerinde sınıf dağılımı dengesizdir:

| Sınıf | Tahmini Oran |
|---|---|
| Normal | ~47% |
| AFIB | ~18% |
| AFL | ~4% |
| LBBB | ~14% |
| RBBB | ~17% |

**Çözümler:**
1. **Class-Balanced Sampler:** Her mini-batch'te tüm sınıflar eşit temsil edilir
2. **Focal Loss:** gamma=2.0, azınlık sınıflarına odaklanır
3. **Hard Example Mining:** Modelin en çok karıştırdığı örnekler tekrar gösterilir
4. **AFL Oversampling:** SMOTE veya kopyalama ile AFL verisi 3x artırılır

### 5.2 Augmentasyon

| Yapılır | Yapılmaz |
|---|---|
| Lead-wise amplitude scale (0.9-1.1) | Zaman kaydırma |
| Lead-wise Gaussian noise (SNR>20dB) | Rastgele crop |
| Lead dropout (1-2 lead) | Global amplitude scale |
| Mixup (alpha=0.2) | — |

### 5.3 Hyperparametreler

```yaml
seed: 42
sampling_rate: 250
filter_low: 0.5
filter_high: 40
filter_order: 4
batch_size: 64
lr: 0.001
weight_decay: 1e-4
focal_gamma: 2.0
dann_lambda: 0.1
curriculum_epochs: [20, 60, 20]  # Aşama 1, 2, 3
patience: 15
```

## 6. DEĞERLENDİRME

### 6.1 Metrikler

**Birincil Metrik:** Macro F1-Score

```python
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_true, y_pred, average='macro')
```

**İkincil Metrikler:**
- Sınıf başına F1
- Confusion matrix
- Cross-dataset F1 (leave-one-out)

### 6.2 Cross-Dataset Değerlendirme

Her veri seti ayrı test edilir:

```python
for test_dataset in ['physionet', 'ptbxl', 'cpsc', 'georgia', 'chapman']:
    train_datasets = [d for d in all_datasets if d != test_dataset]
    model = train_on(train_datasets)
    f1 = evaluate_on(test_dataset)
    print(f"Train: {train_datasets} -> Test: {test_dataset} | F1: {f1:.3f}")
```

**Kabul Kriteri:** Std < 0.05 (domain overfit yok)

## 7. SONUÇLAR

### 7.1 Beklenen Performans

| Senaryo | Tahmini Macro F1 |
|---|---|
| Sadece TEKNOFEST (5.000) | 0.88-0.90 |
| Çoklu veri (85.000) + DANN | 0.90-0.93 |
| Ensemble + TTA | 0.92-0.95 |

### 7.2 Avantajlar

1. **Daha fazla veri:** 85.000 vs 5.000
2. **Daha iyi genelleme:** 5 farklı hasta popülasyonu
3. **Daha robust model:** Farklı cihazlara karşı dayanıklı
4. **Domain adaptasyon:** DANN ile veri seti bağımsızlığı

### 7.3 Riskler

1. **Domain overfit:** Çözüm: DANN + Curriculum + Leave-one-out
2. **Etiket hatası:** Çözüm: SNOMED harmonizasyonu + kalite filtre
3. **Sınıf dengesizliği:** Çözüm: Balanced sampler + Focal Loss + Oversampling

## 8. SONUÇ

Bu çalışmada, TEKNOFEST 2026 2. Aşama kapsamında 5-sınıf EKG sınıflandırma problemi için çoklu veri seti stratejisi sunulmuştur. 5 farklı açık kaynaklı veri seti harmanlanarak 85.000+ kayıtlık bir eğitim havuzu oluşturulmuş, domain-adversarial training ve curriculum learning ile genelleme yeteneği artırılmıştır.

**Ana Katkılar:**
1. Çoklu veri seti entegrasyonu ve SNOMED harmonizasyonu
2. Domain-adversarial training ile cihaz/popülasyon bağımsızlığı
3. Curriculum learning ile aşamalı eğitim
4. SQI-gated cross-lead attention ile kalite farkındalığı

**Beklenen Sonuç:** Macro F1 >= 0.92

## KAYNAKLAR

1. PhysioNet ECG Arrhythmia Dataset v1.0.0
2. PTB-XL Dataset
3. CPSC 2018 Challenge Dataset
4. Georgia 12-Lead ECG Dataset
5. Chapman-Shaoxing/Ningbo Dataset
6. Ganin et al. "Domain-Adversarial Training of Neural Networks" (JMLR 2016)
7. Bengio et al. "Curriculum Learning" (ICML 2009)
8. Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
