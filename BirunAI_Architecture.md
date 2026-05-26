# BirunAI — CardioFusion-5: 2. Aşama Ana Mimari Spesifikasyonu

**TEKNOFEST 2026 — Sağlıkta Yapay Zeka Yarışması, Lise Kategorisi**  
**Doküman Türü: Master Architecture Specification (Single Source of Truth)**  
**Revizyon: 3.0 — PhysioNet 2020 Şampiyonları Entegrasyonu (Mayıs 2026)**

---

> **Kapsam:** Bu doküman, BirunAI CardioFusion-5 sisteminin 2. aşama için tüm teknik kararlarını,
> PhysioNet 2020 yarışması ilk 5 takımının (prna, Between a ROC, HeartBeats, Triage, SharifAITeam) "Altın Stratejileri" ile harmanlayarak tek kaynak-doğruluk prensibiyle kayıt altına alır.
>
> **1. Aşama Skoru:** Macro F1 = 0.7829 (3 sınıf: Normal / Ritim / İletim), 97 puan ile geçildi.  
> **2. Aşama Hedefi:** Macro F1 ≥ 0.93 (5 sınıf: Normal / AFIB / AFL / LBBB / RBBB)

---

## İçindekiler

1. [Sistem Özeti ve Hedefler](#1-sistem-özeti-ve-hedefler)
2. [Veri Boru Hattı (Data Pipeline) ve Zenginleştirme](#2-veri-boru-hattı-data-pipeline-ve-zenginleştirme)
3. [Yapay Zeka Mimarisi — "Altın Strateji" (Unified Model)](#3-yapay-zeka-mimarisi--altın-strateji-unified-model)
4. [Eğitim Stratejisi ve Domain Adaptasyonu](#4-eğitim-stratejisi-ve-domain-adaptasyonu)
5. [Eşik Optimizasyonu ve Karar Mekanizması](#5-eşik-optimizasyonu-ve-karar-mekanizması)
6. [Mevcut Kodun Adapte Edilmesi — 1. Aşama → 2. Aşama Geçiş Haritası](#6-mevcut-kodun-adapte-edilmesi)
7. [8 Haftalık Yol Haritası](#7-sekiz-haftalık-yol-haritası)
8. [Riskler ve Önlemler](#8-riskler-ve-önlemler)
9. [Özet: Dört Altın Kural](#9-özet-dört-altın-kural)

---

## 1. Sistem Özeti ve Hedefler

### 1.1 Problem Tanımı

12 derivasyonlu EKG kayıtlarından **5 kardiyak sınıfın** otomatik sınıflandırılması:

| Sınıf Kodu | Sınıf Adı | Türkçe | Ayırt Edici Özellik | Kritik Lead(ler) |
|:---:|---|---|---|---|
| 0 | Normal | Normal Sinüs Ritmi | Düzenli RR, P her QRS öncesinde, QRS < 120 ms | Tümü |
| 1 | AFIB | Atriyal Fibrilasyon | Düzensiz RR (irregularly irregular), P yok, kaotik f-dalgası | **Lead II** |
| 2 | AFL | Atriyal Flutter | Düzenli testere dişi (250-350 atım/dk), RR varyansı **düşük** | **Lead II, V1** |
| 3 | LBBB | Sol Dal Bloğu | QRS > 120 ms, V1'de rS/QS, V6'da geniş monofazik R | **V1, V6, I, aVL** |
| 4 | RBBB | Sağ Dal Bloğu | QRS > 120 ms, V1'de rsR' ("tavşan kulağı"), V6'da geniş S | **V1** (en kritik) |

### 1.2 Değerlendirme Metriği

$$\text{Macro F1} = \frac{1}{5} \sum_{c=1}^{5} F1_c$$

- **Accuracy kullanılmaz.**
- **Her sınıf F1 ≥ 0.85** hedeflenir; toplam Macro F1 ≥ 0.93.

### 1.3 Veri Seti Havuzu

TEKNOFEST 2026 yarışma verisi (5.000 kayıt) ve İnternet açık kaynak EKG veri setleri (PhysioNet, PTB-XL, Georgia vb. toplam ~85.000 kayıt) kullanılarak **Çoklu Veri Seti (Multi-Dataset)** stratejisi benimsenecektir.

---

## 2. Veri Boru Hattı (Data Pipeline) ve Zenginleştirme

PhysioNet 2020 şampiyonlarının (özellikle Takım 4 ve 5) veriyi hazırlama yöntemleri entegre edilmiştir.

### 2.0 Pipeline Akışı

```text
Ham WFDB Kaydı
       │
       ▼
[1] Okuma (N, 12)
       │
       ▼
[2] Hz Uyumu (500 Hz → 250 Hz)
       │
       ▼
[3] Bandpass Filtreleme (0.5 – 40 Hz)
       │
       ▼
[4] ArcTan Normalizasyonu (R-Peak Baskılama)
       │
       ▼
[5] Lead-Wise Z-Score (Bağımsız Lead)
       │
       ▼
[6] Çıktı: (2500, 12) + Etiketler
```

### 2.1 ArcTan Normalizasyonu (Takım 4: Triage Stratejisi)

**Gerekçe:** EKG'deki R tepeleri (R-peaks) voltaj olarak çok yüksektir. Model R tepesine odaklanırken P dalgası gibi küçük morfolojileri gözden kaçırabilir. Sinyal ArcTan fonksiyonundan geçirilerek R tepeleri baskılanır.
*Formül:* `signal = np.arctan(signal)` (Z-Score öncesinde veya sonrasında uygulanarak test edilecek).

### 2.2 Lead-Wise Z-Score Normalizasyonu

Her derivasyon **kendi** $\mu$ ve $\sigma$ değerleriyle normalize edilir. Global normalizasyon RBBB tanısı koyduran V1'i ezeceği için **YASAKTIR**. $\mu$ ve $\sigma$ sadece Eğitim (Train) setinden hesaplanır.

### 2.3 Agresif Augmentation (Takım 5: SharifAITeam Stratejisi)

Sinyali temizlemek yerine modeli gürültüye alıştırma prensibi:
1. **Düşük Frekanslı Gürültü (Sinusoidal Drift / Baseline Wander):** Rastgele faz, frekans ve genlikte sinüzoidal dalga ekleme.
2. **Lead Dropout (Derivasyon Kopması):** Rastgele 1 veya 2 lead'i sıfırlama veya tamamen gürültüyle değiştirme.
3. **Lead-Wise Amplitüd Ölçekleme:** Her lead bağımsız olarak ×[0.9, 1.1] arasında ölçeklenir.
4. **Gaussian Noise:** Sıfır ortalamalı, düşük standart sapmalı Gauss gürültüsü.
*NOT: `np.roll` (zaman kaydırma) YASAKTIR (P-QRS-T sırasını bozar).*

---

## 3. Yapay Zeka Mimarisi — "Altın Strateji" (Unified Model)

Tüm takımların en iyi yönlerini birleştiren **CardioFusion-5** mimarisi:

```text
Girdi: (batch, 2500, 12) @ 250 Hz
             │
             ▼
┌──────────────────────────────────────────────┐
│  1. FEATURE EXTRACTOR: SE-ResNet             │ (Takım 2 ve 3)
│  - Geniş Kernel (İlk katman kernel=15)       │
│  - Squeeze-and-Excitation (SE) Blokları ile  │
│    Lead Kanal Ağırlıklandırması              │
│  → Çıktı: (batch, zaman_adımı, 256)          │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  2. ZAMANSAL İŞLEME: Transformer Encoder     │ (Takım 1 ve 4)
│  - Multi-Head Self-Attention                 │
│  - RNN/LSTM yerine tam paralel işleme        │
│  → Global Average Pooling (batch, 128)       │
└──────────────────────┬───────────────────────┘
                       │
         ┌─────────────┴────────────┐
         │                          │
         ▼                          │
┌─────────────────────────┐         │ (Takım 1: Wide & Deep)
│  3. WIDE FEATURES       │         │
│  - NeuroKit2 ile çıkarılmış       │
│  (HR, RR, Yaş, Cinsiyet)          │
│  → (batch, 8)           │         │
└────────┬────────────────┘         │
         │                          │
         ▼                          ▼
┌──────────────────────────────────────────────┐
│  4. BİRLEŞTİRME (CONCATENATION)              │
│  - concat(Deep[128] + Wide[8]) = (batch, 136)│
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴────────────────┐
       │                                │
       ▼                                ▼
┌──────────────┐                 ┌──────────────┐
│  ANA KARAR   │                 │  DANN MODÜLÜ │ (Takım 5)
│  KATMANI     │                 │(Gradient Rev)│
│  (5 Sınıf)   │                 │ (5 Domain)   │
└──────────────┘                 └──────────────┘
```

### 3.1 Squeeze-and-Excitation ResNet (SE-ResNet)
12 lead EKG verisi için mükemmel bir uyumdur. SE bloğu her kanalın (lead) varyansını inceleyerek önemli kanallara yüksek ağırlık verir. Örneğin RBBB hastasında V1 kanalının feature map ağırlığı otomatik artar.

### 3.2 Transformer Encoder
Zaman serisinin başı ve sonu (örneğin düzensiz AFIB atımları) arasındaki ilişkiyi Self-Attention mekanizması ile LSTM'e kıyasla çok daha hızlı ve etkili öğrenir.

### 3.3 Wide & Deep Yaklaşımı
Ağ sadece ham sinyali değil, uzman bilgisini de (fizyolojik özellikler) alır. Yaş, Cinsiyet, Minimum/Maksimum Kalp Hızı, RR varyansı gibi "Wide" özellikler modelin en son katmanına enjekte edilir.

---

## 4. Eğitim Stratejisi ve Domain Adaptasyonu

### 4.1 Çoklu Veri Seti ve DANN (Domain-Adversarial Neural Network)
Veriler farklı cihazlar ve hastanelerden geldiği için modelin cihaz özelliklerini ezberlemesi engellenmelidir. Takım 5'in stratejisi olan DANN kullanılır.
- **Gradient Reversal Layer:** Geri yayılım (backprop) sırasında domain sınıflandırıcısının gradyanı ters çevrilerek eksi (-) olarak ağın ana gövdesine iletilir. Böylece özellik çıkarıcı "Domain Bağımsız" (Domain Invariant) özellikler üretmeye zorlanır.

### 4.2 Curriculum Learning (3 Aşamalı Eğitim)
1. **Aşama 1 (Temel):** Yalnızca TEKNOFEST (5.000, dengeli). Dengeli yapı ile temel kardiyak morfoloji öğrenilir.
2. **Aşama 2 (Genişleme):** TEKNOFEST + İnternet (~90K). DANN modülü aktiftir. Farklı popülasyon/cihaz genellemesi yapılır.
3. **Aşama 3 (İnce Ayar/Finetune):** Yalnızca TEKNOFEST. LR düşürülür, yarışma domain'ine özel F1 maksimizasyonu yapılır.

### 4.3 Kayıp Fonksiyonu (Sign Loss / Weighted BCE)
AFL gibi veri setinde az (%4) bulunan sınıfların Macro F1'i çökertmesini önlemek için:
- **Sign Loss (Takım 3):** Doğru tahminlerin (True Negatives dahil) kaybını küçültüp, yanlış tahminlere (False Positives/Negatives) ağır ceza veren özel bir Loss formülasyonu.

---

## 5. Eşik Optimizasyonu ve Karar Mekanizması

### 5.1 Sınıf Bazlı Eşik Optimizasyonu (Threshold Optimization - Takım 2, 5)
Derin öğrenme modellerinin sigmoid çıkışı geleneksel olarak `0.5` kabul edilir. Ancak dengesiz veri setlerinde bu F1 skorunu mahveder.
- Eğitim sonrası Validation set üzerinde **Grid-Search** yapılarak (0.1'den 0.9'a kadar adım adım) 5 sınıfın her biri için Macro F1'i maksimize eden en uygun eşik değerleri (örneğin AFIB: 0.3, RBBB: 0.6) hesaplanır.
- Test aşamasında bu optimize edilmiş eşikler kullanılır.

### 5.2 Ensemble Stratejisi
3 farklı random seed ile eğitilen modellerin Logit/Softmax çıktıları toplanarak (Average Ensembling) nihai eşik değerlerinden geçirilir.

---

## 6. Mevcut Kodun Adapte Edilmesi

| Mevcut Dosya / Altyapı | Yapılacak Değişiklik |
|---|---|
| `config.py` | NUM_CLASSES=5, SNOMED_TO_LABEL harmonizasyonu güncellenecek. |
| `adim02_filtreleme.py` | 250Hz resample korunacak. **ArcTan normalizasyonu** eklenecek. |
| `EKGAugmentation` sınıfı | `np.roll` YASAKLANACAK. Baseline wander, lead dropout, scale eklenecek. |
| `adim07_model_mimarisi.py` | YENİDEN YAZILACAK: SE-ResNet + Transformer + Wide Features + DANN eklenecek. |
| **YENİ:** `threshold_opt.py` | Grid-search eşik optimizasyonu modülü eklenecek. |
| **YENİ:** `features.py` | NeuroKit2 ile Yaş/Cinsiyet/RR/HR "Wide" özellikleri çıkarılacak. |

---

## 7. Sekiz Haftalık Yol Haritası

1. **Hafta 1-2 (Veri Hazırlığı):** 250Hz Downsample, ArcTan Normalizasyonu, Agresif Augmentation modüllerinin kodlanması.
2. **Hafta 3-4 (Mimari İnşası):** SE-ResNet Backbone + Wide Features + Transformer Attention (CardioFusion-5) inşası ve testleri.
3. **Hafta 5 (Çoklu Veri Seti Eğitimi):** DANN Modülü entegrasyonu, İnternet veri setlerinin hazırlığı ve Curriculum Learning uygulanması.
4. **Hafta 6 (Optimizasyon):** Sınıf Dengesizliği Çözümleri (Threshold Optimization, Weighted/Sign Loss), Hiperparametre araması.
5. **Hafta 7-8 (Ensemble ve Teslimat):** Farklı seed'lere ait modellerin birleşimi, Docker container, Inference Script'i ve Teknik Raporun yazımı.

---

## 8. Riskler ve Önlemler

| Risk | Etki | Önlem | Öncelik |
|---|---|---|:---:|
| **Domain Overfit** | İnternet verisine aşırı uyum, TEKNOFEST'te çöküş | DANN Modülü + Aşama 3 Fine-tuning | 🔴 Yüksek |
| **AFL Dengesizliği** | Az veri F1'i ezer | Class-Balanced Sampler + Sınıfa Özel Eşik | 🔴 Yüksek |
| **Tasarım Karmaşası** | Wide & Deep, Transformer derken eğitimin çökmesi | PyTorch ile Dummy Forward Pass testleri, basitten karmaşığa gidiş | 🟡 Orta |

---

## 9. Özet: Dört Altın Kural

> 1. **12 derivasyon · Lead-wise Z-Score · ArcTan Normalizasyonu.**
> 2. **DANN Modülü Olmadan İnternet Verisi Kullanılmaz.**
> 3. **0.5 Sabit Eşik Yasaktır.** Validation'da Grid-Search yapılarak F1 maksimize eden eşik bulunur.
> 4. **Görülmemiş Veri Mindset'i:** Hedef accuracy değil, cihaz/hastane bağımsız Macro F1 skorudur.

---
*Doküman Versiyonu: 3.0 · Son Güncelleme: Mayıs 2026 · BirunAI / CardioFusion-5 Ekibi*
