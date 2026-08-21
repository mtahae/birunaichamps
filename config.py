"""
config.py — BirunAI EKG Siniflandirma: Merkezi Konfigurasyon
==============================================================

Tum proje parametreleri bu dosyada tanimlanir.
2. Asama: 5 Sinif (Normal, AFIB, AFL, LBBB, RBBB) ve Unified Architecture

KURAL: Her parametre burada tanimlanir, baska yere hardcode yapilmaz.
"""

import os
import random
import numpy as np
import torch

# =============================================================================
# REPRODUCIBILITY (TEKRAR URETILEBILIRLIK) — PDF BOLUM 9
# =============================================================================
SEED = 42

def set_seed(seed=SEED):
    """Tam deterministic egitim icin tum seed'leri sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# PROJE DIZIN YAPISI
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

# Teknofest 2. Asama Veri Seti
TEKNOFEST_DIR = os.path.join(DATASETS_ROOT, "lise-veri-seti", "teknofest_ECG_2026_dataset")

# Challenge 2020 kok dizini (Internet Verileri)
_CH2020_ROOT = os.path.join(
    DATASETS_ROOT,
    "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2",
    "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2",
    "training"
)
_ECGARR_ROOT = os.path.join(
    DATASETS_ROOT,
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
    "WFDBRecords"
)

DATASET_PATHS = {
    "cpsc_2018":            os.path.join(_CH2020_ROOT, "cpsc_2018"),
    "cpsc_2018_extra":      os.path.join(_CH2020_ROOT, "cpsc_2018_extra"),
    "ptb_xl":               os.path.join(_CH2020_ROOT, "ptb-xl"),
    "georgia":              os.path.join(_CH2020_ROOT, "georgia"),
    "st_petersburg_incart": os.path.join(_CH2020_ROOT, "st_petersburg_incart"),
    "ecg_arrhythmia":       _ECGARR_ROOT,
}

# Records Dataset (H5 formati, AHA kodlari)
RECORDS_DATASET_DIR = os.path.join(DATASETS_ROOT, "records", "records")
RECORDS_METADATA_CSV = os.path.join(RECORDS_DATASET_DIR, "metadata (1).csv")

# Cikti dizinleri
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

for d in [PROCESSED_DATA_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# SINYAL ISLEME PARAMETRELERI — PDF BOLUM 4 (Filtreleme)
# =============================================================================
ORIGINAL_FS = 500        # Orijinal ornekleme frekansi (TEKNOFEST=500Hz)
TARGET_FS = 250          # Hedef ornekleme frekansi (Hz)
WINDOW_SEC = 10          # Pencere suresi (saniye)
TARGET_LENGTH = TARGET_FS * WINDOW_SEC  # 2500 ornek
NUM_LEADS = 12           # 12 derivasyonlu EKG

# Butterworth bandpass filtre — PDF: 0.5-40Hz, 4. derece, filtfilt
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0
BANDPASS_ORDER = 4

# SQI (Sinyal Kalitesi) — PDF BOLUM 7
SQI_THRESHOLD = 0.2      # Bu deger altindaki kayitlar egitimden cikarilir

# =============================================================================
# SINIFLANDIRMA PARAMETRELERI (2. ASAMA - 5 SINIF)
# =============================================================================
NUM_CLASSES = 5
NUM_AUX_CLASSES = 3  # Multi-Task Loss icin (Normal / Ritim / Iletim)

LABEL_NAMES = {
    0: "Normal",
    1: "AFIB",
    2: "AFL",
    3: "LBBB",
    4: "RBBB"
}

AUX_LABEL_NAMES = {
    0: "Normal",
    1: "Ritim Bozuklugu",
    2: "Iletim Bozuklugu"
}

# 5-siniftan 3-sinifa esleme (Multi-Task Aux Head icin)
MAIN_TO_AUX_LABEL = {
    0: 0,  # Normal -> Normal
    1: 1,  # AFIB -> Ritim
    2: 1,  # AFL -> Ritim
    3: 2,  # LBBB -> Iletim
    4: 2,  # RBBB -> Iletim
}

# Cakisma onceligi (eger birden fazla etiket varsa, en yuksek oncelikli alinir)
LABEL_PRIORITY = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

# Sadece hedef 5 sinifa karsilik gelen kesin SNOMED-CT kodlari.
SNOMED_TO_LABEL = {
    # Normal Sinus Ritmi
    426783006: 0,
    427084000: 0,
    426177001: 0,
    427393009: 0,

    # AFIB
    164889003: 1,
    426749004: 1,
    282825002: 1,
    314208002: 1,

    # AFL
    164890007: 2,
    195080001: 2,

    # LBBB
    164909002: 3,
    251120003: 3,

    # RBBB
    59118001: 4,
    713427006: 4,
    713426002: 4,
}

# AHA kod esleme tablosu (records dataseti icin)
# AHA diagnostic kodlari -> 5 sinif
AHA_TO_LABEL = {
    # Normal Sinus Rhythm
    1: 0,

    # AFIB (Atrial Fibrillation)
    22: 1,

    # AFL (Atrial Flutter)
    23: 2,

    # LBBB (Left Bundle Branch Block)
    50: 3,   # Complete LBBB
    51: 3,   # Incomplete LBBB

    # RBBB (Right Bundle Branch Block)
    60: 4,   # Complete RBBB
}

# =============================================================================
# MODEL MIMARISI PARAMETRELERI (CardioFusion-5)
# =============================================================================
# SE-ResNet
CNN_FILTERS = [64, 128, 256, 256, 384]
CNN_KERNEL_SIZE_1 = 15  # Ilk katman buyuk kernel
CNN_KERNEL_SIZE_2 = 7   # Sonraki katmanlar

# Transformer Encoder
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
TRANSFORMER_DIM = 384
TRANSFORMER_DROPOUT = 0.2

# Wide Features
WIDE_FEATURE_DIM = 12  # 8 + 4 P-dalga ozelligi (rr_cv, p_present, p_regularity, atrial_rate)

# =============================================================================
# EGITIM PARAMETRELERI — PDF BOLUM 3.3 (Curriculum Learning)
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.2e-3  # Overfitting onleme: 5e-4 -> 1.2e-3 (train/val makasi 0.11'di)
NUM_EPOCHS = 100

# Curriculum Learning Asamalari
# REVIZE 2 (2026-07-19): Iki ayri run'da (seed42, seed123) P2'nin val-loss/F1 zirvesi
# HER IKISINDE de P2'nin ilk ~9 epoch'unda geliyor, sonrasi TUTARLI sekilde overfit
# (val loss 0.225->0.440'a katlaniyor). P2=30+patience=12 bu zirveyi epoch 21'de
# yakaliyor ama epoch 10-21 arasi (11 epoch, ~%35 P2 suresi) tamamen bosa gidiyor.
# P2 kisaltildi (30->18) + PATIENCE_P2 dusuruldu (bkz adim08_egitim.py) — sonuc
# degismez (checkpoint zaten zirveyi kurtariyordu), sadece wall-clock kazanilir.
EPOCHS_PHASE_1 = 25   # Sadece Teknofest (uretken faz — biraz daha alan)
EPOCHS_PHASE_2 = 18   # Teknofest + Internet (DANN aktif) — zirve ~epoch9'da, guvenli marj
EPOCHS_PHASE_3 = 25   # Fine-tune (dusuk LR, TEKNOFEST) — erken ve yeterli ince ayar

# DANN — domain overfit'e karsi guclendirildi (md: 0.1 -> 0.3)
DANN_LAMBDA = 0.3

# GURULTULU-ETIKET-FARKINDA AGIRLIKLANDIRMA (AFIB/AFL darbogazi icin)
# Kanit: 4 mimari iterasyonda AFIB/AFL kimildamadi -> sorun VERI/ETIKET.
# Internet ritim etiketleri (AFIB/AFL) gurultulu ve TEKNOFEST ile celisiyor.
# P2'de internet (domain>0) AFIB/AFL ornekleri loss'a bu kat katki yapar (<1.0),
# boylece ritim ogrenimi TEMIZ TEKNOFEST etiketlerinden baskin sekilde gelir.
# Morfoloji siniflari (Normal/LBBB/RBBB) internet verisinin tamamini kullanir.
INTERNET_RHYTHM_WEIGHT = 0.4

# Loss Hyperparametreleri — PDF BOLUM 6
FOCAL_LOSS_GAMMA = 2.0
FOCAL_GAMMA = FOCAL_LOSS_GAMMA  # Alias (eski kod uyumlulugu)
LABEL_SMOOTHING = 0.03  # 0.05'ten dusuruldu — AFL/AFIB icin daha keskin kararlar
AUX_LOSS_WEIGHT = 0.3  # Multi-Task: L = L_main + 0.3 * L_aux

# Egitim Kontrolleri
EARLY_STOPPING_PATIENCE = 15
WARMUP_EPOCHS = 5
GRAD_CLIP_MAX_NORM = 1.0
USE_AMP = True  # Mixed Precision Training


# =============================================================================
# DEVICE
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
