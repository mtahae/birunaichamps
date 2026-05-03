"""
config.py — BirunAI EKG Siniflandirma: Merkezi Konfigürasyon
==============================================================

Tum proje parametreleri bu dosyada tanimlanir.
Herhangi bir parametreyi degistirmek icin SADECE bu dosyayi duzenleyin.
"""

import os

# =============================================================================
# PROJE DIZIN YAPISI
# =============================================================================

# Proje kok dizini (bu dosyanin bulundugu yer)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# PTB-XL veri seti dizini
PTBXL_ROOT = os.path.join(
    PROJECT_ROOT,
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# Cikti dizinleri
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, "processed_data")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# Dizinleri olustur
for d in [PROCESSED_DATA_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# SINYAL ISLEME PARAMETRELERI
# =============================================================================

# Ornekleme frekanslari
ORIGINAL_FS = 500       # PTB-XL orijinal ornekleme frekansi (Hz)
TARGET_FS = 250          # Hedef ornekleme frekansi (Hz)

# Pencere parametreleri
WINDOW_SEC = 10          # Pencere suresi (saniye) — klinik EKG standardi
TARGET_LENGTH = TARGET_FS * WINDOW_SEC  # 2500 ornek

# EKG parametreleri
NUM_LEADS = 12           # 12 derivasyonlu EKG

# Butterworth bandpass filtre
BANDPASS_LOW = 0.5       # Alt kesim frekansi (Hz) — taban cizgisi kaymasini eler
BANDPASS_HIGH = 40.0     # Ust kesim frekansi (Hz) — EMG + sebeke gurultusunu eler
BANDPASS_ORDER = 4       # Filtre derecesi

# =============================================================================
# SINIFLANDIRMA PARAMETRELERI
# =============================================================================

NUM_CLASSES = 3

LABEL_NAMES = {
    0: "Normal",
    1: "Ritim Bozuklugu",
    2: "Iletim Bozuklugu"
}

# SCP kodu -> Sinif eslestirme tablosu
# Kaynak: scp_statements.csv'deki diagnostic_class ve rhythm sutunlari
#
# Eslestirme mantigi:
#   - NORM -> 0 (Normal)
#   - rhythm=1.0 olan kodlar -> 1 (Ritim Bozuklugu)
#     (AFIB, AFLT, SVTAC, SVARR, PSVT, BIGU, TRIGU, SR, STACH, SBRAD, SARRH, PACE)
#     NOT: SR, STACH, SBRAD, SARRH sinus varyantlaridir ama rhythm grubundadir.
#          Bunlari Normal olarak siniflandiriyoruz (klinik olarak normal varyant).
#   - CD (Conduction Disturbance) -> 2 (Iletim Bozuklugu)
#   - MI (Myocardial Infarction) -> 2 (Iletim Bozuklugu)
#   - STTC (ST/T Changes) -> 2 (Iletim Bozuklugu)
#   - HYP (Hypertrophy) -> 2 (Iletim Bozuklugu)
#
# Cakisma onceligi: Iletim (2) > Ritim (1) > Normal (0)

SCP_TO_LABEL = {
    # === SINIF 0: NORMAL ===
    "NORM": 0,      # Normal ECG
    "SR": 0,        # Sinus ritmi (normal)
    "SBRAD": 0,     # Sinus bradikardisi (normal varyant)
    "STACH": 0,     # Sinus tasikardisi (normal varyant)
    "SARRH": 0,     # Sinus aritmisi (normal varyant)

    # === SINIF 1: RITIM BOZUKLUKLARI ===
    "AFIB": 1,      # Atriyal fibrilasyon
    "AFLT": 1,      # Atriyal flutter
    "SVTAC": 1,     # Supraventrikuler tasikardi
    "PSVT": 1,      # Paroksismal SVT
    "SVARR": 1,     # Supraventrikuler aritmi
    "PVC": 1,       # Ventrikuler prematüre kompleks
    "PAC": 1,       # Atriyal prematüre kompleks
    "BIGU": 1,      # Bigeminal patern
    "TRIGU": 1,     # Trigeminal patern
    "PACE": 1,      # Yapay pacemaker ritmi
    "PRC(S)": 1,    # Prematüre kompleksler

    # === SINIF 2: ILETIM BOZUKLUKLARI (CD + MI + STTC + HYP) ===
    # -- Iletim bozukluklari (CD) --
    "1AVB": 2,      # Birinci derece AV blok
    "2AVB": 2,      # Ikinci derece AV blok
    "3AVB": 2,      # Ucuncu derece AV blok
    "CRBBB": 2,     # Komplet sag dal blogu
    "IRBBB": 2,     # Inkomplet sag dal blogu
    "CLBBB": 2,     # Komplet sol dal blogu
    "ILBBB": 2,     # Inkomplet sol dal blogu
    "LAFB": 2,      # Sol anterior fasikul blogu
    "LPFB": 2,      # Sol posterior fasikul blogu
    "WPW": 2,       # Wolf-Parkinson-White sendromu
    "IVCD": 2,      # Spesifik olmayan intraventrikuler iletim bozuklugu
    "LPR": 2,       # Uzamis PR intervali

    # -- Miyokard enfarktusu (MI) --
    "IMI": 2,       # Inferior MI
    "AMI": 2,       # Anterior MI
    "LMI": 2,       # Lateral MI
    "PMI": 2,       # Posterior MI
    "ASMI": 2,      # Anteroseptal MI
    "ILMI": 2,      # Inferolateral MI
    "IPMI": 2,      # Inferoposterior MI
    "ALMI": 2,      # Anterolateral MI
    "IPLMI": 2,     # Inferoposterolateral MI

    # -- Subendokardiyal hasar (MI alt grubu) --
    "INJAS": 2,     # Anteroseptal subendokardiyal hasar
    "INJAL": 2,     # Anterolateral subendokardiyal hasar
    "INJIN": 2,     # Inferior subendokardiyal hasar
    "INJLA": 2,     # Lateral subendokardiyal hasar
    "INJIL": 2,     # Inferolateral subendokardiyal hasar

    # -- ST/T degisiklikleri (STTC) --
    "NST_": 2,      # Nonspesifik ST degisiklikleri
    "ISC_": 2,      # Nonspesifik iskemik
    "ISCA": 2,      # Iskemik (anterolateral)
    "ISCAL": 2,     # Iskemik anterolateral derivasyonlarda
    "ISCAN": 2,     # Iskemik anterior derivasyonlarda
    "ISCAS": 2,     # Iskemik anteroseptal derivasyonlarda
    "ISCIL": 2,     # Iskemik inferolateral derivasyonlarda
    "ISCIN": 2,     # Iskemik inferior derivasyonlarda
    "ISCLA": 2,     # Iskemik lateral derivasyonlarda
    "NDT": 2,       # Nondiagnostik T anormallikleri
    "DIG": 2,       # Dijitalis etkisi
    "LNGQT": 2,     # Uzun QT intervali
    "ANEUR": 2,     # Ventrikuler anevrizma ile uyumlu ST-T degisiklikleri
    "EL": 2,        # Elektrolit bozuklugu

    # -- Form ozellikleri (diagnostic_class yok ama patolojik) --
    "STD_": 2,      # Nonspesifik ST depresyonu
    "STE_": 2,      # Nonspesifik ST elevasyonu
    "INVT": 2,      # Invert T dalgalari
    "QWAVE": 2,     # Q dalgalari mevcut
    "NT_": 2,       # Nonspesifik T-dalga degisiklikleri
    "TAB_": 2,      # T-dalga anormalligi
    "LOWT": 2,      # Dusuk amplitudlu T dalgalari
    "ABQRS": 2,     # Anormal QRS

    # -- Hipertrofi (HYP) --
    "LVH": 2,       # Sol ventrikul hipertrofisi
    "RVH": 2,       # Sag ventrikul hipertrofisi
    "SEHYP": 2,     # Septal hipertrofi
    "LAO/LAE": 2,   # Sol atriyal buyume
    "RAO/RAE": 2,   # Sag atriyal buyume
    "VCLVH": 2,     # Voltaj kriterleri sol ventrikul hipertrofisi
    "LVOLT": 2,     # Dusuk QRS voltajlari
    "HVOLT": 2,     # Yuksek QRS voltaji
}

# Cakisma onceligi: Iletim (2) > Ritim (1) > Normal (0)
LABEL_PRIORITY = {2: 3, 1: 2, 0: 1}

# =============================================================================
# MODEL HIPERPARAMETRELERI
# =============================================================================

# CNN
CNN_FILTERS = [64, 128, 256]
CNN_KERNEL_SIZE = 7
CNN_DROPOUT = 0.3

# BiLSTM
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

# Fully Connected
FC_HIDDEN_SIZE = 128
FC_DROPOUT = 0.5

# =============================================================================
# EGITIM PARAMETRELERI
# =============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 80                # Augmentasyon + SWA icin yeterli sure
EARLY_STOPPING_PATIENCE = 15  # Val F1 15 epoch iyilesmezse dur

# Warmup
WARMUP_EPOCHS = 5             # Ilk 5 epoch LR lineer olarak artar

# Focal Loss
FOCAL_LOSS_GAMMA = 2.0
LABEL_SMOOTHING = 0.1         # Overconfidence onleme — soft labels

# Gradient Clipping
GRAD_CLIP_MAX_NORM = 1.0

# Mixed Precision
USE_AMP = True

# =============================================================================
# TEKRARLANABILIRLIK (REPRODUCIBILITY)
# =============================================================================

SEED = 42
