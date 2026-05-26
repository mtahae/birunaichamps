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

# PTB-XL veri seti dizini (eski — tek dataset pipeline icin)
PTBXL_ROOT = os.path.join(
    PROJECT_ROOT,
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)

# =============================================================================
# MULTI-DATASET YAPILANDIRMASI
# =============================================================================

DATASETS_ROOT = os.path.join(PROJECT_ROOT, "datasets")

# Challenge 2020 kok dizini
_CH2020_ROOT = os.path.join(
    DATASETS_ROOT,
    "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2",
    "classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2",
    "training"
)

# ECG Arrhythmia kok dizini
_ECGARR_ROOT = os.path.join(
    DATASETS_ROOT,
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0",
    "WFDBRecords"
)

# Kullanilacak veri setleri — INCART (257 Hz) ve PTB Diagnostic (1000 Hz) HARIC
DATASET_PATHS = {
    "cpsc_2018":      os.path.join(_CH2020_ROOT, "cpsc_2018"),
    "cpsc_2018_extra": os.path.join(_CH2020_ROOT, "cpsc_2018_extra"),
    "ptb_xl":         os.path.join(_CH2020_ROOT, "ptb-xl"),
    "georgia":        os.path.join(_CH2020_ROOT, "georgia"),
    "ecg_arrhythmia":  _ECGARR_ROOT,
}

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
# SNOMED-CT -> 3-SINIF ESLEME TABLOSU
# =============================================================================
# Challenge 2020 ve ECG Arrhythmia veri setleri icin.
# .hea dosyalarindaki #Dx: satirindan okunan SNOMED-CT kodlari.

SNOMED_TO_LABEL = {
    # === SINIF 0: NORMAL ===
    426783006: 0,   # Sinus rhythm (NSR)
    427084000: 0,   # Sinus tachycardia (STach)
    426177001: 0,   # Sinus bradycardia (SB)
    427393009: 0,   # Sinus arrhythmia (SA)

    # === SINIF 1: RITIM BOZUKLUKLARI ===
    164889003: 1,   # Atrial fibrillation (AF)
    164890007: 1,   # Atrial flutter (AFL)
    195080001: 1,   # AF and flutter (AFAFL)
    426749004: 1,   # Chronic AF (CAF)
    282825002: 1,   # Paroxysmal AF (PAF)
    314208002: 1,   # Rapid AF (RAF)
    713422000: 1,   # Atrial tachycardia (ATach)
    233896004: 1,   # AV nodal reentrant tachycardia (AVNRT)
    233897008: 1,   # AV reentrant tachycardia (AVRT)
    67198005: 1,    # Paroxysmal SVT (PSVT)
    426761007: 1,   # Supraventricular tachycardia (SVT)
    63593006: 1,    # Supraventricular premature beats (SVPB)
    251168009: 1,   # Supraventricular bigeminy (SVB)
    284470004: 1,   # Premature atrial contraction (PAC)
    251170000: 1,   # Blocked PAC (BPAC)
    427172004: 1,   # Premature ventricular contraction (PVC)
    17338001: 1,    # Ventricular premature beats (VPB)
    251182009: 1,   # Paired ventricular premature complexes (VPVC)
    164895002: 1,   # Ventricular tachycardia (VTach)
    425856008: 1,   # Paroxysmal VTach (PVT)
    164896001: 1,   # Ventricular fibrillation (VF)
    111288001: 1,   # Ventricular flutter (VFL)
    251173003: 1,   # Atrial bigeminy (AB)
    11157007: 1,    # Ventricular bigeminy (VBig)
    251180001: 1,   # Ventricular trigeminy (VTrig)
    164884008: 1,   # Ventricular ectopics (VEB)
    75532003: 1,    # Ventricular escape beat (VEsB)
    81898007: 1,    # Ventricular escape rhythm (VEsR)
    49260003: 1,    # Idioventricular rhythm (IR)
    13640000: 1,    # Fusion beats (FB)
    10370003: 1,    # Pacing rhythm (PR)
    251268003: 1,   # Atrial pacing pattern (AP)
    251266004: 1,   # Ventricular pacing pattern (VPP)
    698247007: 1,   # Cardiac dysrhythmia (CD)
    74615001: 1,    # Brady-tachy syndrome (BTS)
    426627000: 1,   # Bradycardia (Brady) — patolojik bradikardi
    426664006: 1,   # Accelerated junctional rhythm (AJR)
    29320008: 1,    # AV junctional rhythm (AVJR)
    426648003: 1,   # Junctional tachycardia (JTach)
    426995002: 1,   # Junctional escape (JE)
    251164006: 1,   # Junctional premature complex (JPC)
    65778007: 1,    # Sinoatrial block (SAB)
    60423000: 1,    # Sinus node dysfunction (SND)
    195101003: 1,   # Wandering atrial pacemaker (WAP)

    # === SINIF 2: ILETIM BOZUKLUKLARI + MI + STTC + HYP ===
    # -- AV bloklar --
    270492004: 2,   # 1st degree AV block (IAVB)
    195042002: 2,   # 2nd degree AV block (IIAVB)
    54016002: 2,    # Mobitz type I (MoI)
    28189009: 2,    # 2nd degree AV block type II
    27885002: 2,    # Complete heart block (CHB)
    233917008: 2,   # AV block (AVB)
    204384007: 2,   # Congenital incomplete AV heart block (CIAHB)

    # -- Dal bloklari --
    713427006: 2,   # Complete RBBB (CRBBB)
    59118001: 2,    # Right bundle branch block (RBBB)
    164909002: 2,   # Left bundle branch block (LBBB)
    713426002: 2,   # Incomplete RBBB (IRBBB)
    251120003: 2,   # Incomplete LBBB (ILBBB)
    6374002: 2,     # Bundle branch block (BBB)
    445118002: 2,   # Left anterior fascicular block (LAnFB)
    445211001: 2,   # Left posterior fascicular block (LPFB)
    82226007: 2,    # Diffuse intraventricular block (DIB)
    698252002: 2,   # Nonspecific IVCD (NSIVCB)
    74390002: 2,    # WPW pattern (WPW)
    195060002: 2,   # Ventricular pre-excitation (VPEx)
    164947007: 2,   # Prolonged PR interval (LPR)
    49578007: 2,    # Shortened PR interval (SPRI)

    # -- MI --
    164865005: 2,   # Myocardial infarction (MI)
    57054005: 2,    # Acute MI (AMI)
    54329005: 2,    # Anterior MI (AnMI)
    164867002: 2,   # Old MI (OldMI)

    # -- Iskemi --
    413444003: 2,   # Acute myocardial ischemia (AMIs)
    164861001: 2,   # Myocardial ischemia (MIs)
    413844008: 2,   # Chronic myocardial ischemia (CMI)
    426434006: 2,   # Anterior ischemia (AnMIs)
    425419005: 2,   # Inferior ischaemia (IIs)
    425623009: 2,   # Lateral ischaemia (LIs)

    # -- ST/T degisiklikleri --
    429622005: 2,   # ST depression (STD)
    164931005: 2,   # ST elevation (STE)
    164930006: 2,   # ST interval abnormal (STIAb)
    704997005: 2,   # Inferior ST segment depression (ISTD)
    55930002: 2,    # ST changes (STC)
    428750005: 2,   # Nonspecific ST-T abnormality (NSSTTA)
    164934002: 2,   # T wave abnormal (TAb)
    59931005: 2,    # T wave inversion (TInv)
    251259000: 2,   # High T voltage (HTV)
    164912004: 2,   # P wave change (PWC)

    # -- Hipertrofi --
    164873001: 2,   # Left ventricular hypertrophy (LVH)
    89792004: 2,    # Right ventricular hypertrophy (RVH)
    266249003: 2,   # Ventricular hypertrophy (VH)
    370365005: 2,   # Left ventricular strain (LVS)
    253352002: 2,   # Left atrial abnormality (LAA)
    67741000119109: 2, # Left atrial enlargement (LAE)
    446813000: 2,   # Left atrial hypertrophy (LAH)
    195126007: 2,   # Atrial hypertrophy (AH)
    253339007: 2,   # Right atrial abnormality (RAAb)
    446358003: 2,   # Right atrial hypertrophy (RAH)

    # -- Aks/QRS --
    39732003: 2,    # Left axis deviation (LAD)
    47665007: 2,    # Right axis deviation (RAD)
    251200008: 2,   # Indeterminate cardiac axis (ICA)
    164951009: 2,   # Abnormal QRS (abQRS)
    164942001: 2,   # fQRS wave
    251146004: 2,   # Low QRS voltages (LQRSV)
    251148003: 2,   # Low QRS voltages in chest leads
    251147008: 2,   # Low QRS voltages in limb leads
    164921003: 2,   # R wave abnormal (RAb)
    164917005: 2,   # Q wave abnormal (QAb)
    111975006: 2,   # Prolonged QT (LQT)
    77867006: 2,    # Decreased QT interval (SQT)

    # -- Diger --
    53741008: 2,    # Coronary heart disease (CHD)
    84114007: 2,    # Heart failure (HF)
    368009: 2,      # Heart valve disorder (HVD)
    428417006: 2,   # Early repolarization (ERe)
    164937009: 2,   # U wave abnormal (UAb)
    266257000: 2,   # Transient ischemic attack (TIA)
    251139008: 2,   # Suspect arm leads reversed (ALR)
    251199005: 2,   # Counterclockwise rotation (CCR)
    251198002: 2,   # Clockwise rotation (CR)
}

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
# EGITIM PARAMETRELERI (v5 — Multi-Dataset, 87K+ kayit)
# =============================================================================

BATCH_SIZE = 64                # 87K+ veri icin buyuk batch — GPU'da rahat sigar
LEARNING_RATE = 1e-4           # Buyuk dataset icin daha dusuk LR — overfitting onleme
WEIGHT_DECAY = 1e-3            # Guclu regularizasyon — 5 farkli domain
NUM_EPOCHS = 60                # Buyuk veri seti daha az epoch gerektirir
EARLY_STOPPING_PATIENCE = 15   # Buyuk veri ile daha kararlı ogrenme

# Warmup
WARMUP_EPOCHS = 3              # Buyuk batch ile kisa warmup yeterli

# Focal Loss
FOCAL_LOSS_GAMMA = 1.5         # SMOTE sonrasi hala hafif dengesizlik olabilir
LABEL_SMOOTHING = 0.1          # Multi-domain icin biraz daha yuksek smoothing

# Gradient Clipping
GRAD_CLIP_MAX_NORM = 1.0

# Mixed Precision
USE_AMP = True

# =============================================================================
# TEKRARLANABILIRLIK (REPRODUCIBILITY)
# =============================================================================

SEED = 42
