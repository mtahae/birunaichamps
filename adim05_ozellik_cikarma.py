"""
adim05_ozellik_cikarma.py — BirunAI EKG Siniflandirma: Adim 5 – Ozellik Cikarma
==================================================================================

Bu modul, 1D-CNN + BiLSTM modelimiz icin ek el-yapimi (handcrafted) ozellikler cikarir.
Ana modelin CNN katmanlari otomatik ozellik ogrenir, ancak ek ozellikler
istegebagli olarak model performansini artirabilir.

NOT: Projemizde birincil yaklasim "end-to-end" ogrenimdir.
     Ham sinyal dogrudan modele verilir, CNN kendi ozelliklerini ogenir.
     Bu modul, EDA (Exploratory Data Analysis) ve potansiyel
     feature-augmented model denemeleri icindir.

Cikarilan Ozellikler:
    - Istatistiksel: min, max, mean, std, skewness, kurtosis (her derivasyon)
    - Morfolojik: R-pike algilama (basit esik), RR aralik istatistikleri
    - Frekans alani: Baskin frekans, guc yogunlugu bantlari

Ciktilar:
    - outputs/processed_data/features.csv
    - outputs/processed_data/feature_stats.csv

Kullanim:
    python adim05_ozellik_cikarma.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import find_peaks, welch
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# OZELLIK CIKARMA FONKSIYONLARI
# =============================================================================

def istatistiksel_ozellikler(sinyal_2d):
    """
    Her derivasyondan temel istatistiksel ozellikler cikarir.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array

    Returns:
        dict: {ozellik_adi: deger} formati
    """
    ozellikler = {}
    for ch in range(sinyal_2d.shape[0]):
        kanal = sinyal_2d[ch]
        prefix = f"ch{ch}"
        ozellikler[f"{prefix}_mean"] = np.mean(kanal)
        ozellikler[f"{prefix}_std"] = np.std(kanal)
        ozellikler[f"{prefix}_min"] = np.min(kanal)
        ozellikler[f"{prefix}_max"] = np.max(kanal)
        ozellikler[f"{prefix}_skew"] = float(sp_stats.skew(kanal))
        ozellikler[f"{prefix}_kurt"] = float(sp_stats.kurtosis(kanal))
        ozellikler[f"{prefix}_ptp"] = np.ptp(kanal)  # peak-to-peak
        ozellikler[f"{prefix}_rms"] = np.sqrt(np.mean(kanal ** 2))
        # Sifir gecis orani (zero-crossing rate)
        sign_changes = np.diff(np.sign(kanal))
        ozellikler[f"{prefix}_zcr"] = np.sum(sign_changes != 0) / len(kanal)
    return ozellikler


def morfolojik_ozellikler(sinyal_2d, fs=None):
    """
    Lead II (indeks 1) uzerinden R-pike tespiti ve RR araliklarini hesaplar.

    Lead II secilme nedeni: Klinik EKG'de P dalgasi ve QRS kompleksi
    Lead II'de en belirgin gorulur.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        fs: Ornekleme frekansi. Varsayilan: config.TARGET_FS

    Returns:
        dict: RR-araligi istatistikleri
    """
    if fs is None:
        fs = config.TARGET_FS

    ozellikler = {}

    # Lead II (indeks 1) kullan, yoksa Lead I (indeks 0)
    lead_idx = 1 if sinyal_2d.shape[0] > 1 else 0
    lead = sinyal_2d[lead_idx]

    # R-pike tespiti (basit esik tabanli)
    # Minimum mesafe: 0.3sn (200 bpm ustu olmaz), minimum yukseklik: 0.5 std
    min_mesafe = int(0.3 * fs)
    min_yukseklik = 0.5 * np.std(lead)

    peaks, properties = find_peaks(lead, distance=min_mesafe, height=min_yukseklik)

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs  # saniye cinsinden
        ozellikler["rr_mean"] = np.mean(rr_intervals)
        ozellikler["rr_std"] = np.std(rr_intervals)
        ozellikler["rr_min"] = np.min(rr_intervals)
        ozellikler["rr_max"] = np.max(rr_intervals)
        ozellikler["rr_range"] = np.ptp(rr_intervals)
        ozellikler["heart_rate_bpm"] = 60.0 / np.mean(rr_intervals)
        ozellikler["num_peaks"] = len(peaks)
        # HRV (Heart Rate Variability) - basit olcum
        ozellikler["rmssd"] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    else:
        ozellikler["rr_mean"] = 0
        ozellikler["rr_std"] = 0
        ozellikler["rr_min"] = 0
        ozellikler["rr_max"] = 0
        ozellikler["rr_range"] = 0
        ozellikler["heart_rate_bpm"] = 0
        ozellikler["num_peaks"] = len(peaks)
        ozellikler["rmssd"] = 0

    return ozellikler


def frekans_alani_ozellikleri(sinyal_2d, fs=None):
    """
    Frekans alaninda guc yogunlugu bantlarini hesaplar.

    Bantlar:
        - VLF (Very Low Frequency): 0.003 - 0.04 Hz
        - LF  (Low Frequency):      0.04 - 0.15 Hz
        - HF  (High Frequency):     0.15 - 0.4 Hz
        - QRS bandi:                 5 - 15 Hz
        - Toplam guc

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        fs: Ornekleme frekansi

    Returns:
        dict: Frekans alani ozellikleri
    """
    if fs is None:
        fs = config.TARGET_FS

    ozellikler = {}

    # Lead II uzerinden
    lead_idx = 1 if sinyal_2d.shape[0] > 1 else 0
    lead = sinyal_2d[lead_idx]

    # Welch PSD
    freqs, psd = welch(lead, fs=fs, nperseg=min(256, len(lead)))

    # Toplam guc
    ozellikler["total_power"] = np.sum(psd)

    # Bant gucleri
    def bant_gucu(f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        return np.sum(psd[mask]) if np.any(mask) else 0

    ozellikler["power_vlf"] = bant_gucu(0.003, 0.04)
    ozellikler["power_lf"] = bant_gucu(0.04, 0.15)
    ozellikler["power_hf"] = bant_gucu(0.15, 0.4)
    ozellikler["power_qrs"] = bant_gucu(5.0, 15.0)

    # Baskin frekans
    baskin_idx = np.argmax(psd)
    ozellikler["dominant_freq"] = freqs[baskin_idx]

    # LF/HF orani (otonom sinir sistemi gostergesi)
    if ozellikler["power_hf"] > 1e-10:
        ozellikler["lf_hf_ratio"] = ozellikler["power_lf"] / ozellikler["power_hf"]
    else:
        ozellikler["lf_hf_ratio"] = 0

    return ozellikler


# =============================================================================
# ANA OZELLIK CIKARMA PIPELINE'I
# =============================================================================

def ozellik_cikarma_pipeline():
    """
    Tum segmente edilmis sinyallerden ozellik cikarir.

    Returns:
        pd.DataFrame: Ozellik matrisi
    """
    print("=" * 70)
    print("BirunAI -- Adim 5: Ozellik Cikarma (Feature Engineering)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    print(f"\n[1/3] segmented_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"segmented_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim04_segmentasyon.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    segmente = df[df["segmented"] == True].copy()
    print(f"      Segmente kayit: {len(segmente)}")

    # --- 2. Ozellik cikarma ---
    print(f"\n[2/3] Ozellikler cikariliyor...")

    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    tum_ozellikler = []

    for ecg_id, satir in tqdm(segmente.iterrows(), total=len(segmente),
                               desc="      Ozellik Cikarma"):
        try:
            sinyal = np.load(os.path.join(sinyal_dizini, f"{ecg_id}.npy"))

            # 3 ozellik grubu
            oz = {"ecg_id": ecg_id, "label": satir["label"]}
            oz.update(istatistiksel_ozellikler(sinyal))
            oz.update(morfolojik_ozellikler(sinyal))
            oz.update(frekans_alani_ozellikleri(sinyal))

            tum_ozellikler.append(oz)

        except Exception as e:
            pass

    # DataFrame olustur
    ozellik_df = pd.DataFrame(tum_ozellikler)
    ozellik_df.set_index("ecg_id", inplace=True)

    print(f"\n      Cikarilan ozellik sayisi: {len(ozellik_df.columns) - 1}")
    print(f"      Kayit sayisi: {len(ozellik_df)}")

    # --- 3. Kaydet ---
    print(f"\n[3/3] Kaydediliyor...")

    # Tam ozellik matrisi
    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "features.csv")
    ozellik_df.to_csv(cikti_yolu)
    print(f"      features.csv: {cikti_yolu}")

    # Ozellik istatistikleri
    sayisal_sutunlar = ozellik_df.select_dtypes(include=[np.number]).columns.drop("label", errors="ignore")
    stats_df = ozellik_df[sayisal_sutunlar].describe().T
    stats_cikti = os.path.join(config.PROCESSED_DATA_DIR, "feature_stats.csv")
    stats_df.to_csv(stats_cikti)
    print(f"      feature_stats.csv: {stats_cikti}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)
    print(f"  Toplam kayit      : {len(ozellik_df)}")
    print(f"  Toplam ozellik    : {len(sayisal_sutunlar)}")
    print(f"  Ozellik gruplari  :")
    print(f"    - Istatistiksel : 12 kanal x 9 ozellik = 108")
    print(f"    - Morfolojik    : 8 ozellik (RR intervalleri, kalp hizi)")
    print(f"    - Frekans alani : 7 ozellik (guc bantlari, baskin frekans)")

    # Sinifa gore bazi ozellikler
    if "heart_rate_bpm" in ozellik_df.columns:
        print(f"\n  Sinifa Gore Ortalama Kalp Hizi (BPM):")
        for sinif_idx in sorted(ozellik_df["label"].unique()):
            sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "?")
            ortalama_hr = ozellik_df[ozellik_df["label"] == sinif_idx]["heart_rate_bpm"].mean()
            print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {ortalama_hr:.1f} BPM")

    print("\n" + "=" * 70)
    print("Adim 5 tamamlandi. Sonraki adim: adim06_veri_bolme.py")
    print("=" * 70)

    return ozellik_df


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = ozellik_cikarma_pipeline()
