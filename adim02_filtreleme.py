"""
adim02_filtreleme.py — BirunAI EKG Siniflandirma: Adim 2 – Filtreleme ve Alt Ornekleme
========================================================================================

Bu modul, ham EKG sinyallerine sinyal isleme adimlari uygular.
Multi-dataset pipeline: unified_manifest_clean.csv okur, tum setleri isler.

Islem Adimlari:
    1. Sinyal okunmasi (wfdb — .mat + .hea dosyalari)
    2. Alt ornekleme: 500 Hz -> 250 Hz
    3. Butterworth Bandpass Filtre: 0.5-40 Hz, order=4
    4. ArcTan Normalizasyonu (R-tepesi baskilama)
    (NOT: Z-score normalizasyonu adim06/DataLoader asamasinda train istatistikleriyle yapilacaktir.)

Ciktilar:
    - outputs/processed_data/filtered_signals/  (her kayit icin .npy)
    - outputs/processed_data/filtered_manifest.csv

Kullanim:
    python adim02_filtreleme.py
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# SINYAL ISLEME FONKSIYONLARI
# =============================================================================

def butterworth_bandpass_filtre_olustur(lowcut, highcut, fs, order=4):
    """Butterworth bandpass filtre katsayilarini olusturur."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def sinyal_filtrele(sinyal, fs, lowcut=None, highcut=None, order=None):
    """Tek bir EKG kanalina bandpass filtre uygular (filtfilt — sifir faz kaymasi)."""
    if lowcut is None:
        lowcut = config.BANDPASS_LOW
    if highcut is None:
        highcut = config.BANDPASS_HIGH
    if order is None:
        order = config.BANDPASS_ORDER

    b, a = butterworth_bandpass_filtre_olustur(lowcut, highcut, fs, order)
    padlen = min(3 * max(len(b), len(a)), len(sinyal) - 1)
    if padlen < 1:
        return sinyal
    filtrelenmis = filtfilt(b, a, sinyal, padlen=padlen)
    return filtrelenmis


def cok_kanalli_filtrele(sinyal_2d, fs):
    """12 derivasyonun her birine bagimsiz bandpass filtre uygular."""
    filtrelenmis = np.zeros_like(sinyal_2d)
    for kanal_idx in range(sinyal_2d.shape[0]):
        filtrelenmis[kanal_idx] = sinyal_filtrele(sinyal_2d[kanal_idx], fs)
    return filtrelenmis


def alt_ornekle(sinyal_2d, orijinal_fs, hedef_fs):
    """Sinyali hedef frekansa alt ornekler."""
    if orijinal_fs == hedef_fs:
        return sinyal_2d
    orijinal_uzunluk = sinyal_2d.shape[1]
    yeni_uzunluk = int(orijinal_uzunluk * hedef_fs / orijinal_fs)
    alt_orneklenmis = np.zeros((sinyal_2d.shape[0], yeni_uzunluk))
    for kanal_idx in range(sinyal_2d.shape[0]):
        alt_orneklenmis[kanal_idx] = resample(sinyal_2d[kanal_idx], yeni_uzunluk)
    return alt_orneklenmis


def arctan_normalize(sinyal_2d):
    """
    Sinyali arctan fonksiyonundan gecirir.
    Amac: Asiri yuksek voltajli R-tepelerini (R-peaks) baskilayarak modelin 
    P dalgasi gibi daha kucuk morfolojilere de odaklanabilmesini saglamak.
    (PhysioNet 2020 4. Takim - Triage yontemi)
    """
    return np.arctan(sinyal_2d)


# =============================================================================
# ANA FILTRELEME PIPELINE'I
# =============================================================================

def filtreleme_pipeline():
    """
    Tum kayitlara filtreleme, alt ornekleme ve normalizasyon uygular.
    Multi-dataset: unified_manifest_clean.csv okur.
    """
    print("=" * 70)
    print("BirunAI — Adim 2: Filtreleme ve Alt Ornekleme (Multi-Dataset)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "unified_manifest_clean.csv")
    print(f"\n[1/4] unified_manifest_clean.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"unified_manifest_clean.csv bulunamadi: {manifest_yolu}\n"
            "Once adim01_kalite_kontrol_genel.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu)
    print(f"      Toplam kayit: {len(df)}")

    # --- 2. Cikti dizinini olustur ---
    cikti_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    os.makedirs(cikti_dizini, exist_ok=True)

    # --- 3. Filtreleme dongusu ---
    print(f"\n[2/4] Filtreleme isleniyor...")
    print(f"      Hedef Fs      : {config.TARGET_FS} Hz")
    print(f"      Bandpass      : {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz")
    print(f"      Filtre Order  : {config.BANDPASS_ORDER}")
    print(f"      Normalizasyon : ArcTan (R-peak taming)\n")

    basarili = 0
    basarisiz = 0
    hatali_kayitlar = []

    for idx, row in tqdm(df.iterrows(), total=len(df),
                          desc="      Filtreleme", ncols=80):
        ecg_id = row['ecg_id']
        signal_path = row['signal_path']

        try:
            # Sinyal oku (wfdb tum formatlari destekler: .mat/.hea, .dat/.hea)
            rec = wfdb.rdsamp(signal_path)
            sinyal = rec[0]  # (zaman_adimi, kanal_sayisi)
            meta = rec[1]

            # Transpoz: (kanal_sayisi, zaman_adimi) formatina cevir
            sinyal = sinyal.T  # (12, N)

            # Orijinal Fs
            orijinal_fs = meta.get('fs', row.get('original_fs', config.ORIGINAL_FS))
            if orijinal_fs is None:
                orijinal_fs = config.ORIGINAL_FS

            # Alt ornekleme: 500 Hz -> 250 Hz

            sinyal = alt_ornekle(sinyal, orijinal_fs, config.TARGET_FS)

            # Butterworth bandpass filtre
            sinyal = cok_kanalli_filtrele(sinyal, config.TARGET_FS)

            # ArcTan normalizasyonu
            sinyal = arctan_normalize(sinyal)

            # .npy olarak kaydet
            cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
            np.save(cikti_dosyasi, sinyal.astype(np.float32))

            basarili += 1

        except Exception as e:
            basarisiz += 1
            hatali_kayitlar.append((ecg_id, str(e)[:80]))

    print(f"\n      Basarili: {basarili}")
    print(f"      Basarisiz: {basarisiz}")

    if basarisiz > 0:
        print(f"\n      Ilk 5 hatali kayit:")
        for ecg_id, hata in hatali_kayitlar[:5]:
            print(f"        {ecg_id}: {hata}")

    # --- 4. Filtrelenmis manifest olustur ---
    print(f"\n[3/4] filtered_manifest.csv kaydediliyor...")

    basarili_ids = set()
    for ecg_id in df['ecg_id']:
        if os.path.exists(os.path.join(cikti_dizini, f"{ecg_id}.npy")):
            basarili_ids.add(ecg_id)

    df['filtered'] = df['ecg_id'].isin(basarili_ids)
    df['filtered_path'] = df['ecg_id'].apply(
        lambda x: os.path.join("filtered_signals", f"{x}.npy") if x in basarili_ids else None
    )

    manifest_cikti = os.path.join(config.PROCESSED_DATA_DIR, "filtered_manifest.csv")
    df.to_csv(manifest_cikti, index=False)
    print(f"      Kaydedildi: {manifest_cikti}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    filtrelenmis = df[df['filtered'] == True]
    print(f"  Toplam islenen      : {len(df)}")
    print(f"  Basarili filtrelen  : {len(filtrelenmis)}")
    print(f"  Hedef sinyal boyutu : ({config.NUM_LEADS}, {config.TARGET_LENGTH})")

    sinif_dag = filtrelenmis['label'].value_counts().sort_index()
    print(f"\n  Filtrelenmis Sinif Dagilimi:")
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "?")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    # Ornek sinyal dogrulama
    ornek_dosya = os.path.join(cikti_dizini, f"{filtrelenmis.iloc[0]['ecg_id']}.npy")
    if os.path.exists(ornek_dosya):
        ornek = np.load(ornek_dosya)
        print(f"\n  Ornek sinyal shape  : {ornek.shape}")
        print(f"  Ornek sinyal dtype  : {ornek.dtype}")

    print("\n" + "=" * 70)
    print("Adim 2 tamamlandi. Sonraki adim: adim03_kalite_kontrol.py")
    print("=" * 70)

    return df


if __name__ == "__main__":
    sonuc = filtreleme_pipeline()
