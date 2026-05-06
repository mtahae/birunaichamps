"""
adim04_segmentasyon.py — BirunAI EKG: Adim 4 – Segmentasyon
=============================================================

Bu modul, filtrelenmis sinyalleri sabit 10-sn pencereye (2500 ornek @ 250 Hz)
standardize eder.

Strateji:
    - Uzun sinyaller: Merkez crop
    - Kisa sinyaller: Simetrik zero-padding

Ciktilar:
    - outputs/processed_data/segmented_signals/   (her kayit icin .npy)
    - outputs/processed_data/segmented_manifest.csv

Kullanim:
    python adim04_segmentasyon.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def segment_sinyal(sinyal_2d, hedef_uzunluk=None):
    """
    Tek bir sinyali hedef uzunluga getirir.

    Args:
        sinyal_2d: np.ndarray (12, N)
        hedef_uzunluk: int — default config.TARGET_LENGTH

    Returns:
        np.ndarray (12, hedef_uzunluk)
    """
    if hedef_uzunluk is None:
        hedef_uzunluk = config.TARGET_LENGTH

    mevcut = sinyal_2d.shape[1]

    if mevcut == hedef_uzunluk:
        return sinyal_2d

    elif mevcut > hedef_uzunluk:
        # Merkez crop
        baslangic = (mevcut - hedef_uzunluk) // 2
        return sinyal_2d[:, baslangic:baslangic + hedef_uzunluk]

    else:
        # Simetrik zero-padding
        sonuc = np.zeros((sinyal_2d.shape[0], hedef_uzunluk), dtype=sinyal_2d.dtype)
        pad = (hedef_uzunluk - mevcut) // 2
        sonuc[:, pad:pad + mevcut] = sinyal_2d
        return sonuc


def segmentasyon_pipeline():
    print("=" * 70)
    print("BirunAI — Adim 4: Segmentasyon")
    print("=" * 70)

    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "quality_manifest.csv")
    print(f"\n[1/3] quality_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(f"Bulunamadi: {manifest_yolu}")

    df = pd.read_csv(manifest_yolu)
    gecen = df[df['qc_pass'] == True].copy()
    print(f"      QC gecen kayit: {len(gecen)}")

    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    cikti_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    os.makedirs(cikti_dizini, exist_ok=True)

    print(f"\n[2/3] Segmentasyon uygulanıyor...")
    print(f"      Hedef: ({config.NUM_LEADS}, {config.TARGET_LENGTH})\n")

    basarili = 0
    basarisiz = 0
    istatistikler = {'crop': 0, 'pad': 0, 'exact': 0}

    for idx, row in tqdm(gecen.iterrows(), total=len(gecen),
                          desc="      Segmentasyon", ncols=80):
        ecg_id = row['ecg_id']
        giris_dosyasi = os.path.join(sinyal_dizini, f"{ecg_id}.npy")

        try:
            sinyal = np.load(giris_dosyasi)

            if sinyal.shape[1] > config.TARGET_LENGTH:
                istatistikler['crop'] += 1
            elif sinyal.shape[1] < config.TARGET_LENGTH:
                istatistikler['pad'] += 1
            else:
                istatistikler['exact'] += 1

            sinyal_seg = segment_sinyal(sinyal)
            assert sinyal_seg.shape == (config.NUM_LEADS, config.TARGET_LENGTH)

            cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
            np.save(cikti_dosyasi, sinyal_seg.astype(np.float32))
            basarili += 1

        except Exception as e:
            basarisiz += 1

    print(f"\n      Basarili : {basarili}")
    print(f"      Basarisiz: {basarisiz}")
    print(f"      Crop     : {istatistikler['crop']}")
    print(f"      Pad      : {istatistikler['pad']}")
    print(f"      Exact    : {istatistikler['exact']}")

    # Manifest kaydet
    print(f"\n[3/3] segmented_manifest.csv kaydediliyor...")
    basarili_ids = set()
    for ecg_id in gecen['ecg_id']:
        if os.path.exists(os.path.join(cikti_dizini, f"{ecg_id}.npy")):
            basarili_ids.add(ecg_id)

    gecen['segmented'] = gecen['ecg_id'].isin(basarili_ids)
    sonuc = gecen[gecen['segmented'] == True].copy()

    manifest_cikti = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    sonuc.to_csv(manifest_cikti, index=False)
    print(f"      Kaydedildi: {manifest_cikti}")
    print(f"      Toplam segmentlenmis: {len(sonuc)}")

    sinif_dag = sonuc['label'].value_counts().sort_index()
    print(f"\n  Sinif Dagilimi:")
    for s, n in sinif_dag.items():
        print(f"    [{int(s)}] {config.LABEL_NAMES.get(int(s),'?'):20s}: {n:6d} ({n/sinif_dag.sum()*100:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 4 tamamlandi. Sonraki: adim06_veri_bolme.py")
    print("=" * 70)
    return sonuc


if __name__ == "__main__":
    segmentasyon_pipeline()
