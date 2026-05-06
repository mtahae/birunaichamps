"""
adim03_kalite_kontrol.py — BirunAI EKG: Adim 3 – Sinyal Kalite Kontrol
========================================================================

Bu modul, filtrelenmis EKG sinyallerinin kalitesini denetler.
Dataset-agnostik: tum veri setlerinden gelen sinyallere ayni kurallar uygulanir.

Kalite Kriterleri:
    1. Duz Sinyal Tespiti (Flat-line): std < 0.01
    2. Asiri Genlik (Clipping): |z| > 20 orani > %5
    3. Kayit Uzunlugu Kontrolu

Ciktilar:
    - outputs/processed_data/quality_manifest.csv

Kullanim:
    python adim03_kalite_kontrol.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def duz_sinyal_tespit(sinyal_2d, esik=0.01):
    """Standart sapmasi < esik olan kanal sayisini dondurur."""
    duz = 0
    for k in range(sinyal_2d.shape[0]):
        if np.std(sinyal_2d[k]) < esik:
            duz += 1
    return duz


def clipping_tespit(sinyal_2d, z_esik=20.0, oran_esik=0.05):
    """Asiri genlik orani > esik mi?"""
    toplam = sinyal_2d.size
    asiri = np.sum(np.abs(sinyal_2d) > z_esik)
    return (asiri / toplam) > oran_esik


def uzunluk_kontrol(sinyal_2d, hedef=None, tolerans=0.1):
    """Sinyal uzunlugu hedef aralıkta mi?"""
    if hedef is None:
        hedef = config.TARGET_LENGTH
    sapma = abs(sinyal_2d.shape[1] - hedef) / hedef
    return sapma <= tolerans


def kalite_kontrol_pipeline():
    print("=" * 70)
    print("BirunAI — Adim 3: Sinyal Kalite Kontrol")
    print("=" * 70)

    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "filtered_manifest.csv")
    print(f"\n[1/3] filtered_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(f"Bulunamadi: {manifest_yolu}")

    df = pd.read_csv(manifest_yolu)
    filtrelenmis = df[df['filtered'] == True].copy()
    print(f"      Filtrelenmis kayit: {len(filtrelenmis)}")

    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")

    print(f"\n[2/3] Kalite kontrol uygulanyor...")

    qc_duz = []
    qc_clip = []
    qc_uzunluk = []

    for idx, row in tqdm(filtrelenmis.iterrows(), total=len(filtrelenmis),
                          desc="      Kalite Kontrol", ncols=80):
        ecg_id = row['ecg_id']
        sinyal_dosyasi = os.path.join(sinyal_dizini, f"{ecg_id}.npy")

        try:
            sinyal = np.load(sinyal_dosyasi)
            qc_duz.append(duz_sinyal_tespit(sinyal))
            qc_clip.append(clipping_tespit(sinyal))
            qc_uzunluk.append(uzunluk_kontrol(sinyal))
        except Exception:
            qc_duz.append(12)
            qc_clip.append(True)
            qc_uzunluk.append(False)

    filtrelenmis['qc_flat_channels'] = qc_duz
    filtrelenmis['qc_clipping'] = qc_clip
    filtrelenmis['qc_length_ok'] = qc_uzunluk

    # QC karari
    filtrelenmis['qc_pass'] = (
        (filtrelenmis['qc_flat_channels'] < config.NUM_LEADS) &
        (~filtrelenmis['qc_clipping']) &
        (filtrelenmis['qc_length_ok'])
    )

    print(f"\n[3/3] quality_manifest.csv kaydediliyor...")
    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "quality_manifest.csv")
    filtrelenmis.to_csv(cikti_yolu, index=False)
    print(f"      Kaydedildi: {cikti_yolu}")

    # Ozet
    print("\n" + "=" * 70)
    gecen = filtrelenmis[filtrelenmis['qc_pass'] == True]
    kalan = filtrelenmis[filtrelenmis['qc_pass'] == False]

    print(f"  QC GECEN  : {len(gecen)} ({len(gecen)/len(filtrelenmis)*100:.1f}%)")
    print(f"  QC KALAN  : {len(kalan)} ({len(kalan)/len(filtrelenmis)*100:.1f}%)")
    print(f"  Flat-line : {(filtrelenmis['qc_flat_channels'] >= 12).sum()}")
    print(f"  Clipping  : {filtrelenmis['qc_clipping'].sum()}")
    print(f"  Uzunluk   : {(~filtrelenmis['qc_length_ok']).sum()}")

    sinif_dag = gecen['label'].value_counts().sort_index()
    print(f"\n  QC Gecen Sinif Dagilimi:")
    for s, n in sinif_dag.items():
        print(f"    [{int(s)}] {config.LABEL_NAMES.get(int(s),'?'):20s}: {n:6d} ({n/sinif_dag.sum()*100:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 3 tamamlandi. Sonraki: adim04_segmentasyon.py")
    print("=" * 70)
    return filtrelenmis


if __name__ == "__main__":
    kalite_kontrol_pipeline()
