"""
adim04_segmentasyon.py — BirunAI EKG Siniflandirma: Adim 4 – Segmentasyon
==========================================================================

Bu modul, filtrelenmis EKG sinyallerini sabit uzunlukta pencerelere boler.

Projemizde belirttigimiz gibi:
    - 10 saniye sabit pencere: Klinik 12-lead EKG standardi ile uyumlu.
    - 250 Hz x 10 sn = 2500 zaman adimi.
    - PTB-XL kayitlari zaten 10 sn oldugundan, cogu kayit direkt kullanilir.
    - Kisa kayitlar: Sifir-padding (zero-pad) uygulanir.
    - Uzun kayitlar: Ortadan kirpilir (center-crop).

Ciktilar:
    - outputs/processed_data/segmented_signals/  (her kayit icin .npy dosyasi)
    - outputs/processed_data/segmented_manifest.csv

Kullanim:
    python adim04_segmentasyon.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# SEGMENTASYON FONKSIYONLARI
# =============================================================================

def sabit_pencere_uygula(sinyal_2d, hedef_uzunluk=None):
    """
    Sinyali sabit uzunlukta pencereye uyarlar.

    Projemizde belirttigimiz gibi:
    - Hedef uzunluk: 2500 ornek (10 sn @ 250 Hz)
    - Kisa sinyaller: Sona sifir-padding eklenir.
    - Uzun sinyaller: Bastan ve sondan esit olarak kirpilir (center-crop).
    - Tam uzunlukta sinyaller: Olduklari gibi korunur.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        hedef_uzunluk: Hedef zaman adimi sayisi. Varsayilan: config.TARGET_LENGTH

    Returns:
        numpy array: (kanal_sayisi, hedef_uzunluk) formatinda numpy array
    """
    if hedef_uzunluk is None:
        hedef_uzunluk = config.TARGET_LENGTH

    kanal_sayisi = sinyal_2d.shape[0]
    mevcut_uzunluk = sinyal_2d.shape[1]

    if mevcut_uzunluk == hedef_uzunluk:
        # Tam uyum — olduklari gibi don
        return sinyal_2d

    elif mevcut_uzunluk < hedef_uzunluk:
        # KISA sinyal — sifir-padding
        padded = np.zeros((kanal_sayisi, hedef_uzunluk), dtype=sinyal_2d.dtype)
        padded[:, :mevcut_uzunluk] = sinyal_2d
        return padded

    else:
        # UZUN sinyal — center-crop
        baslangic = (mevcut_uzunluk - hedef_uzunluk) // 2
        return sinyal_2d[:, baslangic:baslangic + hedef_uzunluk]


# =============================================================================
# ANA SEGMENTASYON PIPELINE'I
# =============================================================================

def segmentasyon_pipeline():
    """
    QC'den gecen tum sinyallere segmentasyon uygular.

    Islem Akisi:
        1. quality_manifest.csv okunur.
        2. Sadece qc_pass=True olan kayitlar secilir.
        3. Her kayit icin sabit pencere uygulanir.
        4. Segmente edilmis sinyaller .npy olarak kaydedilir.
        5. segmented_manifest.csv kaydedilir.

    Returns:
        pd.DataFrame: Segmente edilmis manifest.
    """
    print("=" * 70)
    print("BirunAI -- Adim 4: Segmentasyon (10sn Sabit Pencere)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "quality_manifest.csv")
    print(f"\n[1/3] quality_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"quality_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim03_kalite_kontrol.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    gecerli = df[df["qc_pass"] == True].copy()
    print(f"      QC gecen kayit: {len(gecerli)}")

    # --- 2. Cikti dizinini olustur ---
    kaynak_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    cikti_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    os.makedirs(cikti_dizini, exist_ok=True)

    # --- 3. Segmentasyon dongusu ---
    print(f"\n[2/3] Segmentasyon uygulanyor...")
    print(f"      Hedef uzunluk : {config.TARGET_LENGTH} ornek")
    print(f"      = {config.WINDOW_SEC} saniye @ {config.TARGET_FS} Hz")

    basarili = 0
    padded_sayisi = 0
    cropped_sayisi = 0
    tam_sayisi = 0
    hatali = 0

    for ecg_id, satir in tqdm(gecerli.iterrows(), total=len(gecerli),
                               desc="      Segmentasyon"):
        try:
            # Sinyal yukle
            sinyal_dosyasi = os.path.join(kaynak_dizini, f"{ecg_id}.npy")
            sinyal = np.load(sinyal_dosyasi)

            orijinal_uzunluk = sinyal.shape[1]

            # Sabit pencere uygula
            segmente = sabit_pencere_uygula(sinyal)

            # Istatistik
            if orijinal_uzunluk == config.TARGET_LENGTH:
                tam_sayisi += 1
            elif orijinal_uzunluk < config.TARGET_LENGTH:
                padded_sayisi += 1
            else:
                cropped_sayisi += 1

            # Kaydet
            cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
            np.save(cikti_dosyasi, segmente.astype(np.float32))

            basarili += 1

        except Exception as e:
            hatali += 1

    print(f"\n      Basarili     : {basarili}")
    print(f"      Hatali       : {hatali}")
    print(f"      Tam uzunluk  : {tam_sayisi}")
    print(f"      Padding      : {padded_sayisi}")
    print(f"      Cropping     : {cropped_sayisi}")

    # --- 4. Segmente manifest ---
    print(f"\n[3/3] segmented_manifest.csv kaydediliyor...")

    # Basarili kayitlari isaretmek
    basarili_ids = set()
    for ecg_id in gecerli.index:
        cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
        if os.path.exists(cikti_dosyasi):
            basarili_ids.add(ecg_id)

    gecerli["segmented"] = gecerli.index.isin(basarili_ids)
    gecerli["segmented_path"] = gecerli.index.map(
        lambda x: os.path.join("segmented_signals", f"{x}.npy") if x in basarili_ids else None
    )

    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    gecerli.to_csv(cikti_yolu)
    print(f"      Kaydedildi: {cikti_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    segmente_kayitlar = gecerli[gecerli["segmented"] == True]
    print(f"  Toplam segmente edilen : {len(segmente_kayitlar)}")
    print(f"  Sinyal boyutu          : ({config.NUM_LEADS}, {config.TARGET_LENGTH})")

    # Sinif dagilimi
    sinif_dag = segmente_kayitlar["label"].value_counts().sort_index()
    print(f"\n  Sinif Dagilimi:")
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    # Ornek sinyal dogrulama
    ornek_dosya = os.path.join(cikti_dizini, f"{segmente_kayitlar.index[0]}.npy")
    if os.path.exists(ornek_dosya):
        ornek = np.load(ornek_dosya)
        print(f"\n  Ornek sinyal shape  : {ornek.shape}")
        print(f"  Ornek sinyal dtype  : {ornek.dtype}")

    # Disk kullanimi
    toplam_boyut_mb = 0
    for f in os.listdir(cikti_dizini):
        if f.endswith(".npy"):
            toplam_boyut_mb += os.path.getsize(os.path.join(cikti_dizini, f))
    toplam_boyut_mb /= (1024 * 1024)
    print(f"\n  Toplam disk kullanimi  : {toplam_boyut_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("Adim 4 tamamlandi. Sonraki adim: adim05_ozellik_cikarma.py")
    print("=" * 70)

    return gecerli


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = segmentasyon_pipeline()
