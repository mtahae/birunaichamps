"""
adim03_kalite_kontrol.py — BirunAI EKG Siniflandirma: Adim 3 – Kalite Kontrol
==============================================================================

Bu modul, filtrelenmis EKG sinyallerinin kalitesini denetler ve
kullanim disi birakilacak kayitlari isaretler.

Kalite Kriterleri:
    1. Duz Sinyal Tespiti (Flat-line):
       - Standart sapma < 0.01 olan derivasyonlar "duz sinyal" olarak isaretlenir.
       - Tum derivasyonlari duz olan kayitlar elenecek (elektrot kopmasi).

    2. Asiri Genlik (Amplitude Clipping):
       - |z-score| > 20 olan orneklerin orani > %5 ise "clipping" isaretlenir.
       - Cihaz saturation veya hareket artefakti gostergesidir.

    3. PTB-XL Kalite Bayraklari (metadata):
       - ptbxl_database.csv dosyasindaki baseline_drift, static_noise,
         burst_noise, electrodes_problems bayraklari degerlendirilir.
       - electrodes_problems olan kayitlar dogrudan elenecek.

    4. Kayit Uzunlugu Kontrolu:
       - Hedef uzunluktan (2500 ornek = 10sn @ 250Hz) belirgin sapma.

Ciktilar:
    - outputs/processed_data/quality_manifest.csv
      Eklenen sutunlar: qc_flat_channels, qc_clipping, qc_electrode_problem,
                        qc_length_ok, qc_pass (tum kontrollerden geciyor mu)

Kullanim:
    python adim03_kalite_kontrol.py
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
# KALITE KONTROL FONKSIYONLARI
# =============================================================================

def duz_sinyal_tespit(sinyal_2d, esik=0.01):
    """
    Duz sinyal (flat-line) tespiti.

    Elektrot kopmasi veya baglanti sorunlarinda bir veya birden fazla
    derivasyon duz bir cizgi gosterir (standart sapma ~ 0).

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        esik: Standart sapma esigi. Altindaki kanallar "duz" kabul edilir.

    Returns:
        int: Duz sinyal gosteren kanal sayisi (0-12)
    """
    duz_kanal_sayisi = 0
    for kanal_idx in range(sinyal_2d.shape[0]):
        if np.std(sinyal_2d[kanal_idx]) < esik:
            duz_kanal_sayisi += 1
    return duz_kanal_sayisi


def clipping_tespit(sinyal_2d, z_esik=20.0, oran_esik=0.05):
    """
    Asiri genlik (clipping/saturation) tespiti.

    Cihaz saturation veya asiri hareket artefaktlarinda sinyal
    surekli maksimum/minimum degerde kalir.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        z_esik: |z-score| esigi
        oran_esik: Asilan orneklerin toplam ornege orani esigi

    Returns:
        bool: True ise clipping var
    """
    toplam_ornek = sinyal_2d.size
    asiri_ornek = np.sum(np.abs(sinyal_2d) > z_esik)
    oran = asiri_ornek / toplam_ornek
    return oran > oran_esik


def uzunluk_kontrol(sinyal_2d, hedef_uzunluk=None, tolerans=0.1):
    """
    Sinyal uzunlugunun hedef uzunluga uygunlugunun kontrolu.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        hedef_uzunluk: Beklenen zaman adimi sayisi. Varsayilan: config.TARGET_LENGTH
        tolerans: Kabul edilebilir sapma orani (0.1 = %10)

    Returns:
        bool: True ise uzunluk kabul edilebilir
    """
    if hedef_uzunluk is None:
        hedef_uzunluk = config.TARGET_LENGTH

    gercek_uzunluk = sinyal_2d.shape[1]
    sapma = abs(gercek_uzunluk - hedef_uzunluk) / hedef_uzunluk
    return sapma <= tolerans


# =============================================================================
# ANA KALITE KONTROL PIPELINE'I
# =============================================================================

def kalite_kontrol_pipeline():
    """
    Tum filtrelenmis sinyallere kalite kontrol uygular.

    Islem Akisi:
        1. filtered_manifest.csv okunur.
        2. Her sinyal icin kalite metrikleri hesaplanir.
        3. Elektrot problemi bayraklari kontrol edilir.
        4. Nihai QC sonucu belirlenir.
        5. quality_manifest.csv kaydedilir.

    Returns:
        pd.DataFrame: Kalite kontrol edilmis manifest.
    """
    print("=" * 70)
    print("BirunAI -- Adim 3: Kalite Kontrol")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "filtered_manifest.csv")
    print(f"\n[1/3] filtered_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"filtered_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim02_filtreleme.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    filtrelenmis = df[df["filtered"] == True].copy()
    print(f"      Filtrelenmis kayit: {len(filtrelenmis)}")

    # --- 2. Kalite kontrol ---
    print(f"\n[2/3] Kalite kontrol uygulanyor...")

    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")

    qc_duz_kanallar = []
    qc_clipping = []
    qc_uzunluk = []
    qc_elektrot = []

    for ecg_id, satir in tqdm(filtrelenmis.iterrows(), total=len(filtrelenmis),
                               desc="      Kalite Kontrol"):
        # Sinyal yukle
        sinyal_dosyasi = os.path.join(sinyal_dizini, f"{ecg_id}.npy")

        try:
            sinyal = np.load(sinyal_dosyasi)

            # Test 1: Duz sinyal tespiti
            duz_sayisi = duz_sinyal_tespit(sinyal)
            qc_duz_kanallar.append(duz_sayisi)

            # Test 2: Clipping tespiti
            clip = clipping_tespit(sinyal)
            qc_clipping.append(clip)

            # Test 3: Uzunluk kontrolu
            uzunluk_ok = uzunluk_kontrol(sinyal)
            qc_uzunluk.append(uzunluk_ok)

        except Exception as e:
            qc_duz_kanallar.append(12)  # Tum kanallar "kotu"
            qc_clipping.append(True)
            qc_uzunluk.append(False)

        # Test 4: PTB-XL elektrot problemi bayragi
        # Bu sutun NaN, 0, veya elektrot ismi (orn: 'V6', 'aVR') icerebilir
        elektrot_prob = satir.get("electrodes_problems", 0)
        if pd.isna(elektrot_prob):
            has_problem = False
        elif isinstance(elektrot_prob, str):
            # String ise: '0' degeri yok demek, diger degerler problem var demek
            has_problem = elektrot_prob.strip() != "" and elektrot_prob.strip() != "0"
        else:
            has_problem = float(elektrot_prob) > 0
        qc_elektrot.append(has_problem)

    # Sonuclari DataFrame'e ekle
    filtrelenmis["qc_flat_channels"] = qc_duz_kanallar
    filtrelenmis["qc_clipping"] = qc_clipping
    filtrelenmis["qc_length_ok"] = qc_uzunluk
    filtrelenmis["qc_electrode_problem"] = qc_elektrot

    # Nihai QC karari:
    # - Tum kanallar duz degilse (en az 1 gecerli kanal)
    # - Clipping yok
    # - Uzunluk uygun
    # - Elektrot problemi yok
    filtrelenmis["qc_pass"] = (
        (filtrelenmis["qc_flat_channels"] < config.NUM_LEADS) &
        (~filtrelenmis["qc_clipping"]) &
        (filtrelenmis["qc_length_ok"]) &
        (~filtrelenmis["qc_electrode_problem"])
    )

    # --- 3. Sonuclari kaydet ---
    print(f"\n[3/3] quality_manifest.csv kaydediliyor...")
    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "quality_manifest.csv")
    filtrelenmis.to_csv(cikti_yolu)
    print(f"      Kaydedildi: {cikti_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    gecen = filtrelenmis[filtrelenmis["qc_pass"] == True]
    kalan = filtrelenmis[filtrelenmis["qc_pass"] == False]

    print(f"  Toplam kontrol edilen : {len(filtrelenmis)}")
    print(f"  QC GECEN              : {len(gecen)} ({len(gecen)/len(filtrelenmis)*100:.1f}%)")
    print(f"  QC KALAN (elenen)     : {len(kalan)} ({len(kalan)/len(filtrelenmis)*100:.1f}%)")

    # Eleme nedenleri
    duz_eleme = (filtrelenmis["qc_flat_channels"] >= config.NUM_LEADS).sum()
    clip_eleme = filtrelenmis["qc_clipping"].sum()
    uzunluk_eleme = (~filtrelenmis["qc_length_ok"]).sum()
    elektrot_eleme = filtrelenmis["qc_electrode_problem"].sum()

    print(f"\n  Eleme Nedenleri (cakisabilir):")
    print(f"    Tum kanallar duz (flat-line)  : {duz_eleme}")
    print(f"    Asiri genlik (clipping)       : {clip_eleme}")
    print(f"    Uzunluk uyumsuzlugu           : {uzunluk_eleme}")
    print(f"    Elektrot problemi (metadata)  : {elektrot_eleme}")

    # Duz kanal dagilimi
    duz_dagilim = filtrelenmis["qc_flat_channels"].value_counts().sort_index()
    print(f"\n  Duz Kanal Sayisi Dagilimi:")
    for duz_sayi, kayit_sayi in duz_dagilim.items():
        durum = " [ELENECEK]" if duz_sayi >= config.NUM_LEADS else ""
        print(f"    {duz_sayi:2d} duz kanal : {kayit_sayi:6d} kayit{durum}")

    # QC gecen kayitlarin sinif dagilimi
    if len(gecen) > 0:
        sinif_dag = gecen["label"].value_counts().sort_index()
        print(f"\n  QC Gecen Kayitlarda Sinif Dagilimi:")
        for sinif_idx, sayi in sinif_dag.items():
            sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
            oran = sayi / sinif_dag.sum() * 100
            print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 3 tamamlandi. Sonraki adim: adim04_segmentasyon.py")
    print("=" * 70)

    return filtrelenmis


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = kalite_kontrol_pipeline()
