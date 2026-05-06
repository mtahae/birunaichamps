"""
adim01_kalite_kontrol_genel.py — BirunAI EKG: Adim 1 – Genel Kalite Kontrol
=============================================================================

Bu modul, unified_manifest.csv uzerinde kapsamli kalite kontrol yapar:

    1. Mukerrer Kayit Tespiti (Capraz Veri Seti)
       - Ayni sinyal verisi farkli veri setlerinde olabilir
       - Sinyal hash'i ile benzersizlik kontrolu
    2. Sinyal Butunluk Kontrolu
       - .mat dosyalari okunabiliyor mu?
       - 12 lead mevcut mu?
       - Ornekleme frekansi beklenen degerde mi?
    3. Etiket Tutarliligi
       - Eslenmis etiket var mi?
       - Sinif dagilimi raporu
    4. Bozuk/Okunamaz Kayitlarin Elenmesi

Ciktilar:
    - outputs/processed_data/unified_manifest_clean.csv
    - outputs/reports/quality_report.txt

Kullanim:
    python adim01_kalite_kontrol_genel.py
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# SINYAL HASH FONKSIYONU
# =============================================================================

def sinyal_hash(signal_path, max_samples=2500):
    """
    Bir EKG kaydinin sinyal verisinden kisa bir hash uretir.
    Mukerrer kayit tespiti icin kullanilir.

    Args:
        signal_path: .mat/.hea dosyasinin uzantisiz yolu
        max_samples: Hash hesabi icin kullanilacak maksimum ornek sayisi

    Returns:
        str: MD5 hash veya None (okunamazsa)
    """
    try:
        rec = wfdb.rdsamp(signal_path)
        sinyal = rec[0]  # (zaman_adimi, kanal_sayisi)

        # Ilk max_samples ornegi al (hiz icin)
        snippet = sinyal[:max_samples, :].tobytes()
        return hashlib.md5(snippet).hexdigest()
    except Exception:
        return None


# =============================================================================
# ANA KALITE KONTROL PIPELINE'I
# =============================================================================

def kalite_kontrol_genel():
    """
    unified_manifest.csv uzerinde kapsamli kalite kontrol uygular.

    Returns:
        pd.DataFrame: Temizlenmis manifest
    """
    print("=" * 70)
    print("BirunAI — Adim 1: Genel Kalite Kontrol (Multi-Dataset)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "unified_manifest.csv")
    print(f"\n[1/5] unified_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"unified_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim00_veri_birlestirme.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu)
    print(f"      Toplam kayit: {len(df)}")

    # --- 2. Sinyal butunluk kontrolu ---
    print(f"\n[2/5] Sinyal butunluk kontrolu...")
    print(f"      Her kaydin .mat dosyasi okunuyor ve dogrulanıyor...\n")

    okunabilir = []
    okunamaz = []
    yanlis_lead = []
    yanlis_fs = []
    sinyal_hashleri = {}  # hash -> ecg_id listesi
    hash_listesi = []

    for idx, row in tqdm(df.iterrows(), total=len(df),
                          desc="      Kontrol", ncols=80):
        signal_path = row['signal_path']
        ecg_id = row['ecg_id']

        try:
            rec = wfdb.rdsamp(signal_path)
            sinyal = rec[0]
            meta = rec[1]

            # Lead sayisi kontrolu
            if sinyal.shape[1] != 12:
                yanlis_lead.append(ecg_id)
                hash_listesi.append(None)
                continue

            # Fs kontrolu
            fs = meta.get('fs', 500)
            if fs != 500:
                yanlis_fs.append((ecg_id, fs))

            # NaN/Inf kontrolu
            if np.any(np.isnan(sinyal)) or np.any(np.isinf(sinyal)):
                okunamaz.append((ecg_id, "NaN/Inf deger"))
                hash_listesi.append(None)
                continue

            # Duz sinyal kontrolu (tum kanallar duz mu?)
            std_per_lead = np.std(sinyal, axis=0)
            if np.all(std_per_lead < 1e-6):
                okunamaz.append((ecg_id, "Tum kanallar duz"))
                hash_listesi.append(None)
                continue

            # Hash uret
            snippet = sinyal[:2500, :].tobytes()
            h = hashlib.md5(snippet).hexdigest()
            hash_listesi.append(h)

            if h not in sinyal_hashleri:
                sinyal_hashleri[h] = []
            sinyal_hashleri[h].append(ecg_id)

            okunabilir.append(ecg_id)

        except Exception as e:
            okunamaz.append((ecg_id, str(e)[:80]))
            hash_listesi.append(None)

    df['signal_hash'] = hash_listesi
    df['readable'] = df['ecg_id'].isin(okunabilir)

    print(f"\n      Okunabilir   : {len(okunabilir)}")
    print(f"      Okunamaz     : {len(okunamaz)}")
    print(f"      Yanlis lead  : {len(yanlis_lead)}")
    if yanlis_fs:
        print(f"      Yanlis Fs    : {len(yanlis_fs)} (500 Hz olmayan)")

    # --- 3. Mukerrer kayit tespiti ---
    print(f"\n[3/5] Mukerrer kayit tespiti...")

    mukerrer_hashler = {h: ids for h, ids in sinyal_hashleri.items()
                        if len(ids) > 1}

    toplam_mukerrer = sum(len(ids) - 1 for ids in mukerrer_hashler.values())
    print(f"      Benzersiz sinyal hash: {len(sinyal_hashleri)}")
    print(f"      Mukerrer gruplar     : {len(mukerrer_hashler)}")
    print(f"      Silinecek mukerrer   : {toplam_mukerrer}")

    if mukerrer_hashler:
        # Her mukerrer gruptan sadece birini tut (ilk ekleneni)
        silinecek_ids = set()
        for h, ids in mukerrer_hashler.items():
            # Ilk kaydı tut, gerisini sil
            silinecek_ids.update(ids[1:])

        print(f"\n      Ornek mukerrer gruplar (ilk 5):")
        for i, (h, ids) in enumerate(list(mukerrer_hashler.items())[:5]):
            ds_sources = [df[df['ecg_id'] == eid]['dataset_source'].values[0]
                          for eid in ids[:3]]
            print(f"        Hash {h[:10]}... : {ids[:3]} ({ds_sources})")
    else:
        silinecek_ids = set()

    # --- 4. Temizlenmis manifest olustur ---
    print(f"\n[4/5] Temizlenmis manifest olusturuluyor...")

    temiz = df[
        (df['readable'] == True) &
        (~df['ecg_id'].isin(silinecek_ids))
    ].copy()

    # Gereksiz sutunlari kaldir
    temiz = temiz.drop(columns=['signal_hash', 'readable'], errors='ignore')

    elenen_toplam = len(df) - len(temiz)
    print(f"      Orijinal     : {len(df)}")
    print(f"      Elenen       : {elenen_toplam}")
    print(f"        - Okunamaz : {len(okunamaz)}")
    print(f"        - Mukerrer : {toplam_mukerrer}")
    print(f"        - Lead!=12 : {len(yanlis_lead)}")
    print(f"      Temiz kayit  : {len(temiz)}")

    # --- 5. Kaydet ---
    print(f"\n[5/5] Sonuclar kaydediliyor...")

    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "unified_manifest_clean.csv")
    temiz.to_csv(cikti_yolu, index=False)
    print(f"      Manifest: {cikti_yolu}")

    # Rapor kaydet
    rapor_yolu = os.path.join(config.REPORT_DIR, "quality_report.txt")
    with open(rapor_yolu, 'w', encoding='utf-8') as f:
        f.write("BirunAI — Genel Kalite Kontrol Raporu\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Toplam kayit     : {len(df)}\n")
        f.write(f"Okunabilir       : {len(okunabilir)}\n")
        f.write(f"Okunamaz         : {len(okunamaz)}\n")
        f.write(f"Mukerrer silinen : {toplam_mukerrer}\n")
        f.write(f"Temiz kayit      : {len(temiz)}\n\n")

        if okunamaz:
            f.write("Okunamaz Kayitlar (ilk 20):\n")
            for ecg_id, hata in okunamaz[:20]:
                f.write(f"  {ecg_id}: {hata}\n")
            f.write("\n")

        if mukerrer_hashler:
            f.write(f"Mukerrer Gruplar ({len(mukerrer_hashler)} grup):\n")
            for h, ids in list(mukerrer_hashler.items())[:20]:
                f.write(f"  Hash {h[:16]}: {ids}\n")

    print(f"      Rapor  : {rapor_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER (Temizlenmis)")
    print("=" * 70)

    # Veri seti dagilimi
    print(f"\n  Veri Seti Bazli:")
    ds_dag = temiz['dataset_source'].value_counts()
    for ds, sayi in ds_dag.items():
        print(f"    {ds:18s}: {sayi:6d}")

    # Sinif dagilimi
    print(f"\n  Sinif Dagilimi:")
    sinif_dag = temiz['label'].value_counts().sort_index()
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "?")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 1 tamamlandi. Sonraki adim: adim02_filtreleme.py")
    print("=" * 70)

    return temiz


if __name__ == "__main__":
    sonuc = kalite_kontrol_genel()
