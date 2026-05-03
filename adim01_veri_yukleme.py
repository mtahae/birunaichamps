"""
adim01_veri_yukleme.py — BirunAI EKG Siniflandirma: Adim 1 – Veri Yukleme
============================================================================

Bu modul, PTB-XL veri setini okur, SCP kodlarini 3 sinifa esler ve
raw_manifest.csv dosyasini uretir.

Islem Akisi:
    1. ptbxl_database.csv okunur (21,799 satir)
    2. scp_codes sutunu parse edilir (string -> Python dict)
    3. Her kayit icin SCP kodlari SCP_TO_LABEL tablosuna gore 3 sinifa eslenir
    4. Cakisma onceligi: Iletim (2) > Ritim (1) > Normal (0)
    5. .dat + .hea dosya ciftlerinin varligi kontrol edilir
    6. raw_manifest.csv uretilir

Ciktilar:
    - outputs/processed_data/raw_manifest.csv

Kullanim:
    python adim01_veri_yukleme.py
"""

import os
import sys
import ast
import numpy as np
import pandas as pd

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def scp_kodlarini_parse_et(scp_str):
    """
    scp_codes sutunundaki string ifadeyi Python dict'e cevirir.

    PTB-XL'de scp_codes sutunu su formatta:
        "{'NORM': 100.0}"
        "{'AFIB': 80.0, 'SR': 0.0}"

    Args:
        scp_str: String formatta SCP kodlari

    Returns:
        dict: {SCP_kodu: olabilirlik_skoru} veya bos dict
    """
    try:
        if pd.isna(scp_str):
            return {}
        return ast.literal_eval(scp_str)
    except (ValueError, SyntaxError):
        return {}


def scp_to_label(scp_dict, esik=0.0):
    """
    SCP kodlarindan 3-sinifli etikete donusturur.

    Cakisma mantigi:
        Bir hastada birden fazla SCP kodu olabilir.
        Oncelik: Iletim (2) > Ritim (1) > Normal (0)
        En yuksek oncelikli sinif atanir.

    Args:
        scp_dict: {SCP_kodu: olabilirlik_skoru} dict'i
        esik: Minimum olabilirlik skoru (altindaki kodlar dikkate alinmaz)

    Returns:
        int or None: 0, 1, 2 veya None (eslenemezse)
    """
    if not scp_dict:
        return None

    bulunan_etiketler = set()

    for kod, skor in scp_dict.items():
        # Dusuk olabilirlikli kodlari atla
        if skor <= esik:
            continue

        # SCP_TO_LABEL tablosunda ara
        if kod in config.SCP_TO_LABEL:
            bulunan_etiketler.add(config.SCP_TO_LABEL[kod])

    if not bulunan_etiketler:
        return None

    # Cakisma onceligi: Iletim (2) > Ritim (1) > Normal (0)
    return max(bulunan_etiketler, key=lambda x: config.LABEL_PRIORITY.get(x, 0))


def dosya_varligini_kontrol_et(filename_hr):
    """
    Bir EKG kaydinin .dat ve .hea dosyalarinin var olup olmadigini kontrol eder.

    Args:
        filename_hr: records500 altindaki dosya yolu (uzantisiz)

    Returns:
        bool: Her iki dosya da mevcutsa True
    """
    tam_yol = os.path.join(config.PTBXL_ROOT, filename_hr)
    dat_var = os.path.exists(tam_yol + ".dat")
    hea_var = os.path.exists(tam_yol + ".hea")
    return dat_var and hea_var


def veri_yukleme_pipeline():
    """
    PTB-XL verisini yukler, etiketler ve raw_manifest.csv uretir.

    Returns:
        pd.DataFrame: raw_manifest DataFrame'i
    """
    print("=" * 70)
    print("BirunAI -- Adim 1: Veri Yukleme ve Etiketleme")
    print("=" * 70)

    # --- 1. Veritabanini oku ---
    csv_yolu = os.path.join(config.PTBXL_ROOT, "ptbxl_database.csv")
    print(f"\n[1/4] ptbxl_database.csv okunuyor...")

    if not os.path.exists(csv_yolu):
        raise FileNotFoundError(
            f"ptbxl_database.csv bulunamadi: {csv_yolu}\n"
            "PTB-XL veri seti yolunu config.py'de kontrol edin."
        )

    df = pd.read_csv(csv_yolu, index_col="ecg_id")
    print(f"      Toplam kayit: {len(df)}")

    # --- 2. SCP kodlarini parse et ve etiketle ---
    print(f"\n[2/4] SCP kodlari parse ediliyor ve etiketleniyor...")

    df["scp_dict"] = df["scp_codes"].apply(scp_kodlarini_parse_et)
    df["label"] = df["scp_dict"].apply(scp_to_label)

    etiketlenen = df[df["label"].notna()]
    etiketlenemeyen = df[df["label"].isna()]
    print(f"      Etiketlenen   : {len(etiketlenen)}")
    print(f"      Etiketlenemeyen: {len(etiketlenemeyen)}")

    # --- 3. Dosya varligini kontrol et ---
    print(f"\n[3/4] Dosya varliklari kontrol ediliyor...")

    df["file_exists"] = df["filename_hr"].apply(dosya_varligini_kontrol_et)

    dosya_var = df["file_exists"].sum()
    dosya_yok = (~df["file_exists"]).sum()
    print(f"      Dosyasi mevcut : {dosya_var}")
    print(f"      Dosyasi eksik  : {dosya_yok}")

    # --- 4. raw_manifest.csv olustur ---
    print(f"\n[4/4] raw_manifest.csv kaydediliyor...")

    # Gerekli sutunlari sec
    manifest_cols = ["filename_hr", "label", "patient_id", "strat_fold",
                     "file_exists"]

    # Eger electrodes_problems sutunu varsa ekle (QC icin lazim)
    if "electrodes_problems" in df.columns:
        manifest_cols.append("electrodes_problems")

    manifest = df[manifest_cols].copy()

    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "raw_manifest.csv")
    manifest.to_csv(cikti_yolu)
    print(f"      Kaydedildi: {cikti_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    gecerli = manifest[(manifest["label"].notna()) & (manifest["file_exists"] == True)]
    print(f"  Toplam kayit      : {len(df)}")
    print(f"  Gecerli kayit     : {len(gecerli)}")

    sinif_dag = gecerli["label"].value_counts().sort_index()
    print(f"\n  Sinif Dagilimi:")
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 1 tamamlandi. Sonraki adim: adim02_filtreleme.py")
    print("=" * 70)

    return manifest


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = veri_yukleme_pipeline()
