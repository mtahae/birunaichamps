"""
adim06_veri_bolme.py — BirunAI EKG Siniflandirma: Adim 6 – Veri Bolme
======================================================================

Bu modul, segmente edilmis veriyi hasta bazli train/val/test setlerine boler.

Projemizde belirttigimiz kritik tasarim karari:
    - HASTA BAZLI BOLME: Ayni hastanin farkli kayitlari ASLA farkli
      setlerde yer almaz. Bu, veri sizintisini (data leakage) onler.
    - PTB-XL'in kendi strat_fold sutunu kullanilir (1-10 arasi fold).
      -> Fold 9-10: Test seti (sabit, degismez)
      -> Fold 1-8: Train + Validation (kendi icinde %87.5-%12.5 bolunur)

Veri Sizintisi Onleme:
    Eger ayni hasta birden fazla kayda sahipse ve bu kayitlar
    train/test'e dagitilirsa, model hastanin bireysel ozelliklerini
    ogenir (generalizasyon degil, ezberleme). Bu nedenle bolme
    HASTA bazli yapilir.

Sinif Dengesizligi Ele Alma:
    - WeightedRandomSampler icin sinif agirliklari hesaplanir.
    - Agirliklar: 1 / sinif_sayisi (inverse frequency)
    - Adim 8'de Focal Loss ile birlikte kullanilacak.

Ciktilar:
    - outputs/processed_data/train_manifest.csv
    - outputs/processed_data/val_manifest.csv
    - outputs/processed_data/test_manifest.csv
    - outputs/processed_data/class_weights.npy

Kullanim:
    python adim06_veri_bolme.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# VERI BOLME FONKSIYONLARI
# =============================================================================

def ptbxl_strat_fold_bolme(df, test_folds=(9, 10), val_ratio=0.125):
    """
    PTB-XL'in strat_fold sutununu kullanarak hasta bazli bolme yapar.

    PTB-XL veri seti, kendi icinde stratified fold yapisina sahiptir.
    Bu foldlar hasta bazli olusturulmustur, yani ayni hastanin
    tum kayitlari ayni fold'dadir.

    Args:
        df: Segmente edilmis manifest DataFrame (ecg_id index).
        test_folds: Test seti icin kullanilacak fold numaralari.
        val_ratio: Train setinden validation'a ayrilacak oran.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Test seti: fold 9 ve 10
    test_mask = df["strat_fold"].isin(test_folds)
    test_df = df[test_mask].copy()
    train_val_df = df[~test_mask].copy()

    # Train/Val bolmesi: Hasta bazli
    # Kalan foldlardan (1-8), son %12.5'i validation olarak ayir
    # Fold 8 -> validation, Fold 1-7 -> train
    val_fold = max(f for f in train_val_df["strat_fold"].unique() if f not in test_folds)
    val_mask = train_val_df["strat_fold"] == val_fold
    val_df = train_val_df[val_mask].copy()
    train_df = train_val_df[~val_mask].copy()

    return train_df, val_df, test_df


def sinif_agirliklari_hesapla(labels):
    """
    Sinif dengesizligini ele almak icin sinif agirliklarini hesaplar.

    Agirlik formulu: w_i = toplam_ornek / (sinif_sayisi * sinif_i_ornek)
    Bu formul, sklearn'in class_weight='balanced' yaklasimi ile aynidir.

    Args:
        labels: Etiket dizisi (numpy array veya list)

    Returns:
        numpy array: Her sinif icin agirlik (sinif indeksine gore sirali)
    """
    labels = np.array(labels, dtype=int)
    siniflar = np.unique(labels)
    toplam = len(labels)
    n_sinif = len(siniflar)

    agirliklar = np.zeros(config.NUM_CLASSES)
    for sinif in siniflar:
        sinif_sayisi = np.sum(labels == sinif)
        agirliklar[int(sinif)] = toplam / (n_sinif * sinif_sayisi)

    return agirliklar


# =============================================================================
# ANA VERI BOLME PIPELINE'I
# =============================================================================

def veri_bolme_pipeline():
    """
    Segmente edilmis veriyi train/val/test setlerine boler.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("=" * 70)
    print("BirunAI -- Adim 6: Veri Bolme (Hasta Bazli)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    print(f"\n[1/4] segmented_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"segmented_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim04_segmentasyon.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    segmente = df[df["segmented"] == True].copy()
    print(f"      Segmente kayit: {len(segmente)}")

    # strat_fold kontrolu
    if "strat_fold" not in segmente.columns:
        raise ValueError("strat_fold sutunu bulunamadi! PTB-XL metadata eksik.")

    print(f"      Fold dagilimi:")
    fold_dag = segmente["strat_fold"].value_counts().sort_index()
    for fold, sayi in fold_dag.items():
        print(f"        Fold {int(fold)}: {sayi} kayit")

    # --- 2. Bolme ---
    print(f"\n[2/4] Hasta bazli bolme uygulanyor...")
    print(f"      Test folds  : 9, 10")
    print(f"      Val fold    : 8")
    print(f"      Train folds : 1-7")

    train_df, val_df, test_df = ptbxl_strat_fold_bolme(segmente)

    print(f"\n      Train : {len(train_df)} kayit ({len(train_df)/len(segmente)*100:.1f}%)")
    print(f"      Val   : {len(val_df)} kayit ({len(val_df)/len(segmente)*100:.1f}%)")
    print(f"      Test  : {len(test_df)} kayit ({len(test_df)/len(segmente)*100:.1f}%)")

    # --- 3. Veri sizintisi kontrolu ---
    print(f"\n[3/4] Veri sizintisi kontrolu...")

    train_hastalar = set(train_df["patient_id"].unique())
    val_hastalar = set(val_df["patient_id"].unique())
    test_hastalar = set(test_df["patient_id"].unique())

    train_val_overlap = train_hastalar & val_hastalar
    train_test_overlap = train_hastalar & test_hastalar
    val_test_overlap = val_hastalar & test_hastalar

    print(f"      Train-Val hasta cakismasi  : {len(train_val_overlap)}")
    print(f"      Train-Test hasta cakismasi : {len(train_test_overlap)}")
    print(f"      Val-Test hasta cakismasi   : {len(val_test_overlap)}")

    if len(train_test_overlap) == 0 and len(train_val_overlap) == 0:
        print(f"      [OK] Veri sizintisi YOK!")
    else:
        print(f"      [UYARI] Veri sizintisi tespit edildi!")

    # --- 4. Sinif agirliklari ---
    print(f"\n[4/4] Sinif agirliklari hesaplaniyor ve kaydediliyor...")

    train_labels = train_df["label"].values
    agirliklar = sinif_agirliklari_hesapla(train_labels)

    print(f"      Sinif agirliklari:")
    for sinif_idx in range(config.NUM_CLASSES):
        sinif_adi = config.LABEL_NAMES.get(sinif_idx, "?")
        print(f"        [{sinif_idx}] {sinif_adi:20s}: {agirliklar[sinif_idx]:.4f}")

    # Kaydet
    agirlik_yolu = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
    np.save(agirlik_yolu, agirliklar)
    print(f"      Kaydedildi: {agirlik_yolu}")

    # Manifestleri kaydet
    train_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    val_yolu = os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv")
    test_yolu = os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv")

    train_df.to_csv(train_yolu)
    val_df.to_csv(val_yolu)
    test_df.to_csv(test_yolu)

    print(f"      train_manifest.csv: {len(train_df)} kayit")
    print(f"      val_manifest.csv  : {len(val_df)} kayit")
    print(f"      test_manifest.csv : {len(test_df)} kayit")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    for isim, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        sinif_dag = subset["label"].value_counts().sort_index()
        print(f"\n  {isim} Seti Sinif Dagilimi ({len(subset)} kayit):")
        for sinif_idx, sayi in sinif_dag.items():
            sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "?")
            oran = sayi / sinif_dag.sum() * 100
            print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 6 tamamlandi. Sonraki adim: adim07_model_mimarisi.py")
    print("=" * 70)

    return train_df, val_df, test_df


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    train, val, test = veri_bolme_pipeline()
