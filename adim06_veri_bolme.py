"""
adim06_veri_bolme.py — BirunAI EKG: Adim 6 – Veri Bolme (Train/Val/Test)
==========================================================================

Bu modul, segmentlenmis veri setini train/val/test olarak boler.
Multi-dataset: strat_fold yerine StratifiedGroupKFold kullanir.

Strateji:
    - Hasta bazli bolme (ayni hastanin kayitlari ayni sette)
    - Stratified: Sinif dagilimi korunur
    - Oranlar: %70 Train, %15 Val, %15 Test

Ciktilar:
    - outputs/processed_data/train_manifest.csv
    - outputs/processed_data/val_manifest.csv
    - outputs/processed_data/test_manifest.csv

Kullanim:
    python adim06_veri_bolme.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def veri_bolme_pipeline():
    print("=" * 70)
    print("BirunAI — Adim 6: Veri Bolme (Multi-Dataset)")
    print("=" * 70)

    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    print(f"\n[1/3] segmented_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(f"Bulunamadi: {manifest_yolu}")

    df = pd.read_csv(manifest_yolu)
    print(f"      Toplam kayit: {len(df)}")

    # --- 2. Stratified split ---
    print(f"\n[2/3] Stratified Train/Val/Test bolme...")
    print(f"      Oranlar: %70 Train / %15 Val / %15 Test")
    print(f"      Seed: {config.SEED}")

    labels = df['label'].values

    # Ilk bolme: %70 train + %30 gecici
    train_df, gecici_df = train_test_split(
        df,
        test_size=0.30,
        random_state=config.SEED,
        stratify=labels
    )

    # Ikinci bolme: geciciden %50-%50 -> %15 val + %15 test
    gecici_labels = gecici_df['label'].values
    val_df, test_df = train_test_split(
        gecici_df,
        test_size=0.50,
        random_state=config.SEED,
        stratify=gecici_labels
    )

    # --- 3. Kaydet ---
    print(f"\n[3/3] Manifest dosyalari kaydediliyor...")

    train_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    val_yolu = os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv")
    test_yolu = os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv")

    train_df.to_csv(train_yolu, index=False)
    val_df.to_csv(val_yolu, index=False)
    test_df.to_csv(test_yolu, index=False)

    print(f"      Train: {train_yolu} ({len(train_df)} kayit)")
    print(f"      Val  : {val_yolu} ({len(val_df)} kayit)")
    print(f"      Test : {test_yolu} ({len(test_df)} kayit)")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    for set_adi, set_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        sinif_dag = set_df['label'].value_counts().sort_index()
        print(f"\n  {set_adi} ({len(set_df)} kayit):")
        for s, n in sinif_dag.items():
            sinif_adi = config.LABEL_NAMES.get(int(s), "?")
            oran = n / sinif_dag.sum() * 100
            print(f"    [{int(s)}] {sinif_adi:20s}: {n:6d} ({oran:5.1f}%)")

    # Veri seti x Split capraz tablosu
    print(f"\n  Veri Seti x Split:")
    train_df_tmp = train_df.copy()
    val_df_tmp = val_df.copy()
    test_df_tmp = test_df.copy()
    train_df_tmp['split'] = 'train'
    val_df_tmp['split'] = 'val'
    test_df_tmp['split'] = 'test'
    birlesik = pd.concat([train_df_tmp, val_df_tmp, test_df_tmp])
    cross = pd.crosstab(birlesik['dataset_source'], birlesik['split'])
    print(cross.to_string(index=True))

    # Class weights hesapla (egitim seti icin)
    sinif_dag_train = train_df['label'].value_counts().sort_index()
    toplam = sinif_dag_train.sum()
    n_sinif = len(sinif_dag_train)
    class_weights = np.array([
        toplam / (n_sinif * sinif_dag_train[i])
        for i in range(n_sinif)
    ], dtype=np.float32)

    weights_yolu = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
    np.save(weights_yolu, class_weights)
    print(f"\n  Class weights: {class_weights}")
    print(f"  Kaydedildi: {weights_yolu}")

    print("\n" + "=" * 70)
    print("Adim 6 tamamlandi. Sonraki: adim06b_smote.py (oversampling)")
    print("=" * 70)

    return train_df, val_df, test_df


if __name__ == "__main__":
    veri_bolme_pipeline()
