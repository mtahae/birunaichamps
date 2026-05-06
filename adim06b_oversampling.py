"""
adim06b_smote.py — BirunAI EKG: Adim 6b – SMOTE Oversampling
==============================================================

Bu modul, egitim setindeki Ritim Bozuklugu (sinif 1) kayitlarini
SMOTE (Synthetic Minority Over-sampling Technique) ile coğaltir.

Strateji:
    - Sadece TRAIN setine uygulanir (val/test dokunulmaz)
    - PCA-SMOTE: Yuksek boyutlu sinyal verisini (12*2500=30000) PCA ile
      dusuk boyutlu uzaya indirir, SMOTE uygular, geri donusturur
    - Hedef: Ritim sinifini SMOTE_RITIM_HEDEF kadar arttirir
      (Normal/Iletim seviyesine getirmez — fazla sentetik veri kaliteyi dusurebilir)

Ek olarak sinyal-seviyesi augmentasyon da uygulanir:
    - Time Shift (zaman kaydirma)
    - Gaussian Noise (gurultu ekleme)
    - Amplitude Scaling (genlik olcekleme)

Ciktilar:
    - outputs/processed_data/segmented_signals/ (yeni sentetik .npy dosyalari)
    - outputs/processed_data/train_manifest_smote.csv

Kullanim:
    python adim06b_smote.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# SINYAL AUGMENTASYON FONKSIYONLARI
# =============================================================================

def time_shift(sinyal, max_shift=25):
    """Sinyali rastgele sağa/sola kaydırır (circular shift)."""
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(sinyal, shift, axis=1)


def gaussian_noise(sinyal, std=0.05):
    """Sinyale Gaussian gürültü ekler."""
    noise = np.random.normal(0, std, sinyal.shape).astype(sinyal.dtype)
    return sinyal + noise


def amplitude_scale(sinyal, min_scale=0.85, max_scale=1.15):
    """Sinyal genliğini rastgele ölçekler."""
    scale = np.random.uniform(min_scale, max_scale)
    return sinyal * scale


def augment_signal(sinyal):
    """Üç augmentasyon tekniğinden rastgele bir kombinasyon uygular."""
    aug = sinyal.copy()
    if np.random.random() < 0.5:
        aug = time_shift(aug)
    if np.random.random() < 0.5:
        aug = gaussian_noise(aug)
    if np.random.random() < 0.5:
        aug = amplitude_scale(aug)
    return aug


# =============================================================================
# SMOTE PIPELINE
# =============================================================================

def smote_pipeline():
    print("=" * 70)
    print("BirunAI — Adim 6b: SMOTE Oversampling")
    print("=" * 70)

    # --- 1. Train manifest oku ---
    train_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    print(f"\n[1/5] train_manifest.csv okunuyor...")

    if not os.path.exists(train_yolu):
        raise FileNotFoundError(f"Bulunamadi: {train_yolu}")

    train_df = pd.read_csv(train_yolu)
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")

    # Mevcut dagilim
    sinif_dag = train_df['label'].value_counts().sort_index()
    print(f"      Train kayit: {len(train_df)}")
    for s, n in sinif_dag.items():
        print(f"        [{int(s)}] {config.LABEL_NAMES.get(int(s),'?'):20s}: {n}")

    hedef_sinif = 1  # Ritim Bozuklugu
    normal_sayisi = sinif_dag.get(0, 0)
    ritim_sayisi = sinif_dag.get(1, 0)
    iletim_sayisi = sinif_dag.get(2, 0)

    # Hedef: Ritim sayisini 2 katina cikar, maksimum 15000 ile sinirla.
    # Normal/Iletim kadar cikarmaya GEREK YOK — asiri sentetik veri kaliteyi
    # dusurebilir. %25-30 oraninda tutmak yeterli.
    SMOTE_HEDEF_CAP = 15000
    hedef_sayi = min(ritim_sayisi * 2, SMOTE_HEDEF_CAP)

    if ritim_sayisi >= hedef_sayi:
        print(f"\n      Ritim sinifi zaten yeterli ({ritim_sayisi} >= {hedef_sayi}).")
        print("      SMOTE uygulanmayacak.")
        return train_df

    uretilecek = hedef_sayi - ritim_sayisi
    print(f"\n      Mevcut Ritim sayisi: {ritim_sayisi}")
    print(f"      Hedef  Ritim sayisi: {hedef_sayi}")
    print(f"      Uretilecek sentetik : {uretilecek}")

    # --- 2. Tum egitim sinyallerini yukle ---
    print(f"\n[2/5] Egitim sinyalleri yukleniyor...")

    sinyaller = []
    etiketler = []
    ecg_idler = []

    for idx, row in tqdm(train_df.iterrows(), total=len(train_df),
                          desc="      Yukleme", ncols=80):
        dosya = os.path.join(sinyal_dizini, f"{row['ecg_id']}.npy")
        try:
            sinyal = np.load(dosya)
            sinyaller.append(sinyal.flatten())  # (12, 2500) -> (30000,)
            etiketler.append(int(row['label']))
            ecg_idler.append(row['ecg_id'])
        except Exception:
            continue

    X = np.array(sinyaller, dtype=np.float32)
    y = np.array(etiketler, dtype=np.int32)
    print(f"      Yuklenen: {X.shape[0]} kayit, {X.shape[1]} ozellik")

    # --- 3. PCA boyut indirgeme ---
    print(f"\n[3/5] PCA boyut indirgeme...")

    n_components = min(256, X.shape[0] - 1, X.shape[1])
    print(f"      PCA bilesenleri: {n_components}")

    pca = PCA(n_components=n_components, random_state=config.SEED)
    X_pca = pca.fit_transform(X)
    varyans_orani = pca.explained_variance_ratio_.sum()
    print(f"      Aciklanan varyans : {varyans_orani*100:.1f}%")
    print(f"      PCA cikti boyutu : {X_pca.shape}")

    # --- 4. SMOTE uygula ---
    print(f"\n[4/5] SMOTE uygulanıyor...")

    smote_hedef = {
        0: int(np.sum(y == 0)),  # Normal degismesin
        1: hedef_sayi,            # Ritim arttir
        2: int(np.sum(y == 2)),  # Iletim degismesin
    }
    print(f"      SMOTE hedef: {smote_hedef}")

    smote = SMOTE(
        sampling_strategy=smote_hedef,
        k_neighbors=min(5, ritim_sayisi - 1),
        random_state=config.SEED
    )

    X_smote, y_smote = smote.fit_resample(X_pca, y)
    print(f"      SMOTE oncesi: {len(X_pca)} | SMOTE sonrasi: {len(X_smote)}")

    # Yeni sentetik ornekleri bul (sondakiler)
    n_original = len(X_pca)
    n_sentetik = len(X_smote) - n_original

    if n_sentetik > 0:
        X_sentetik_pca = X_smote[n_original:]
        y_sentetik = y_smote[n_original:]

        # PCA inverse transform
        X_sentetik_original = pca.inverse_transform(X_sentetik_pca)

        print(f"      Sentetik uretilen: {n_sentetik} kayit")

        # --- 5. Sentetik sinyalleri kaydet + augment ---
        print(f"\n[5/5] Sentetik sinyaller kaydediliyor...")

        yeni_kayitlar = []
        for i in tqdm(range(n_sentetik), desc="      Kaydetme", ncols=80):
            ecg_id = f"smote_ritim_{i:06d}"
            sinyal = X_sentetik_original[i].reshape(config.NUM_LEADS, config.TARGET_LENGTH)

            # Hafif augmentasyon ekle (daha gercekci hale getir)
            sinyal = augment_signal(sinyal.astype(np.float32))

            # Kaydet
            cikti = os.path.join(sinyal_dizini, f"{ecg_id}.npy")
            np.save(cikti, sinyal.astype(np.float32))

            yeni_kayitlar.append({
                'ecg_id': ecg_id,
                'dataset_source': 'smote_synthetic',
                'signal_path': '',
                'label': int(y_sentetik[i]),
                'original_fs': config.TARGET_FS,
                'num_samples': config.TARGET_LENGTH,
                'num_leads': config.NUM_LEADS,
                'age': 'Synthetic',
                'sex': 'Synthetic',
                'dx_codes': '',
            })

        sentetik_df = pd.DataFrame(yeni_kayitlar)
    else:
        print(f"      Sentetik uretim gerekmedi.")
        sentetik_df = pd.DataFrame()

    # SMOTE sonrasi train manifest
    if not sentetik_df.empty:
        # Orijinal train manifest sutunlariyla uyumlu hale getir
        ortak_sutunlar = list(set(train_df.columns) & set(sentetik_df.columns))
        train_smote = pd.concat([
            train_df[ortak_sutunlar],
            sentetik_df[ortak_sutunlar]
        ], ignore_index=True)
    else:
        train_smote = train_df.copy()

    smote_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest_smote.csv")
    train_smote.to_csv(smote_yolu, index=False)
    print(f"\n      Kaydedildi: {smote_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET — SMOTE Sonrasi Egitim Seti Dagilimi")
    print("=" * 70)

    sinif_dag_sonra = train_smote['label'].value_counts().sort_index()
    for s, n in sinif_dag_sonra.items():
        sinif_adi = config.LABEL_NAMES.get(int(s), "?")
        oran = n / sinif_dag_sonra.sum() * 100
        print(f"    [{int(s)}] {sinif_adi:20s}: {n:6d} ({oran:5.1f}%)")

    print(f"\n  Toplam egitim kaydi: {len(train_smote)}")
    print(f"  Sentetik uretilen  : {n_sentetik}")

    # Yeni class weights
    sinif_dag_yeni = train_smote['label'].value_counts().sort_index()
    toplam = sinif_dag_yeni.sum()
    n_sinif = len(sinif_dag_yeni)
    class_weights = np.array([
        toplam / (n_sinif * sinif_dag_yeni[i])
        for i in range(n_sinif)
    ], dtype=np.float32)
    weights_yolu = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
    np.save(weights_yolu, class_weights)
    print(f"  Yeni class weights: {class_weights}")

    print("\n" + "=" * 70)
    print("Adim 6b tamamlandi. Sonraki: adim08_egitim.py")
    print("=" * 70)

    return train_smote


if __name__ == "__main__":
    smote_pipeline()
