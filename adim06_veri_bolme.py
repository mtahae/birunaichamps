"""
adim06_veri_bolme.py — BirunAI EKG: Adim 6 – Veri Bolme (Train/Val/Test) ve DANN Domain Atamasi
=============================================================================================

CardioFusion-5 Mimarisi (2. Asama) İcin Guncellenmistir.

Strateji:
    - TEKNOFEST verisi (varsa) onceden tanimli split'lere uyar. (Ogrenci tarafindan hazirlanmistir)
    - Internet verileri %70 Train, %15 Val, %15 Test olarak bolunur.
    - DANN (Domain-Adversarial Neural Network) modulu icin her dataset kaynagina 'domain_id' atanir:
        0: TEKNOFEST
        1: cpsc_2018
        2: ptb_xl
        3: georgia
        4: ecg_arrhythmia
        ...
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# DOMAIN KODLAMASI (DANN Modulu Icin)
DOMAIN_MAP = {
    "TEKNOFEST": 0,
    "cpsc_2018": 1,
    "cpsc_2018_extra": 1,
    "ptb_xl": 2,
    "georgia": 3,
    "ecg_arrhythmia": 4
}

def veri_bolme_pipeline():
    print("=" * 70)
    print("BirunAI — Adim 6: Veri Bolme ve Domain Atamasi (Multi-Dataset)")
    print("=" * 70)

    # 1. Manifest Okunuyor (adim00 ve adim02'den uretilen filtrelenmis manifest)
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "filtered_manifest.csv")
    if not os.path.exists(manifest_yolu):
        print(f"[UYARI] {manifest_yolu} bulunamadi, fallback olarak unified_manifest.csv deneniyor...")
        manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "unified_manifest.csv")
        
    df = pd.read_csv(manifest_yolu)
    
    # Eger 'filtered' sutunu varsa ve bazi kayitlar filtreden gecemediyse onlari cikar
    if 'filtered' in df.columns:
        df = df[df['filtered'] == True].copy()
        
    print(f"Toplam Gecerli Kayit: {len(df)}")

    # 2. DANN Domain ID Atamasi
    df['domain_id'] = df['dataset_source'].map(lambda x: DOMAIN_MAP.get(x, 5))

    # 3. Veriyi TEKNOFEST ve INTERNET olarak ikiye ayiralim
    df_tekno = df[df['dataset_source'] == 'TEKNOFEST'].copy()
    df_inter = df[df['dataset_source'] != 'TEKNOFEST'].copy()
    
    train_list = []
    val_list = []
    test_list = []

    # TEKNOFEST Bolmesi
    if len(df_tekno) > 0:
        # Eger hazir split csv varsa onu kullan (Yarisma formati)
        # Yoksa 70-15-15 bol
        tekno_labels = df_tekno['label'].values
        t_train, t_temp = train_test_split(df_tekno, test_size=0.30, random_state=config.SEED, stratify=tekno_labels)
        t_val, t_test = train_test_split(t_temp, test_size=0.50, random_state=config.SEED, stratify=t_temp['label'].values)
        
        train_list.append(t_train)
        val_list.append(t_val)
        test_list.append(t_test)
        print(f"TEKNOFEST eklendi: {len(t_train)} Train, {len(t_val)} Val, {len(t_test)} Test")

    # Internet Bolmesi
    if len(df_inter) > 0:
        # Internet verisinin TAMAMINI egitim (Karma Phase) setine ekliyoruz!
        # Amacimiz TEKNOFEST oldugu icin Val/Test setlerini Internet verisiyle kirletmiyoruz.
        train_list.append(df_inter)
        print(f"Internet eklendi: {len(df_inter)} Train, 0 Val, 0 Test")

    # Birlestirme
    train_df = pd.concat(train_list).sample(frac=1, random_state=config.SEED).reset_index(drop=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list).sample(frac=1, random_state=config.SEED).reset_index(drop=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list).sample(frac=1, random_state=config.SEED).reset_index(drop=True) if test_list else pd.DataFrame()

    # 4. Kaydet
    train_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    val_yolu = os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv")
    test_yolu = os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv")

    train_df.to_csv(train_yolu, index=False)
    val_df.to_csv(val_yolu, index=False)
    test_df.to_csv(test_yolu, index=False)

    print(f"\n[BASARILI] Bolunme Tamamlandi.")
    print(f"Train : {len(train_df)}")
    print(f"Val   : {len(val_df)}")
    print(f"Test  : {len(test_df)}")
    
    # Class weights hesapla (egitim seti icin - Focal Loss / Weighted Loss icin)
    if not train_df.empty:
        sinif_dag_train = train_df['label'].value_counts().sort_index()
        toplam = sinif_dag_train.sum()
        n_sinif = len(config.LABEL_NAMES)
        
        class_weights = np.zeros(n_sinif, dtype=np.float32)
        for i in range(n_sinif):
            if i in sinif_dag_train:
                # Invers Frekans Ağırlıklandırması (AFL gibi azinlik siniflara agir ceza)
                class_weights[i] = toplam / (n_sinif * sinif_dag_train[i])
            else:
                class_weights[i] = 1.0
                
        weights_yolu = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
        np.save(weights_yolu, class_weights)
        print(f"\nClass weights (Loss icin): {class_weights}")

    # --- Train istatistikleri hesapla (Z-Score icin) ---
    # PDF BOLUM 1 KRITIK: "mu ve sigma SADECE train setinden hesaplanir"
    if not train_df.empty:
        print(f"\n[Z-SCORE] Train istatistikleri hesaplaniyor (lead-wise)...")
        sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
        
        lead_sums = np.zeros(12, dtype=np.float64)
        lead_sq_sums = np.zeros(12, dtype=np.float64)
        total_samples = 0
        
        for _, row in train_df.iterrows():
            ecg_id = row['ecg_id']
            npy_path = os.path.join(sinyal_dizini, f"{ecg_id}.npy")
            if os.path.exists(npy_path):
                signal = np.load(npy_path)  # (12, 2500)
                lead_sums += signal.sum(axis=1)
                lead_sq_sums += (signal ** 2).sum(axis=1)
                total_samples += signal.shape[1]
        
        if total_samples > 0:
            train_mean = lead_sums / total_samples  # (12,)
            train_std = np.sqrt(lead_sq_sums / total_samples - train_mean ** 2)  # (12,)
            train_std = np.maximum(train_std, 1e-6)  # Bolu sifir korunmasi
            
            stats_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_stats.npz")
            np.savez(stats_yolu, mean=train_mean.astype(np.float32), std=train_std.astype(np.float32))
            print(f"  Kaydedildi: {stats_yolu}")
            print(f"  Mean (per lead): {np.round(train_mean, 4)}")
            print(f"  Std  (per lead): {np.round(train_std, 4)}")

    return train_df, val_df, test_df

if __name__ == "__main__":
    veri_bolme_pipeline()
