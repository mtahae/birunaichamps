"""
adim07b_wide_features.py — BirunAI: Wide Features Pre-compute
================================================================

Her filtrelenmis sinyal icin 8 boyutlu wide features vektoru hesaplar
ve .npy olarak kaydeder. Bu ozellikler modelin "Wide" kismi icin kullanilir.

Wide Features (8 Boyutlu):
    0: Mean HR     — Ortalama kalp hizi (normalize)
    1: HR Std      — Kalp hizi standart sapmasi
    2: Mean RR     — Ortalama RR araligi
    3: RR Std      — RR duzensizligi (AFIB vs AFL ayrimi icin KRITIK)
    4: NN50        — >50ms farkli ardisik RR sayisi
    5: pNN50       — NN50 / toplam RR orani
    6: Age         — Yas (0-1 arasi normalize)
    7: Gender      — Cinsiyet (Erkek=0, Kadin=1, Bilinmiyor=0.5)

Kullanim:
    python adim07b_wide_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def extract_wide_features_fast(sinyal, age=50.0, gender=0.5, fs=250):
    """
    Hizli wide feature cikarimi (neurokit2 yerine scipy kullanir).
    
    Args:
        sinyal: (12, N) numpy array — filtrelenmis EKG sinyali
        age: Yas
        gender: 0=Erkek, 1=Kadin, 0.5=Bilinmiyor
        fs: Ornekleme frekansi
    
    Returns:
        (8,) float32 numpy array
    """
    features = np.zeros(8, dtype=np.float32)
    
    # Age ve Gender
    features[6] = min(age / 100.0, 1.0)
    features[7] = gender
    
    # Lead II (index 1) uzerinden R-pike tespiti
    try:
        lead_ii = sinyal[1] if sinyal.shape[0] > 1 else sinyal[0]
        
        # R-pike tespiti — scipy find_peaks
        min_distance = int(0.3 * fs)  # Min 0.3sn (max 200 bpm)
        min_height = 0.5 * np.std(lead_ii)
        
        peaks, _ = find_peaks(lead_ii, distance=min_distance, height=min_height)
        
        if len(peaks) >= 3:
            # RR intervalleri (milisaniye)
            rr_intervals = np.diff(peaks) / fs * 1000.0
            
            # HR hesaplama (bpm)
            hr_values = 60000.0 / rr_intervals  # ms -> bpm
            mean_hr = np.mean(hr_values)
            std_hr = np.std(hr_values)
            
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            
            # NN50 ve pNN50
            rr_diffs = np.abs(np.diff(rr_intervals))
            nn50 = np.sum(rr_diffs > 50)
            pnn50 = nn50 / len(rr_intervals) if len(rr_intervals) > 0 else 0.0
            
            # Normalize ve ata
            features[0] = min(mean_hr / 200.0, 1.0)      # Max 200 bpm
            features[1] = min(std_hr / 50.0, 1.0)         # Max 50 bpm std
            features[2] = min(mean_rr / 1500.0, 1.0)      # Max 1500ms
            features[3] = min(std_rr / 300.0, 1.0)        # Max 300ms std
            features[4] = min(nn50 / 30.0, 1.0)           # Max 30
            features[5] = min(pnn50, 1.0)
            
    except Exception:
        pass  # Hata durumunda 0'lar kalir
    
    return features


def _process_one(args):
    """Tek bir sinyal icin wide features hesapla (multiprocessing worker)."""
    ecg_id, sinyal_path, age, gender, fs = args
    try:
        sinyal = np.load(sinyal_path)
        feats = extract_wide_features_fast(sinyal, age=age, gender=gender, fs=fs)
        return ecg_id, feats, None
    except Exception as e:
        return ecg_id, np.zeros(8, dtype=np.float32), str(e)


def wide_features_pipeline():
    """
    Tum train ve val sinyalleri icin wide features hesapla ve kaydet.
    """
    print("=" * 70)
    print("BirunAI -- Adim 7b: Wide Features Pre-Compute")
    print("=" * 70)
    
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    output_dizini = os.path.join(config.PROCESSED_DATA_DIR, "wide_features")
    os.makedirs(output_dizini, exist_ok=True)
    
    # Train ve Val manifestlerini birlesik isle
    manifests = []
    for name in ["train_manifest.csv", "val_manifest.csv", "test_manifest.csv"]:
        path = os.path.join(config.PROCESSED_DATA_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col="ecg_id")
            manifests.append((name, df))
            print(f"  {name}: {len(df)} kayit")
    
    if not manifests:
        raise FileNotFoundError("Manifest dosyalari bulunamadi!")
    
    # Tum kayitlari birlestir
    all_df = pd.concat([df for _, df in manifests])
    total = len(all_df)
    print(f"\n  Toplam islenecek: {total} sinyal")
    
    # Zaten hesaplanmis olanlari atla
    existing = set()
    for f in os.listdir(output_dizini):
        if f.endswith('.npy'):
            existing.add(f.replace('.npy', ''))
    
    if existing:
        print(f"  Zaten hesaplanmis: {len(existing)} (atlaniyor)")
    
    # Islenecek listesi hazirla
    tasks = []
    for ecg_id, row in all_df.iterrows():
        if ecg_id in existing:
            continue
        
        sinyal_path = os.path.join(sinyal_dizini, f"{ecg_id}.npy")
        if not os.path.exists(sinyal_path):
            continue
        
        # Age ve Gender cikarimi
        age = 50.0  # Varsayilan
        gender = 0.5
        if 'age' in row.index:
            try:
                a = float(row['age'])
                if 0 < a < 120:
                    age = a
            except (ValueError, TypeError):
                pass
        if 'sex' in row.index:
            s = str(row.get('sex', '')).upper().strip()
            if s in ('M', 'MALE', '0'):
                gender = 0.0
            elif s in ('F', 'FEMALE', '1'):
                gender = 1.0
        
        tasks.append((ecg_id, sinyal_path, age, gender, config.TARGET_FS))
    
    print(f"  Yeni hesaplanacak: {len(tasks)} sinyal")
    
    if len(tasks) == 0:
        print("  Tum wide features zaten mevcut!")
        return
    
    # Islenme
    error_count = 0
    done = 0
    
    # Sequential processing (daha guvenli Windows'ta)
    for task in tqdm(tasks, desc="  Wide Features"):
        ecg_id, feats, err = _process_one(task)
        if err:
            error_count += 1
        else:
            out_path = os.path.join(output_dizini, f"{ecg_id}.npy")
            np.save(out_path, feats)
            done += 1
    
    print(f"\n  Tamamlanan: {done}")
    print(f"  Hata      : {error_count}")
    print(f"  Kayit yeri: {output_dizini}")
    print("=" * 70)
    print("Adim 7b tamamlandi.")
    print("=" * 70)


if __name__ == "__main__":
    wide_features_pipeline()
