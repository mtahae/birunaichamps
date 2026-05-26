"""
sqi.py — BirunAI EKG: Sinyal Kalitesi Indeksi (Signal Quality Index)
=====================================================================

PDF BOLUM 7'ye tam uyumlu SQI hesaplama modulu.

Her lead icin 0-1 arasi bir kalite skoru hesaplar:
    1. kSQI: Kurtosis (QRS varsa yuksek)
    2. bSQI: Baseline wander (dusuk olmali)
    3. pSQI: QRS band gucu (5-15 Hz)
    4. rSQI: R-peak detection basarisi

Kullanim:
    - Egitimde: SQI < 0.2 olan kayitlar ELENIR
    - Inference'da: SQI agirlikli fusion (lead atilmaz, sadece agirligi duser)
"""

import numpy as np
from scipy.stats import kurtosis
from scipy.signal import butter, filtfilt, welch

def compute_sqi(signal_12lead, fs=250):
    """
    12 lead'in her biri icin SQI hesaplar.
    
    Args:
        signal_12lead: (12, N) veya (N, 12) boyutunda sinyal
        fs: Ornekleme frekansi
    
    Returns:
        sqis: (12,) boyutunda float array, 0-1 arasi
    """
    # Eger (N, 12) formatinda geldiyse transpoze et
    if signal_12lead.shape[0] != 12 and signal_12lead.shape[1] == 12:
        signal_12lead = signal_12lead.T
    
    sqis = []
    
    for lead in range(12):
        s = signal_12lead[lead]
        
        # 1. kSQI: Kurtosis (QRS sinyali varsa kurtosis yuksektir)
        try:
            k = kurtosis(s)
            k_norm = min(abs(k) / 10.0, 1.0)
        except:
            k_norm = 0.0
        
        # 2. bSQI: Baseline wander (dusuk frekansli bozulma)
        try:
            nyq = 0.5 * fs
            b, a = butter(2, 0.5 / nyq, btype='low')
            baseline = filtfilt(b, a, s)
            b_ratio = np.std(baseline) / (np.std(s) + 1e-6)
            b_norm = max(0.0, 1.0 - b_ratio)
        except:
            b_norm = 0.5
        
        # 3. pSQI: QRS band gucu (5-15 Hz)
        try:
            f, psd = welch(s, fs=fs, nperseg=min(256, len(s)))
            qrs_mask = (f >= 5) & (f <= 15)
            total_mask = (f >= 0.5) & (f <= 40)
            qrs_power = np.sum(psd[qrs_mask])
            total_power = np.sum(psd[total_mask]) + 1e-6
            p_norm = qrs_power / total_power
        except:
            p_norm = 0.0
        
        # 4. rSQI: R-peak detection basarisi
        try:
            import neurokit2 as nk
            _, info = nk.ecg_peaks(s, sampling_rate=fs)
            r_peaks = info["ECG_R_Peaks"]
            duration_min = len(s) / fs / 60.0
            hr = len(r_peaks) / duration_min if duration_min > 0 else 0
            r_norm = 1.0 if 40 <= hr <= 150 else 0.3
        except:
            r_norm = 0.0
        
        # Agirlikli ortalama
        sqi = 0.3 * k_norm + 0.3 * b_norm + 0.2 * p_norm + 0.2 * r_norm
        sqis.append(sqi)
    
    return np.array(sqis, dtype=np.float32)


def filter_by_sqi(manifest_df, sinyal_dizini, fs=250, threshold=0.2):
    """
    Manifest'teki tum kayitlarin SQI'sini hesaplar.
    Ortalama SQI < threshold olan kayitlari isaretler.
    
    Args:
        manifest_df: ecg_id ve signal_path iceren DataFrame
        sinyal_dizini: .npy dosyalarinin bulundugu dizin
        fs: Ornekleme frekansi
        threshold: SQI esik degeri
    
    Returns:
        DataFrame: 'sqi_mean' ve 'sqi_pass' sutunlari eklenmis DataFrame
    """
    import os
    from tqdm import tqdm
    
    sqi_means = []
    
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="SQI Hesaplama"):
        ecg_id = row['ecg_id']
        npy_path = os.path.join(sinyal_dizini, f"{ecg_id}.npy")
        
        try:
            if os.path.exists(npy_path):
                signal = np.load(npy_path)
                sqis = compute_sqi(signal, fs=fs)
                sqi_means.append(np.mean(sqis))
            else:
                sqi_means.append(0.0)
        except:
            sqi_means.append(0.0)
    
    manifest_df = manifest_df.copy()
    manifest_df['sqi_mean'] = sqi_means
    manifest_df['sqi_pass'] = manifest_df['sqi_mean'] >= threshold
    
    passed = manifest_df['sqi_pass'].sum()
    failed = len(manifest_df) - passed
    print(f"SQI Filtre: {passed} gecti, {failed} elendi (threshold={threshold})")
    
    return manifest_df


if __name__ == "__main__":
    # Test
    print("SQI Test")
    dummy = np.random.randn(12, 2500).astype(np.float32)
    sqis = compute_sqi(dummy, fs=250)
    print(f"SQI per lead: {sqis}")
    print(f"Mean SQI: {np.mean(sqis):.3f}")
