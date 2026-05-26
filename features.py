"""
features.py — BirunAI EKG Siniflandirma: "Wide Features" Cikarimi
===================================================================

Bu modul, NeuroKit2 kullanarak ham EKG sinyalinden (ozellikle Lead II'den)
fizyolojik ozellikleri (Kalp atis hizi, RR araligi varyansi vb.) cikarir.
Ayrice .hea dosyasindan yas ve cinsiyet bilgilerini alarak 
CardioFusion-5'in "Wide" kismi icin 8 boyutlu bir vektor olusturur.

Wide Features (8 Boyutlu):
1. Mean HR (Ortalama Kalp Hizi)
2. HR Std (Kalp Hizi Standart Sapmasi)
3. Mean RR Interval
4. RR Std (AFIB/AFL ayrimi icin en kritik ozellik)
5. NN50 (Birbirini izleyen RR araliklari arasindaki 50ms'den buyuk farklarin sayisi)
6. pNN50 (NN50'nin tum RR araliklarina orani)
7. Age (Yas - 0-100 arasi, 100'e bolunerek normalize edilir)
8. Gender (Erkek=0, Kadin=1)
"""

import os
import numpy as np
import wfdb
import neurokit2 as nk

def extract_wide_features(signal, hea_path, sampling_rate=250):
    """
    signal: (12, N) boyutunda EKG sinyali. Genelde Lead II (index 1) kullanilir.
    hea_path: WFDB .hea dosyasinin tam yolu (yas ve cinsiyet icin).
    
    Dondurur: (8,) boyutunda float32 numpy array.
    """
    features = np.zeros(8, dtype=np.float32)
    
    # --- 1. Yas ve Cinsiyet Cikarimi ---
    age = 50.0  # Varsayilan
    gender = 0.5  # Varsayilan (Bilinmiyor)
    
    try:
        if os.path.exists(hea_path):
            with open(hea_path, 'r') as f:
                for line in f:
                    if line.startswith('#Age:'):
                        val = line.split(':')[1].strip()
                        if val.isdigit():
                            age = float(val)
                    elif line.startswith('#Sex:'):
                        val = line.split(':')[1].strip().upper()
                        if val == 'M' or val == 'MALE':
                            gender = 0.0
                        elif val == 'F' or val == 'FEMALE':
                            gender = 1.0
    except Exception:
        pass
        
    features[6] = age / 100.0  # Normalize (0-1)
    features[7] = gender
    
    # --- 2. NeuroKit2 ile RR ve HR Ozellikleri (Lead II = index 1) ---
    lead_ii = signal[1]
    
    try:
        # R-tepelerini tespit et
        _, rpeaks = nk.ecg_peaks(lead_ii, sampling_rate=sampling_rate)
        r_indices = rpeaks["ECG_R_Peaks"]
        
        if len(r_indices) > 2:
            # HR ve HRV (Heart Rate Variability) hesaplama
            hrv_time = nk.hrv_time(r_indices, sampling_rate=sampling_rate, show=False)
            
            # DataFrame'den degerleri al (Varsayilan degerler 0)
            mean_hr = hrv_time["HRV_MeanNN"].values[0] if "HRV_MeanNN" in hrv_time else 0.0
            sdnn = hrv_time["HRV_SDNN"].values[0] if "HRV_SDNN" in hrv_time else 0.0
            
            # RR Intervalleri (milisaniye cinsinden)
            rr_intervals = np.diff(r_indices) / sampling_rate * 1000.0
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            
            # NN50 ve pNN50
            rr_diffs = np.abs(np.diff(rr_intervals))
            nn50 = np.sum(rr_diffs > 50)
            pnn50 = nn50 / len(rr_intervals) if len(rr_intervals) > 0 else 0.0
            
            # Features atama ve normalize etme (Yaklasik araliklara gore 0-1 seviyesine)
            features[0] = min(mean_hr / 150.0, 1.0) # Ortalama max 150 bpm kabul edilir
            features[1] = min(sdnn / 100.0, 1.0)
            features[2] = min(mean_rr / 1500.0, 1.0)
            features[3] = min(std_rr / 200.0, 1.0) # Duzensizlik!
            features[4] = min(nn50 / 20.0, 1.0)
            features[5] = pnn50
            
    except Exception as e:
        # Sinyal cok bozuksa NeuroKit hata verebilir, degerler 0 kalir.
        pass
        
    return features

if __name__ == "__main__":
    # Test
    print("Wide Features Test")
    dummy_signal = np.random.randn(12, 2500)
    feats = extract_wide_features(dummy_signal, "dummy.hea")
    print("Features (Random Signal):", feats)
