"""
threshold_opt.py — BirunAI EKG: Sınıf Bazlı Optimal Eşik Bulma (Threshold Optimization)
======================================================================================

Derin ogrenme modellerinde varsayilan 0.5 esik degeri, dengesiz EKG veri setlerinde 
(ozellikle AFL gibi nadir siniflarda) Macro F1 skorunu buyuk olcude dusurur.

PhysioNet 2020 Sampiyonlarinin (Takim 2 ve Takim 5) yaklasimi:
Her bir sinif (Normal, AFIB, AFL, LBBB, RBBB) icin bagimsiz olarak
validation seti uzerinde Grid-Search yapilarak F1 skorunu maksimize eden
optimal esik degerleri (or: AFIB=0.35, AFL=0.20, RBBB=0.60) bulunur.
"""

import numpy as np
from sklearn.metrics import f1_score
import json

def find_optimal_thresholds(y_true, y_prob, num_classes=5):
    """
    Validation setindeki gercek etiketler (y_true) ve modelin ciktisi olan 
    olasiliklar (y_prob - sigmoid sonrasi) kullanilarak, her sinif icin
    F1 skorunu maksimize eden esik (threshold) degeri bulunur.
    
    Args:
        y_true: (N,) boyutlu ground truth (1D array)
        y_prob: (N, num_classes) boyutlu model tahmin olasiliklari (Sigmoid'den gecmis)
        num_classes: Sinif sayisi (default: 5)
        
    Returns:
        optimal_thresholds: list of float (Her sinif icin ideal esik)
        best_f1s: list of float (Her sinif icin elde edilen en iyi F1)
    """
    
    thresholds = np.arange(0.1, 0.95, 0.05)
    optimal_thresholds = [0.5] * num_classes
    best_f1s = [0.0] * num_classes
    
    # y_true (N,) seklindeyse bunu one-hot (N, C) sekline cevirelim
    y_true_onehot = np.zeros_like(y_prob)
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1.0
        
    for c in range(num_classes):
        best_f1_c = 0.0
        best_th_c = 0.5
        
        for th in thresholds:
            # Sadece bu sinif (c) icin th'yi gecenlere 1, gecmeyenlere 0 diyoruz
            y_pred_c = (y_prob[:, c] >= th).astype(int)
            f1_c = f1_score(y_true_onehot[:, c], y_pred_c, zero_division=0)
            
            if f1_c > best_f1_c:
                best_f1_c = f1_c
                best_th_c = th
                
        optimal_thresholds[c] = round(best_th_c, 2)
        best_f1s[c] = round(best_f1_c, 4)
        
    print(f"\n--- Eşik Optimizasyonu Sonuçları ---")
    class_names = ["Normal", "AFIB", "AFL", "LBBB", "RBBB"]
    for c in range(num_classes):
        print(f"{class_names[c]:6s} -> Optimal Esik: {optimal_thresholds[c]:.2f} (F1: {best_f1s[c]:.4f})")
        
    macro_f1 = np.mean(best_f1s)
    print(f"Eşik Optimizasyonu Sonrası Macro F1: {macro_f1:.4f}")
    print("-" * 36)
    
    return optimal_thresholds, best_f1s

def apply_thresholds(y_prob, thresholds):
    """
    Hesaplanmis olan esik degerleri ile test setine tahmin yapar.
    Olasiligi kendi esigini (threshold[c]) en cok gecen sinif 1. secilir.
    
    Args:
        y_prob: (N, C) boyutlu model olasiliklari
        thresholds: (C,) boyutlu esik listesi
        
    Returns:
        y_pred: (N,) tahmin edilen siniflar
    """
    y_pred = np.zeros(y_prob.shape[0], dtype=int)
    
    for i in range(y_prob.shape[0]):
        probs = y_prob[i]
        margins = np.zeros_like(probs)
        for c in range(len(thresholds)):
            margins[c] = probs[c] - thresholds[c]
            
        # Margin'i en yuksek olan sinif (Esigi en cok asan)
        y_pred[i] = np.argmax(margins)
        
    return y_pred

if __name__ == "__main__":
    # Test
    np.random.seed(42)
    dummy_y_true = np.random.randint(0, 5, 1000)
    dummy_y_prob = np.random.rand(1000, 5)
    
    # Bazi classlar icin olasiliklari arttiralim ki dogru gibi gozuksun
    for i in range(1000):
        dummy_y_prob[i, dummy_y_true[i]] += 0.5
        dummy_y_prob[i] = np.clip(dummy_y_prob[i], 0, 1)
        
    th, f1s = find_optimal_thresholds(dummy_y_true, dummy_y_prob)
    preds = apply_thresholds(dummy_y_prob, th)
    print("Test basarili.")
