"""
threshold_opt.py — BirunAI EKG: Sınıf Bazlı Optimal Eşik Bulma (Threshold Optimization)
======================================================================================

Nelder-Mead optimizasyon algoritması ile her sınıf için F1 skorunu maksimize eden
optimal eşik değerleri bulunur. Grid Search'ten çok daha hassas sonuç verir.

Golden Path (model_architecture_and_training_report.md Bölüm 7):
  Nelder-Mead ile eşik optimizasyonu tek başına Macro F1'i %1-2 artırmıştır.
"""

import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import json


def find_optimal_thresholds(y_true, y_prob, num_classes=5):
    """
    Nelder-Mead optimizasyonu ile her sınıf için F1 skorunu maksimize eden
    optimal eşik değerleri bulunur.
    
    Args:
        y_true: (N,) boyutlu ground truth (1D array)
        y_prob: (N, num_classes) boyutlu model tahmin olasılıkları
        num_classes: Sınıf sayısı (default: 5)
        
    Returns:
        optimal_thresholds: list of float (Her sınıf için ideal eşik)
        best_f1s: list of float (Her sınıf için elde edilen en iyi F1)
    """
    
    # y_true (N,) -> one-hot (N, C)
    y_true_onehot = np.zeros_like(y_prob)
    for i in range(len(y_true)):
        y_true_onehot[i, y_true[i]] = 1.0
    
    optimal_thresholds = [0.5] * num_classes
    best_f1s = [0.0] * num_classes
    
    for c in range(num_classes):
        # Negatif F1 minimize ederek F1'i maximize ediyoruz
        def neg_f1(th):
            th_val = th[0]
            if th_val < 0.01 or th_val > 0.99:
                return 1.0  # ceza
            y_pred_c = (y_prob[:, c] >= th_val).astype(int)
            f1_c = f1_score(y_true_onehot[:, c], y_pred_c, zero_division=0)
            return -f1_c
        
        # Nelder-Mead ile optimize et — birden fazla baslangic noktasi dene
        best_f1_c = 0.0
        best_th_c = 0.5
        
        for start_th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            result = minimize(neg_f1, x0=[start_th], method='Nelder-Mead',
                            options={'xatol': 0.001, 'fatol': 1e-6, 'maxiter': 200})
            if -result.fun > best_f1_c:
                best_f1_c = -result.fun
                best_th_c = float(np.clip(result.x[0], 0.01, 0.99))
        
        # Ek olarak ince grid search (Nelder-Mead'in cevresinde)
        # 0.01 hassasiyetle arama
        fine_range = np.arange(max(0.01, best_th_c - 0.15), min(0.99, best_th_c + 0.15), 0.01)
        for th in fine_range:
            y_pred_c = (y_prob[:, c] >= th).astype(int)
            f1_c = f1_score(y_true_onehot[:, c], y_pred_c, zero_division=0)
            if f1_c > best_f1_c:
                best_f1_c = f1_c
                best_th_c = float(th)
        
        optimal_thresholds[c] = round(best_th_c, 3)
        best_f1s[c] = round(best_f1_c, 4)
    
    print(f"\n--- Eşik Optimizasyonu Sonuçları (Nelder-Mead + Fine Grid) ---")
    class_names = ["Normal", "AFIB", "AFL", "LBBB", "RBBB"]
    for c in range(num_classes):
        print(f"{class_names[c]:6s} -> Optimal Esik: {optimal_thresholds[c]:.3f} (F1: {best_f1s[c]:.4f})")
        
    macro_f1 = np.mean(best_f1s)
    print(f"Eşik Optimizasyonu Sonrası Macro F1: {macro_f1:.4f}")
    print("-" * 56)
    
    return optimal_thresholds, best_f1s


def find_optimal_thresholds_joint(y_true, y_prob, num_classes=5, iters=8):
    """
    ORTAK (joint) esik optimizasyonu — apply_thresholds'un GERCEK macro F1'ini
    dogrudan maksimize eder (koordinat yukselisi ile). Per-class bagimsiz
    optimizasyonun aksine AFL<->AFIB rekabetini hesaba katar; boylece AFL->AFIB
    karismasini karar katmaninda dengeleyebilir.

    Returns: (thresholds list, best_macro_f1)
    """
    th = [0.5] * num_classes

    def macro(t):
        return f1_score(y_true, apply_thresholds(y_prob, t), average='macro', zero_division=0)

    best = macro(th)
    grid = np.arange(0.05, 0.96, 0.01)
    for _ in range(iters):
        improved = False
        for c in range(num_classes):
            best_t, best_s = th[c], best
            for t in grid:
                th[c] = t
                s = macro(th)
                if s > best_s:
                    best_s, best_t = s, float(t)
            th[c] = best_t
            if best_s > best:
                best, improved = best_s, True
        if not improved:
            break
    return [round(t, 3) for t in th], round(best, 4)


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
