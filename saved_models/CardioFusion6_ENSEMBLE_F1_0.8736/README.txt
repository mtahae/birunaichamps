CardioFusion-6 — 3-Seed Ensemble (GENEL EN IYI SONUC)
========================================================

Tarih         : 2026-07-20
Uyeler        : seed42 (0.8539) + seed123 (0.8600) + seed2026 (0.8673)
Yontem        : Softmax olasilik ortalamasi + ORTAK/JOINT esik optimizasyonu
GERCEK Macro F1 (yarisma metrigi) : 0.8736

Sinif bazli F1 (esik sonrasi):
  Normal : 0.960
  AFIB   : 0.795
  AFL    : 0.712
  LBBB   : 0.960
  RBBB   : 0.940

Esikler (find_optimal_thresholds_joint ile hesaplandi):
  Normal:0.34 | AFIB:0.42 | AFL:0.50 | LBBB:0.30 | RBBB:0.55

Icerik
------
best_model_seed42.pth     Ensemble uyesi 1
best_model_seed123.pth    Ensemble uyesi 2
best_model_seed2026.pth   Ensemble uyesi 3
train_stats.npz           Lead-wise Z-score mean/std — UCUNUN DE inference'i icin SART
ensemble_result.json      Skorlar + esikler (makine-okunur "tarif")
ensemble_eval.py          Bu sonucu tekrar uretmek icin calistirilacak script

Nasil tekrar uretilir
----------------------
Bu klasoru orijinal proje kokune kopyala (config.py, adim07_model_mimarisi.py,
model_v6.py, threshold_opt.py gerekli — ensemble_eval.py bunlari import eder),
outputs/checkpoints/ altina checkpoint'leri yerlestir, sonra:

    python ensemble_eval.py best_model_seed42.pth best_model_seed123.pth best_model_seed2026.pth

Onemli
------
- BU BIR TEK MODEL DEGIL — ucu de AYNI ANDA yuklenip olasiliklari ortalanarak
  bu skor elde edilir. Sadece bir checkpoint'i kullanmak DAHA DUSUK skor verir
  (tekil skorlar icin yukarida "Uyeler" satirina bak).
- Esikler (0.34/0.42/0.50/0.30/0.55) argmax (0.5 sabit) YERINE kullanilmali —
  bkz threshold_opt.py:apply_thresholds.
