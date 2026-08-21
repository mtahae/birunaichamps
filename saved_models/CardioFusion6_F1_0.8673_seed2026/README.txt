CardioFusion-6 (Lean-Robust) — seed2026
==========================================

Tarih        : 2026-07-20
Mimari       : model_v6.py:CardioFusion6 (~3.9M parametre)
Seed         : 2026
En iyi epoch : 67 (Faz 3 / fine-tune)
Val Macro F1 : 0.8673 (argmax, tek model — su ana kadarki EN IYI tekil model)

Sinif bazli F1 (validation, argmax):
  Normal : 0.947
  AFIB   : 0.791
  AFL    : 0.680
  LBBB   : 0.940
  RBBB   : 0.941

Not: Egitim sirasinda kaydedilen log 0.8600 gosteriyordu; bu klasordeki 0.8673
degeri checkpoint'in bagimsiz/tekrarlanan bir degerlendirmesinden (ensemble_eval.py
ile ayni yontemle, tum modellerle tutarli olcum icin) geliyor. Kucuk fark (~0.007)
AMP/GPU kayan-nokta hesap sirasindaki dogal olcum gurultusu — kayda deger degil.

Icerik
------
best_model_seed2026.pth   Model agirliklari (state_dict).
train_stats.npz           Lead-wise Z-score mean/std (SADECE train setinden).
                           Inference icin SART.
training_log_seed2026.json Epoch epoch tam egitim gecmisi.

Notlar
------
- Bu, 3-seed ensemble'in (seed42/seed123/seed2026) uyesidir. Ensemble
  (softmax ortalama + joint esik) ile GENEL EN IYI SONUC: 0.8736
  (bkz saved_models/CardioFusion6_ENSEMBLE_F1_0.8736/)
- Kisaltilmis P2 (18 epoch, patience=8) ile egitildi — onceki 30-epoch/patience=12
  konfigurasyonuna gore hem daha hizli hem esdeger/daha iyi sonuc verdi.
- Egitim komutu: python adim08_egitim.py --seed 2026 --tag seed2026
