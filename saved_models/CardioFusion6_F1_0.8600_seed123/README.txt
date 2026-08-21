CardioFusion-6 (Lean-Robust) — seed123
=========================================

Tarih        : 2026-07-19
Mimari       : model_v6.py:CardioFusion6 (~3.9M parametre)
Seed         : 123
En iyi epoch : 59 (Faz 3 / fine-tune)
Val Macro F1 : 0.8600 (argmax, tek model)

Sinif bazli F1 (validation, argmax):
  Normal : 0.9333
  AFIB   : 0.7792
  AFL    : 0.7117
  LBBB   : 0.9495
  RBBB   : 0.9281

Icerik
------
best_model_seed123.pth   Model agirliklari (state_dict). Yuklemek icin:
                          model = CardioFusion6(...); model.load_state_dict(torch.load(...))
train_stats.npz          Lead-wise Z-score mean/std (SADECE train setinden). Inference
                          icin SART — bu olmadan model yanlis sonuc verir (bkz CLAUDE.md
                          Kritik Kural #1). EKGDataset(train_stats_path=...) ile kullanilir.
training_log_seed123.json Epoch epoch tam egitim gecmisi (loss, F1, LR, patience).

Notlar
------
- Bu, 3-seed ensemble'in (seed42/seed123/seed2026) TEK bir uyesidir. Ensemble
  (softmax ortalama + joint esik) ile birlikte kullanildiginda daha yuksek skor
  verir (bkz outputs/ensemble_result.json, seed42+seed123 icin 0.8676 raporlandi).
- Tek basina bu checkpoint 0.8600 verir (bu ana kadarki en iyi TEK model skoru).
- Egitim komutu: python adim08_egitim.py --seed 123 --tag seed123
