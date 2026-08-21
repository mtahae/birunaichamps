"""
analiz_confusion.py — Egitilmis modelin val setinde HATA ANALIZI
=================================================================
Mevcut best_model.pth'i yukler, val setinde confusion matrix + per-class
precision/recall + esik optimizasyonu ciktisi verir. AFL/AFIB'in TAM olarak
neyle karistigini gormek icin (tahmin degil, olcum).

Kullanim: python analiz_confusion.py
"""
import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import EKGDataset, FocalLoss
from model_v7 import CardioFusion7 as CardioFusion6  # v7'ye gecildi (isim korundu)
from threshold_opt import find_optimal_thresholds, find_optimal_thresholds_joint, apply_thresholds

device = config.DEVICE
PDD = config.PROCESSED_DATA_DIR
val_ds = EKGDataset(os.path.join(PDD, "val_manifest.csv"),
                    os.path.join(PDD, "filtered_signals"), augment=False,
                    train_stats_path=os.path.join(PDD, "train_stats.npz"),
                    wide_features_dir=os.path.join(PDD, "wide_features"))
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

model = CardioFusion6(num_classes=config.NUM_CLASSES, num_aux_classes=config.NUM_AUX_CLASSES,
                      num_domains=2, wide_feature_dim=config.WIDE_FEATURE_DIM).to(device)
ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()
print(f"Model yuklendi: {ckpt}")

y_true, y_prob = [], []
with torch.no_grad():
    for sig, wf, lbl, _, _ in val_loader:
        sig, wf = sig.to(device), wf.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits, _, _ = model(sig, wf)
        y_prob.append(torch.softmax(logits.float(), 1).cpu().numpy())
        y_true.append(lbl.numpy())
y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
y_pred = y_prob.argmax(1)
names = [config.LABEL_NAMES[i] for i in range(5)]

print("\n=== CONFUSION MATRIX (satir=gercek, sutun=tahmin) ===")
cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
print("gercek\\tahmin  " + "  ".join(f"{n[:5]:>6}" for n in names))
for i, row in enumerate(cm):
    print(f"{names[i]:>12}  " + "  ".join(f"{v:>6}" for v in row))

print("\n=== PER-CLASS ===")
p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(5)), zero_division=0)
for i in range(5):
    print(f"  {names[i]:>7}: P={p[i]:.3f} R={r[i]:.3f} F1={f[i]:.3f} (n={s[i]})")
print(f"  Macro F1 (argmax): {f1_score(y_true, y_pred, average='macro'):.4f}")

print("\n=== En cok karisan ciftler ===")
pairs = []
for i in range(5):
    for j in range(5):
        if i != j and cm[i][j] > 0:
            pairs.append((cm[i][j], names[i], names[j]))
for cnt, a, b in sorted(pairs, reverse=True)[:6]:
    print(f"  {a} -> {b} olarak {cnt} kez")

print("\n=== ESIK: PER-CLASS BAGIMSIZ (eski, TERS calisiyor) ===")
th, f1s = find_optimal_thresholds(y_true, y_prob, num_classes=5)
real = apply_thresholds(y_prob, th)
print(f"  GERCEK Macro F1: {f1_score(y_true, real, average='macro'):.4f}")

print("\n=== ESIK: ORTAK/JOINT (yeni, macro-F1 dogrudan) ===")
jth, jbest = find_optimal_thresholds_joint(y_true, y_prob, num_classes=5)
jpred = apply_thresholds(y_prob, jth)
print(f"  Esikler: " + " | ".join(f"{names[i][:3]}:{jth[i]:.2f}" for i in range(5)))
print(f"  GERCEK Macro F1 (JOINT): {f1_score(y_true, jpred, average='macro'):.4f}")
jf = f1_score(y_true, jpred, average=None, labels=list(range(5)), zero_division=0)
print("  Sinif F1 (joint): " + " | ".join(f"{names[i][:3]}:{jf[i]:.3f}" for i in range(5)))
jcm = confusion_matrix(y_true, jpred, labels=list(range(5)))
print(f"  AFL->AFIB (joint sonrasi): {jcm[2][1]} (argmax'ta 48 idi)")
