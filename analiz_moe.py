"""
analiz_moe.py — CardioFusion7 Mixture-of-Experts TESHIS
========================================================
Kritik soru: Ritim uzmaninin KENDI basligi AFIB/AFL'de iyi mi, yoksa gate onu
bastiriyor mu? Bu, MoE fikrinin kurtarilabilir olup olmadigini soyler.

Raporlar:
  - Fused (final) sinif F1
  - Morfoloji uzmani KENDI basligi sinif F1
  - Ritim uzmani KENDI basligi sinif F1
  - Gate agirliklari: genel + true-AFIB + true-AFL ornekleri icin (ritime yonlendiriyor mu?)
"""
import os, sys, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import EKGDataset
from model_v7 import CardioFusion7

device = config.DEVICE
PDD = config.PROCESSED_DATA_DIR
names = [config.LABEL_NAMES[i] for i in range(5)]

val_ds = EKGDataset(os.path.join(PDD, "val_manifest.csv"),
                    os.path.join(PDD, "filtered_signals"), augment=False,
                    train_stats_path=os.path.join(PDD, "train_stats.npz"),
                    wide_features_dir=os.path.join(PDD, "wide_features"))
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

model = CardioFusion7(num_classes=5, num_aux_classes=config.NUM_AUX_CLASSES,
                      num_domains=2, wide_feature_dim=config.WIDE_FEATURE_DIM).to(device)
ckpt = os.path.join(config.CHECKPOINT_DIR, sys.argv[1] if len(sys.argv) > 1 else "best_model_mev7.pth")
model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
model.eval()
print(f"Model: {ckpt}\n")

y_true, fused_p, morph_p, rhythm_p, gate_all = [], [], [], [], []
with torch.no_grad():
    for sig, wf, lbl, _, _ in val_loader:
        sig, wf = sig.to(device), wf.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            cl, ax, dl, ml, rl = model(sig, wf, return_experts=True)
            # gate agirliklarini yeniden hesapla
            feat = model.stem(sig)
            for s in [model.morph_stage1, model.morph_down1, model.morph_stage2,
                      model.morph_down2, model.morph_stage3, model.morph_down3, model.morph_stage4]:
                feat = s(feat)
            mf = model.morph_pool(feat.permute(0, 2, 1))
            gate = F.softmax(model.gate(mf), dim=1)
        y_true.append(lbl.numpy())
        fused_p.append(cl.float().argmax(1).cpu().numpy())
        morph_p.append(ml.float().argmax(1).cpu().numpy())
        rhythm_p.append(rl.float().argmax(1).cpu().numpy())
        gate_all.append(gate.float().cpu().numpy())

y_true = np.concatenate(y_true)
fused_p = np.concatenate(fused_p); morph_p = np.concatenate(morph_p); rhythm_p = np.concatenate(rhythm_p)
gate_all = np.concatenate(gate_all)  # (N, 2) [morph, rhythm]


def show(tag, pred):
    f1 = f1_score(y_true, pred, average='macro', zero_division=0)
    fc = f1_score(y_true, pred, average=None, labels=list(range(5)), zero_division=0)
    print(f"  {tag:20s} macro={f1:.4f} | " + " ".join(f"{names[i][:3]}:{fc[i]:.3f}" for i in range(5)))

print("=== BASLIK BAZINDA F1 (ayni model, farkli cikislar) ===")
show("FUSED (final)", fused_p)
show("Morfoloji uzmani", morph_p)
show("Ritim uzmani", rhythm_p)

print("\n=== GATE AGIRLIKLARI (ritim uzmanina verilen agirlik) ===")
print(f"  Genel ortalama ritim-agirligi : {gate_all[:, 1].mean():.3f}")
for c in range(5):
    mask = y_true == c
    print(f"  true-{names[c]:7s} ritim-agirligi: {gate_all[mask, 1].mean():.3f}  (n={mask.sum()})")

print("\n=== YORUM ===")
afib_afl = np.isin(y_true, [1, 2])
rhythm_gate_on_rhythm = gate_all[afib_afl, 1].mean()
rhythm_own_afibafl = f1_score(y_true, rhythm_p, average=None, labels=[1, 2], zero_division=0).mean()
fused_afibafl = f1_score(y_true, fused_p, average=None, labels=[1, 2], zero_division=0).mean()
morph_afibafl = f1_score(y_true, morph_p, average=None, labels=[1, 2], zero_division=0).mean()
print(f"  AFIB+AFL ort F1 -> Fused:{fused_afibafl:.3f} Morf:{morph_afibafl:.3f} Ritim:{rhythm_own_afibafl:.3f}")
print(f"  Gate AFIB/AFL orneklerinde ritim uzmanina ort {rhythm_gate_on_rhythm:.2f} agirlik veriyor")
if rhythm_own_afibafl > fused_afibafl + 0.02:
    print("  -> RITIM UZMANI KENDI BASINA DAHA IYI! Gate onu bastiriyor -> DUZELTILEBILIR (fusion sorunu)")
elif morph_afibafl >= rhythm_own_afibafl:
    print("  -> Ritim uzmani morfolojiden IYI DEGIL -> ayri uzman fikri AFIB/AFL'yi cozmuyor (tavan gercek)")
else:
    print("  -> Uzmanlar benzer; fusion optimal civari")
