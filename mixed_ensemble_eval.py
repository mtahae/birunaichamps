"""
mixed_ensemble_eval.py — Farkli mimarileri (v6 + v7) birlikte ensemble et
==========================================================================
v7 mimari olarak v6'dan tamamen farkli (ConvNeXt + MoE) -> decorrelated uye.
Zayif olsa bile cesitlilik ensemble'i iyilestirebilir. Bunu ampirik test eder.

Kullanim: python mixed_ensemble_eval.py
"""
import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import EKGDataset
from model_v6 import CardioFusion6
from model_v7 import CardioFusion7
from threshold_opt import find_optimal_thresholds_joint, apply_thresholds

device = config.DEVICE
PDD = config.PROCESSED_DATA_DIR
names = [config.LABEL_NAMES[i] for i in range(5)]

# (checkpoint dosyasi, mimari sinifi)
MEMBERS = [
    ("best_model_seed42.pth", CardioFusion6),
    ("best_model_seed123.pth", CardioFusion6),
    ("best_model_seed2026.pth", CardioFusion6),
    ("best_model_mev7.pth", CardioFusion7),
]

val_ds = EKGDataset(os.path.join(PDD, "val_manifest.csv"),
                    os.path.join(PDD, "filtered_signals"), augment=False,
                    train_stats_path=os.path.join(PDD, "train_stats.npz"),
                    wide_features_dir=os.path.join(PDD, "wide_features"))
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)


def get_probs(ckpt, arch):
    m = arch(num_classes=5, num_aux_classes=config.NUM_AUX_CLASSES,
             num_domains=2, wide_feature_dim=config.WIDE_FEATURE_DIM).to(device)
    m.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, ckpt),
                                 map_location=device, weights_only=True))
    m.eval()
    yt, yp = [], []
    with torch.no_grad():
        for sig, wf, lbl, _, _ in val_loader:
            sig, wf = sig.to(device), wf.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits, _, _ = m(sig, wf)
            yp.append(torch.softmax(logits.float(), 1).cpu().numpy())
            yt.append(lbl.numpy())
    return np.concatenate(yt), np.concatenate(yp)


probs = {}
y_true = None
for ck, arch in MEMBERS:
    yt, yp = get_probs(ck, arch)
    if y_true is None:
        y_true = yt
    probs[ck] = yp
    f1 = f1_score(yt, yp.argmax(1), average='macro', zero_division=0)
    print(f"  {ck:26s} ({arch.__name__}): {f1:.4f}")


def eval_combo(keys, tag):
    ens = np.mean([probs[k] for k in keys], axis=0)
    argmax_f1 = f1_score(y_true, ens.argmax(1), average='macro', zero_division=0)
    jth, _ = find_optimal_thresholds_joint(y_true, ens, num_classes=5)
    real = apply_thresholds(ens, jth)
    real_f1 = f1_score(y_true, real, average='macro', zero_division=0)
    fc = f1_score(y_true, real, average=None, labels=list(range(5)), zero_division=0)
    print(f"\n{tag}")
    print(f"  argmax: {argmax_f1:.4f} | joint-esik: {real_f1:.4f} | " +
          " ".join(f"{names[i][:3]}:{fc[i]:.3f}" for i in range(5)))
    return real_f1


print("\n=== ENSEMBLE KARSILASTIRMASI ===")
v6_keys = [m[0] for m in MEMBERS if m[1] is CardioFusion6]
all_keys = [m[0] for m in MEMBERS]
f_v6 = eval_combo(v6_keys, "SADECE v6 (3 model) — mevcut en iyi:")
f_all = eval_combo(all_keys, "v6 + v7 (4 model, karisik mimari):")
print(f"\n=== SONUC: v7 eklemek {'IYILESTIRDI (+%.4f)' % (f_all-f_v6) if f_all > f_v6 else 'YARDIM ETMEDI (%.4f)' % (f_all-f_v6)} ===")
