"""
ensemble_eval.py — Birden fazla checkpoint'in softmax ortalamasiyla ensemble degerlendirme
=============================================================================================
Her checkpoint ayni mimariyle (CardioFusion6) farkli seed'le egitilmis olmali.
Val set uzerinde her modelin olasiliklarini ortalar, confusion matrix + joint-esik
optimizasyonu ile GERCEK (yarisma-tipi) Macro F1'i raporlar.

Kullanim:
    python ensemble_eval.py best_model_seed42.pth best_model_seed123.pth [...]
"""
import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import EKGDataset
from model_v7 import CardioFusion7 as CardioFusion6  # v7'ye gecildi (isim korundu)
from threshold_opt import find_optimal_thresholds_joint, apply_thresholds

device = config.DEVICE
PDD = config.PROCESSED_DATA_DIR
names = [config.LABEL_NAMES[i] for i in range(5)]


def get_probs(ckpt_name, val_loader):
    model = CardioFusion6(num_classes=config.NUM_CLASSES, num_aux_classes=config.NUM_AUX_CLASSES,
                          num_domains=2, wide_feature_dim=config.WIDE_FEATURE_DIM).to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for sig, wf, lbl, _, _ in val_loader:
            sig, wf = sig.to(device), wf.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits, _, _ = model(sig, wf)
            y_prob.append(torch.softmax(logits.float(), 1).cpu().numpy())
            y_true.append(lbl.numpy())
    print(f"  [{ckpt_name}] yuklendi ve degerlendirildi.")
    return np.concatenate(y_true), np.concatenate(y_prob)


def main():
    ckpt_names = sys.argv[1:]
    if not ckpt_names:
        ckpt_names = ["best_model_seed42.pth", "best_model_seed123.pth"]
    print(f"Ensemble uyeleri: {ckpt_names}")

    val_ds = EKGDataset(os.path.join(PDD, "val_manifest.csv"),
                        os.path.join(PDD, "filtered_signals"), augment=False,
                        train_stats_path=os.path.join(PDD, "train_stats.npz"),
                        wide_features_dir=os.path.join(PDD, "wide_features"))
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = []
    y_true_ref = None
    per_model_f1 = []
    for ck in ckpt_names:
        yt, yp = get_probs(ck, val_loader)
        if y_true_ref is None:
            y_true_ref = yt
        preds = yp.argmax(1)
        f1 = f1_score(yt, preds, average='macro', zero_division=0)
        per_model_f1.append((ck, f1))
        all_probs.append(yp)

    print("\n=== TEKIL MODEL SKORLARI (argmax) ===")
    for ck, f1 in per_model_f1:
        print(f"  {ck}: {f1:.4f}")

    ens_prob = np.mean(all_probs, axis=0)
    ens_pred = ens_prob.argmax(1)
    ens_f1 = f1_score(y_true_ref, ens_pred, average='macro', zero_division=0)
    ens_f1_class = f1_score(y_true_ref, ens_pred, average=None, labels=list(range(5)), zero_division=0)

    print(f"\n=== ENSEMBLE (softmax ortalamasi, argmax) ===")
    print(f"  Macro F1: {ens_f1:.4f}")
    print(f"  Sinif F1: " + " | ".join(f"{names[i][:3]}:{ens_f1_class[i]:.3f}" for i in range(5)))

    cm = confusion_matrix(y_true_ref, ens_pred, labels=list(range(5)))
    print(f"\n  Confusion Matrix (satir=gercek, sutun=tahmin):")
    print("  " + "  ".join(f"{n[:5]:>6}" for n in names))
    for i, row in enumerate(cm):
        print(f"  {names[i]:>10}  " + "  ".join(f"{v:>6}" for v in row))

    print(f"\n=== ENSEMBLE + JOINT ESIK OPTIMIZASYONU ===")
    jth, jbest = find_optimal_thresholds_joint(y_true_ref, ens_prob, num_classes=5)
    jpred = apply_thresholds(ens_prob, jth)
    real_f1 = f1_score(y_true_ref, jpred, average='macro', zero_division=0)
    real_f1_class = f1_score(y_true_ref, jpred, average=None, labels=list(range(5)), zero_division=0)
    print(f"  Esikler: " + " | ".join(f"{names[i][:3]}:{jth[i]:.2f}" for i in range(5)))
    print(f"  GERCEK Macro F1 (ensemble+esik): {real_f1:.4f}  <-- yarisma metrigi")
    print(f"  Sinif F1: " + " | ".join(f"{names[i][:3]}:{real_f1_class[i]:.3f}" for i in range(5)))

    # Sonucu kaydet — bu, "0.8676" gibi bir sonucu tekrar uretebilmek icin gereken
    # TUM bilgi: hangi checkpoint'ler, hangi esikler. Model dosyasi DEGIL, bir "tarif".
    out_path = os.path.join(config.OUTPUT_DIR, "ensemble_result.json")
    result = {
        "checkpoints": ckpt_names,
        "per_model_f1": {ck: float(f1) for ck, f1 in per_model_f1},
        "ensemble_argmax_f1": float(ens_f1),
        "ensemble_threshold_f1": float(real_f1),
        "thresholds": [float(t) for t in jth],
        "class_names": names,
    }
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Kaydedildi: {out_path}")
    print(f"  (Bu bir model dosyasi DEGIL — hangi checkpoint'lerin, hangi esiklerle")
    print(f"   birlestirildiginin 'tarifi'. Sonucu tekrar uretmek icin checkpoint'lerin")
    print(f"   KENDISI + bu dosya birlikte gerekli.)")


if __name__ == "__main__":
    main()
