"""
adim09_degerlendirme.py — BirunAI EKG Siniflandirma: Adim 9 – Degerlendirme
==============================================================================

Test seti uzerinde detayli performans raporu uretir.

Metrikler:
    - Macro F1 Score (birincil metrik)
    - Per-class Precision, Recall, F1
    - Normalized Confusion Matrix
    - ROC-AUC (One-vs-Rest)
    - Cohen's Kappa

Ciktilar:
    - outputs/reports/classification_report.txt
    - outputs/reports/confusion_matrix.png
    - outputs/reports/roc_curves.png
    - outputs/reports/test_metrics.json
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # GUI olmadan calistir
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, cohen_kappa_score, precision_recall_fscore_support
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import BirunAIModel, EKGDataset


def degerlendirme_pipeline():
    """Test seti uzerinde model degerlendirmesi."""
    print("=" * 70)
    print("BirunAI -- Adim 9: Model Degerlendirmesi")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Cihaz: {device}")

    # --- 1. Model yukle ---
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model bulunamadi: {checkpoint_path}\nOnce egitim yapilmali.")

    model = BirunAIModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Model yuklendi: {checkpoint_path}")

    # --- 2. Test DataLoader ---
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    test_dataset = EKGDataset(
        os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv"),
        sinyal_dizini
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"  Test seti: {len(test_dataset)} kayit")

    # --- 3. Tahmin ---
    print(f"\n  Tahminler yapiliyor...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # --- 4. Metrikler ---
    sinif_isimleri = [config.LABEL_NAMES[i] for i in range(config.NUM_CLASSES)]

    # Classification Report
    report = classification_report(
        all_labels, all_preds,
        target_names=sinif_isimleri,
        digits=4, zero_division=0
    )
    print(f"\n{report}")

    # Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], zero_division=0
    )

    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  Cohen Kappa  : {kappa:.4f}")

    # --- 5. Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds, normalize='true')

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(config.NUM_CLASSES)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(sinif_isimleri, rotation=45, ha='right')
    ax.set_yticklabels(sinif_isimleri)

    # Degerleri hucrelere yaz
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > 0.5 else "black"
            ax.text(j, i, f'{cm[i, j]:.3f}', ha='center', va='center', color=color, fontsize=12)

    ax.set_ylabel('Gercek Etiket', fontsize=12)
    ax.set_xlabel('Tahmin Edilen', fontsize=12)
    plt.tight_layout()

    cm_path = os.path.join(config.REPORT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion Matrix: {cm_path}")

    # --- 6. ROC Egrileri ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3', '#FF5722', '#4CAF50']

    for i in range(config.NUM_CLASSES):
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{sinif_isimleri[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_path = os.path.join(config.REPORT_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"  ROC Curves   : {roc_path}")

    # --- 7. Sonuclari kaydet ---
    metrics = {
        "macro_f1": round(float(macro_f1), 4),
        "cohen_kappa": round(float(kappa), 4),
        "per_class": {
            sinif_isimleri[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i])
            }
            for i in range(config.NUM_CLASSES)
        }
    }

    # ROC-AUC ekle
    for i in range(config.NUM_CLASSES):
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        metrics["per_class"][sinif_isimleri[i]]["roc_auc"] = round(float(roc_auc_val), 4)

    metrics_path = os.path.join(config.REPORT_DIR, "test_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrikler    : {metrics_path}")

    report_path = os.path.join(config.REPORT_DIR, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write(f"\nMacro F1     : {macro_f1:.4f}\n")
        f.write(f"Cohen Kappa  : {kappa:.4f}\n")
    print(f"  Rapor        : {report_path}")

    print(f"\n{'='*70}")
    print(f"Adim 9 tamamlandi.")
    print(f"{'='*70}")

    return metrics


if __name__ == "__main__":
    sonuc = degerlendirme_pipeline()
