"""
adim10_gradcam.py — BirunAI EKG Siniflandirma: Adim 10 – 1D-GradCAM (XAI)
=============================================================================

Son CNN katmaninin gradyanlarini kullanarak EKG uzerinde
isi haritasi uretir.

v3 Duzeltmeler (Multi-Dataset):
    - test_manifest.csv path'i config.PROCESSED_DATA_DIR'den okunuyor
    - ecg_ids attribute EKGDataset'ten guvenle okunuyor
    - Model katmanı araması daha saglam
    - Hata yonetimi iylestirildi — tek hata tum GradCAM'i durdurmuyor
    - Her siniftan buldugu kadar gorsel uretiyor (3 yoksa 1 de yeterli)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import BirunAIModel, EKGDataset


class GradCAM1D:
    """1D-GradCAM — EKG sinyali icin aciklanabilirlik (XAI)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

        self._hooks.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.clone()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone()

    def generate(self, input_tensor, target_class=None):
        """
        GradCAM haritasi uret.

        Args:
            input_tensor: (1, 12, 2500) tek bir EKG sinyali
            target_class: Hedef sinif indeksi. None ise en yuksek skor.

        Returns:
            heatmap: (2500,) normalize edilmis isi haritasi [0-1]
            target_class: Kullanilan hedef sinif
            probs: (3,) sinif olasikliklari
        """
        # LSTM backward icin train modunda calistir
        self.model.train()
        self.gradients = None
        self.activations = None

        with torch.enable_grad():
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()

        self.model.eval()

        # Fallback: hook calismadiysa
        if self.gradients is None or self.activations is None:
            target_len = input_tensor.shape[2]
            probs = output.softmax(dim=1)[0].detach().cpu().numpy()
            return np.zeros(target_len), target_class, probs

        # GAP agirlikli cam
        weights = self.gradients.mean(dim=2).squeeze(0)          # (C,)
        activations = self.activations.squeeze(0)                 # (C, T)
        cam = (weights.unsqueeze(1) * activations).sum(dim=0)    # (T,)

        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.detach().cpu().numpy()
        target_len = input_tensor.shape[2]
        cam_interp = np.interp(
            np.linspace(0, len(cam_np) - 1, target_len),
            np.arange(len(cam_np)),
            cam_np
        )

        probs = output.softmax(dim=1)[0].detach().cpu().numpy()

        self.gradients = None
        self.activations = None

        return cam_interp, target_class, probs

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()


def gradcam_gorsellestir(sinyal, heatmap, gercek_sinif, tahmin_sinif, olasiliklar,
                          ecg_id, kayit_yolu, fs=None):
    """Tek bir EKG kaydinin GradCAM gorsellestirmesi."""
    if fs is None:
        fs = config.TARGET_FS

    zaman = np.arange(sinyal.shape[1]) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 8),
                              gridspec_kw={'height_ratios': [3, 1, 0.5]})

    lead_ii = sinyal[1]
    ax1 = axes[0]
    ax1.plot(zaman, lead_ii, color='#1a1a2e', linewidth=0.8, alpha=0.8)

    extent = [zaman[0], zaman[-1], lead_ii.min(), lead_ii.max()]
    ax1.imshow(heatmap.reshape(1, -1), aspect='auto', extent=extent,
               cmap='jet', alpha=0.35, interpolation='bilinear')

    gercek_adi = config.LABEL_NAMES[gercek_sinif]
    tahmin_adi = config.LABEL_NAMES[tahmin_sinif]
    dogru = "✓ DOGRU" if gercek_sinif == tahmin_sinif else "✗ YANLIS"

    ax1.set_title(f"ECG #{ecg_id} | Gercek: {gercek_adi} | Tahmin: {tahmin_adi} [{dogru}]",
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel("Lead II (z-score)")
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.fill_between(zaman, 0, heatmap, color='red', alpha=0.6)
    ax2.set_ylabel("GradCAM")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    ax3.barh(range(config.NUM_CLASSES), olasiliklar, color=colors, height=0.6)
    ax3.set_yticks(range(config.NUM_CLASSES))
    ax3.set_yticklabels([config.LABEL_NAMES[i] for i in range(config.NUM_CLASSES)],
                         fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Olasilik")
    for i, v in enumerate(olasiliklar):
        ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(kayit_yolu, dpi=150, bbox_inches='tight')
    plt.close(fig)  # fig'i kapatmak daha guvenli


def _hedef_katmani_bul(model):
    """Modelin son Conv1d katmanini bulur."""
    # Oncelikli: model.cnn Sequential'indeki son Conv1d
    if hasattr(model, 'cnn'):
        for i in range(len(model.cnn) - 1, -1, -1):
            if isinstance(model.cnn[i], torch.nn.Conv1d):
                print(f"  GradCAM hedef katman: model.cnn[{i}] (Conv1d)")
                return model.cnn[i]

    # Fallback: tum modelde son Conv1d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module
    if last_conv is not None:
        print(f"  GradCAM hedef katman: son Conv1d (fallback)")
    return last_conv


def gradcam_pipeline():
    """Her siniftan ornek GradCAM gorsellestirmeleri uretir."""
    print("=" * 70)
    print("BirunAI -- Adim 10: 1D-GradCAM (Aciklanabilir YZ)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Cihaz: {device}")

    # --- 1. Model yukle ---
    # Once best_model.pth, yoksa swa_model.pth dene
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    swa_path = os.path.join(config.CHECKPOINT_DIR, "swa_model.pth")

    if os.path.exists(checkpoint_path):
        yuklenecek = checkpoint_path
    elif os.path.exists(swa_path):
        yuklenecek = swa_path
        print("  [BILGI] best_model.pth yok, swa_model.pth kullaniliyor.")
    else:
        print(f"  [HATA] Model bulunamadi: {config.CHECKPOINT_DIR}")
        print("  Lutfen once model egitimini tamamlayin.")
        return

    model = BirunAIModel().to(device)
    try:
        state = torch.load(yuklenecek, map_location=device, weights_only=True)
    except TypeError:
        # Eski PyTorch — weights_only desteklenmiyor
        state = torch.load(yuklenecek, map_location=device)

    model.load_state_dict(state)
    model.eval()
    print(f"  Model yuklendi: {yuklenecek}")

    # --- 2. Hedef katman bul ---
    target_layer = _hedef_katmani_bul(model)
    if target_layer is None:
        print("  [HATA] Conv1d katmani bulunamadi!")
        return

    gradcam = GradCAM1D(model, target_layer)

    # --- 3. Test veri seti ---
    test_manifest = os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv")
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")

    if not os.path.exists(test_manifest):
        print(f"  [HATA] Test manifest bulunamadi: {test_manifest}")
        gradcam.remove_hooks()
        return

    try:
        test_dataset = EKGDataset(test_manifest, sinyal_dizini, augment=False)
    except Exception as e:
        print(f"  [HATA] Test veri seti yuklenemedi: {e}")
        gradcam.remove_hooks()
        return

    print(f"  Test kayit sayisi: {len(test_dataset)}")

    # --- 4. GradCAM uret ---
    ornekler_per_sinif = 3
    gradcam_dir = os.path.join(config.REPORT_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    sinif_sayaclari = {i: 0 for i in range(config.NUM_CLASSES)}
    uretilen = 0
    hatalar = 0

    print(f"\n  Her siniftan {ornekler_per_sinif} ornek GradCAM uretiliyor...")
    print(f"  (Sinif az ise daha az uretilir, hata vermez)\n")

    for idx in range(len(test_dataset)):
        # Tum siniflar doluysa dur
        if all(v >= ornekler_per_sinif for v in sinif_sayaclari.values()):
            break

        try:
            sinyal, etiket = test_dataset[idx]
            sinif = etiket.item()
        except Exception as e:
            continue

        if sinif_sayaclari[sinif] >= ornekler_per_sinif:
            continue

        input_tensor = sinyal.unsqueeze(0).to(device)

        try:
            heatmap, tahmin, probs = gradcam.generate(input_tensor)

            ecg_id = test_dataset.ecg_ids[idx]
            dosya_adi = (f"gradcam_sinif{sinif}_"
                         f"{config.LABEL_NAMES[sinif].replace(' ','_')}_"
                         f"{sinif_sayaclari[sinif]+1}.png")
            kayit_yolu = os.path.join(gradcam_dir, dosya_adi)

            gradcam_gorsellestir(
                sinyal.numpy(), heatmap, sinif, tahmin, probs,
                ecg_id, kayit_yolu
            )

            sinif_sayaclari[sinif] += 1
            uretilen += 1
            print(f"    [{config.LABEL_NAMES[sinif]:20s}] "
                  f"ECG#{ecg_id} → {dosya_adi}")

        except Exception as e:
            hatalar += 1
            print(f"    [ATLA] idx={idx}: {e}")
            continue

    gradcam.remove_hooks()

    print(f"\n  Toplam uretilen : {uretilen} GradCAM gorseli")
    if hatalar > 0:
        print(f"  Atlanan hatalar : {hatalar}")
    print(f"  Kayit yeri      : {gradcam_dir}")

    # Sinif bazli ozet
    print(f"\n  Sinif bazli:")
    for sinif, sayi in sinif_sayaclari.items():
        print(f"    [{sinif}] {config.LABEL_NAMES[sinif]:20s}: {sayi} gorsel")

    print(f"\n{'='*70}")
    print(f"Adim 10 tamamlandi.")
    print(f"{'='*70}")


if __name__ == "__main__":
    gradcam_pipeline()
