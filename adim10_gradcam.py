"""
adim10_gradcam.py — BirunAI EKG Siniflandirma: Adim 10 – 1D-GradCAM (XAI)
=============================================================================

Son CNN katmaninin gradyanlarini kullanarak EKG uzerinde
isi haritasi uretir. Juri savunmasinda "modelimiz bu taniyi
koyarken sinyalin su bolgesine odaklandi" diyebilirsiniz.

v2 Duzeltmeler:
    - forward hook'ta detach() yerine clone() kullaniliyor
      (gradyan akisini kesmemek icin)
    - torch.enable_grad() context manager ile eval modunda
      gradient hesaplama garantileniyor
    - input_tensor icin requires_grad gerekli degil,
      gradyanlar hook uzerinden katmandan yakalaniyor
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

        # Hook kaydet — forward'da activation'i, backward'da gradient'i yakala
        self._hooks.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        # clone() kullan, detach() degil — gradyan akisini korumak icin
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
        # cuDNN LSTM backward sadece train modunda calisir.
        # GradCAM icin gradient hesaplamasi gerektiginden,
        # modeli train moduna aliyoruz.
        # NOT: Tek ornek uzerinde calistigimiz icin BN/Dropout farki ihmal edilebilir.
        self.model.train()

        with torch.enable_grad():
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            target_score = output[0, target_class]
            target_score.backward()

        # Islem bitti, modeli eval'e geri al
        self.model.eval()

        # Gradyanlar ve aktivasyonlar hook'lardan yakalandi
        if self.gradients is None or self.activations is None:
            # Fallback: duz heatmap dondur
            target_len = input_tensor.shape[2]
            return np.zeros(target_len), target_class, output.softmax(dim=1)[0].detach().cpu().numpy()

        # Gradyan agirliklari: Global Average Pooling over time
        # gradients: (1, C, T) -> weights: (C,)
        weights = self.gradients.mean(dim=2).squeeze(0)  # (C,)

        # Agirlikli toplam: sum(w_i * A_i) over channels
        activations = self.activations.squeeze(0)  # (C, T)
        cam = torch.zeros(activations.shape[1], device=activations.device)
        for i in range(weights.shape[0]):
            cam += weights[i] * activations[i]

        cam = F.relu(cam)  # Negatif degerleri sifirla
        if cam.max() > 0:
            cam = cam / cam.max()  # [0, 1] normalize

        # Orijinal sinyal boyutuna interpole et
        cam_np = cam.detach().cpu().numpy()
        target_len = input_tensor.shape[2]
        cam_interp = np.interp(
            np.linspace(0, len(cam_np) - 1, target_len),
            np.arange(len(cam_np)),
            cam_np
        )

        probs = output.softmax(dim=1)[0].detach().cpu().numpy()

        # Temizle — bir sonraki cagri icin
        self.gradients = None
        self.activations = None

        return cam_interp, target_class, probs

    def remove_hooks(self):
        """Hook'lari temizle."""
        for hook in self._hooks:
            hook.remove()


def gradcam_gorsellestir(sinyal, heatmap, gercek_sinif, tahmin_sinif, olasiliklar,
                          ecg_id, kayit_yolu, fs=None):
    """Tek bir EKG kaydinin GradCAM gorsellestirmesi."""
    if fs is None:
        fs = config.TARGET_FS

    zaman = np.arange(sinyal.shape[1]) / fs  # saniye

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1, 0.5]})

    # 1. Lead II + Heatmap overlay
    lead_ii = sinyal[1]  # Lead II
    ax1 = axes[0]
    ax1.plot(zaman, lead_ii, color='#1a1a2e', linewidth=0.8, alpha=0.8)

    # Heatmap overlay
    extent = [zaman[0], zaman[-1], lead_ii.min(), lead_ii.max()]
    ax1.imshow(heatmap.reshape(1, -1), aspect='auto', extent=extent,
               cmap='jet', alpha=0.35, interpolation='bilinear')

    gercek_adi = config.LABEL_NAMES[gercek_sinif]
    tahmin_adi = config.LABEL_NAMES[tahmin_sinif]
    dogru = "OK" if gercek_sinif == tahmin_sinif else "YANLIS"

    ax1.set_title(f"ECG #{ecg_id} | Gercek: {gercek_adi} | Tahmin: {tahmin_adi} [{dogru}]",
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel("Lead II (z-score)")
    ax1.grid(True, alpha=0.2)

    # 2. Heatmap bar
    ax2 = axes[1]
    ax2.fill_between(zaman, 0, heatmap, color='red', alpha=0.6)
    ax2.set_ylabel("GradCAM")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2)

    # 3. Olasilik bar
    ax3 = axes[2]
    colors = ['#4CAF50', '#FF9800', '#F44336']
    ax3.barh(range(config.NUM_CLASSES), olasiliklar, color=colors, height=0.6)
    ax3.set_yticks(range(config.NUM_CLASSES))
    ax3.set_yticklabels([config.LABEL_NAMES[i] for i in range(config.NUM_CLASSES)], fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Olasilik")
    for i, v in enumerate(olasiliklar):
        ax3.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(kayit_yolu, dpi=150, bbox_inches='tight')
    plt.close()


def gradcam_pipeline():
    """Her siniftan ornek GradCAM gorsellestirmeleri uretir."""
    print("=" * 70)
    print("BirunAI -- Adim 10: 1D-GradCAM (Aciklanabilir YZ)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model yukle
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"  [HATA] Model bulunamadi: {checkpoint_path}")
        return

    model = BirunAIModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"  Model yuklendi: {checkpoint_path}")

    # GradCAM — son CNN katmanini hedefle (son Conv1d)
    target_layer = None
    for i in range(len(model.cnn) - 1, -1, -1):
        if isinstance(model.cnn[i], torch.nn.Conv1d):
            target_layer = model.cnn[i]
            break

    if target_layer is None:
        print("  [HATA] CNN katmani bulunamadi!")
        return

    gradcam = GradCAM1D(model, target_layer)

    # Test veri seti
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    test_dataset = EKGDataset(
        os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv"),
        sinyal_dizini
    )

    # Her siniftan 3 ornek sec
    ornekler_per_sinif = 3
    gradcam_dir = os.path.join(config.REPORT_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    sinif_sayaclari = {i: 0 for i in range(config.NUM_CLASSES)}
    uretilen = 0

    print(f"\n  Her siniftan {ornekler_per_sinif} ornek GradCAM uretiliyor...")

    for idx in range(len(test_dataset)):
        sinyal, etiket = test_dataset[idx]
        sinif = etiket.item()

        if sinif_sayaclari[sinif] >= ornekler_per_sinif:
            if all(v >= ornekler_per_sinif for v in sinif_sayaclari.values()):
                break
            continue

        # Gradient hesaplamasini input'tan DEGIL, hook'lardan yakaliyoruz
        input_tensor = sinyal.unsqueeze(0).to(device)

        try:
            heatmap, tahmin, probs = gradcam.generate(input_tensor)

            ecg_id = test_dataset.ecg_ids[idx]
            dosya_adi = f"gradcam_sinif{sinif}_{sinif_sayaclari[sinif]+1}_ecg{ecg_id}.png"
            kayit_yolu = os.path.join(gradcam_dir, dosya_adi)

            gradcam_gorsellestir(
                sinyal.numpy(), heatmap, sinif, tahmin, probs, ecg_id, kayit_yolu
            )

            sinif_sayaclari[sinif] += 1
            uretilen += 1
            print(f"    [{config.LABEL_NAMES[sinif]}] ECG #{ecg_id} -> {dosya_adi}")

        except Exception as e:
            print(f"    [HATA] idx={idx}: {e}")
            continue

    # Hook'lari temizle
    gradcam.remove_hooks()

    print(f"\n  Toplam {uretilen} GradCAM gorseli uretildi: {gradcam_dir}")
    print(f"\n{'='*70}")
    print(f"Adim 10 tamamlandi.")
    print(f"{'='*70}")


if __name__ == "__main__":
    gradcam_pipeline()
