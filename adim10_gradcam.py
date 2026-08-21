"""
adim10_gradcam.py — BirunAI EKG Siniflandirma: Adim 10 – 1D-GradCAM (XAI)
=============================================================================

CardioFusion-5 (5 sinif) modeli icin 1D Grad-CAM gorsellistirmesi.
Her siniftan ornek EKG kayitlarini 12-lead formatinda gorsellestirir.

Ozellikler:
    - 5 sinif destegi: Normal, AFIB, AFL, LBBB, RBBB
    - CardioFusion5 forward uyumlulugu (wide_features, alpha, 3 output)
    - 12-Lead EKG gosterimi (Lead II vurgulu)
    - Profesyonel mor/purple tema (takim standardina uygun)
    - Sinif olasilik cubuk grafigi
    - Otomatik reports/gradcam/ dizinine kayit

Ciktilar:
    - outputs/reports/gradcam/gradcam_{sinif}_{isim}_{no}.png

Kullanim:
    python adim10_gradcam.py
    (veya adim08_egitim.py icerisinden otomatik cagirilir)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Lazy import — dairesel import onleme
_CardioFusion5 = None
_EKGDataset = None

def _lazy_imports():
    global _CardioFusion5, _EKGDataset
    if _CardioFusion5 is None:
        from adim07_model_mimarisi import CardioFusion5, EKGDataset
        _CardioFusion5 = CardioFusion5
        _EKGDataset = EKGDataset


# =============================================================================
# RENK PALETI (Takim standardina uygun — Mor/Purple tema)
# =============================================================================

# Ana mor palet
PURPLE_DARK = '#1a1a2e'
PURPLE_MID = '#4a148c'
PURPLE_LIGHT = '#9c27b0'
PURPLE_ACCENT = '#ce93d8'
BG_COLOR = '#f5f0ff'
GRID_COLOR = '#e0d0f0'

# Sinif renkleri (confusion matrix ile uyumlu)
CLASS_COLORS = {
    0: '#4CAF50',   # Normal — Yesil
    1: '#FF9800',   # AFIB — Turuncu
    2: '#F44336',   # AFL — Kirmizi
    3: '#2196F3',   # LBBB — Mavi
    4: '#9C27B0',   # RBBB — Mor
}

# GradCAM heatmap icin ozel colormap (seffaf -> mor -> kirmizi)
GRADCAM_CMAP = LinearSegmentedColormap.from_list(
    'gradcam_purple',
    [(0.0, (1, 1, 1, 0)),
     (0.3, '#7b1fa2'),
     (0.6, '#e91e63'),
     (0.8, '#ff5722'),
     (1.0, '#ffeb3b')],
    N=256
)


# =============================================================================
# GRAD-CAM 1D — CardioFusion5 Uyumlu
# =============================================================================

class GradCAM1D:
    """1D-GradCAM — CardioFusion5 modeli icin aciklanabilirlik (XAI)."""

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

    def generate(self, input_tensor, wide_features=None, target_class=None):
        """
        GradCAM haritasi uret.

        Args:
            input_tensor: (1, 12, 2500) tek bir EKG sinyali
            wide_features: (1, 8) wide features veya None
            target_class: Hedef sinif indeksi. None ise en yuksek skor.

        Returns:
            heatmap: (2500,) normalize edilmis isi haritasi [0-1]
            target_class: Kullanilan hedef sinif
            probs: (5,) sinif olasikliklari
        """
        # eval modunda calistir (BN batch_size=1 hatasini onler)
        # enable_grad ile backward hesapla
        self.model.eval()
        self.gradients = None
        self.activations = None

        with torch.enable_grad():
            # CardioFusion5 forward: class_logits, aux_logits, domain_logits
            output = self.model(input_tensor, wide_features=wide_features, alpha=0.0)
            
            # output bir tuple — sadece class_logits kullan
            if isinstance(output, tuple):
                class_logits = output[0]
            else:
                class_logits = output

            if target_class is None:
                target_class = class_logits.argmax(dim=1).item()

            self.model.zero_grad()
            target_score = class_logits[0, target_class]
            target_score.backward()

        self.model.eval()

        # Fallback: hook calismadiysa
        if self.gradients is None or self.activations is None:
            target_len = input_tensor.shape[2]
            probs = class_logits.softmax(dim=1)[0].detach().cpu().numpy()
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

        probs = class_logits.softmax(dim=1)[0].detach().cpu().numpy()

        self.gradients = None
        self.activations = None

        return cam_interp, target_class, probs

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()


# =============================================================================
# GORSELLISTIRME — Profesyonel 12-Lead GradCAM
# =============================================================================

# 12-Lead isimleri
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def gradcam_gorsellestir(sinyal, heatmap, gercek_sinif, tahmin_sinif, olasiliklar,
                          ecg_id, kayit_yolu, fs=None):
    """
    Profesyonel 12-Lead EKG GradCAM gorsellestirmesi.
    
    Layout:
        - Ust: 12-Lead EKG (6 sutun x 2 satir) + GradCAM overlay
        - Alt-Sol: Lead II detay (buyuk) + GradCAM heatmap strip
        - Alt-Sag: Sinif olasilik cubuk grafigi
    """
    if fs is None:
        fs = config.TARGET_FS

    zaman = np.arange(sinyal.shape[1]) / fs
    num_leads = min(sinyal.shape[0], 12)
    
    gercek_adi = config.LABEL_NAMES[gercek_sinif]
    tahmin_adi = config.LABEL_NAMES[tahmin_sinif]
    dogru = gercek_sinif == tahmin_sinif
    
    # --- Figure olustur ---
    fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
    
    # Ana grid: 3 satir
    gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.8, 4, 3.5],
                                 hspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # --- Baslik Satiri ---
    ax_title = fig.add_subplot(gs_main[0])
    ax_title.axis('off')
    
    # Durum iconu
    status_color = '#4CAF50' if dogru else '#F44336'
    status_text = '✓ DOĞRU' if dogru else '✗ YANLIŞ'
    
    title_text = (f"ECG #{ecg_id}   |   "
                  f"Gerçek: {gercek_adi}   →   Tahmin: {tahmin_adi}   ")
    
    ax_title.text(0.5, 0.6, title_text, transform=ax_title.transAxes,
                  fontsize=16, fontweight='bold', color=PURPLE_DARK,
                  ha='center', va='center',
                  fontfamily='monospace')
    ax_title.text(0.5, 0.15, status_text, transform=ax_title.transAxes,
                  fontsize=14, fontweight='bold', color=status_color,
                  ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor=status_color, alpha=0.15))
    
    # --- 12-Lead Grid (2 satir x 6 sutun) ---
    gs_leads = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs_main[1],
                                                 hspace=0.35, wspace=0.25)
    
    for lead_idx in range(num_leads):
        row = lead_idx // 6
        col = lead_idx % 6
        ax = fig.add_subplot(gs_leads[row, col])
        
        lead_signal = sinyal[lead_idx]
        
        # Sinyal ciz
        ax.plot(zaman, lead_signal, color=PURPLE_DARK, linewidth=0.6, alpha=0.85)
        
        # GradCAM overlay
        extent = [zaman[0], zaman[-1], lead_signal.min(), lead_signal.max()]
        if lead_signal.max() > lead_signal.min():
            ax.imshow(heatmap.reshape(1, -1), aspect='auto', extent=extent,
                      cmap='YlOrRd', alpha=0.25, interpolation='bilinear')
        
        # Lead ismi
        ax.set_title(LEAD_NAMES[lead_idx], fontsize=10, fontweight='bold',
                     color=PURPLE_MID, pad=3)
        
        # Eksen ayarlari
        ax.set_facecolor(BG_COLOR)
        ax.grid(True, alpha=0.15, color=GRID_COLOR)
        ax.tick_params(labelsize=7, colors=PURPLE_DARK)
        
        if col == 0:
            ax.set_ylabel('mV', fontsize=8, color=PURPLE_DARK)
        if row == 1:
            ax.set_xlabel('Zaman (s)', fontsize=7, color=PURPLE_DARK)
    
    # --- Alt Kisim: Lead II Detay + Olasilik Grafigi ---
    gs_bottom = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[2],
                                                  height_ratios=[3, 1],
                                                  width_ratios=[3, 1],
                                                  hspace=0.15, wspace=0.3)
    
    # Lead II — Buyuk gorsel + GradCAM overlay
    ax_lead2 = fig.add_subplot(gs_bottom[0, 0])
    lead_ii = sinyal[1]  # Lead II
    
    ax_lead2.plot(zaman, lead_ii, color=PURPLE_DARK, linewidth=1.0, alpha=0.9)
    extent_ii = [zaman[0], zaman[-1], lead_ii.min(), lead_ii.max()]
    if lead_ii.max() > lead_ii.min():
        ax_lead2.imshow(heatmap.reshape(1, -1), aspect='auto', extent=extent_ii,
                        cmap='YlOrRd', alpha=0.35, interpolation='bilinear')
    
    ax_lead2.set_title('Lead II — GradCAM Detay', fontsize=12, fontweight='bold',
                       color=PURPLE_MID)
    ax_lead2.set_ylabel('Amplitüd (z-score)', fontsize=9, color=PURPLE_DARK)
    ax_lead2.set_facecolor(BG_COLOR)
    ax_lead2.grid(True, alpha=0.2, color=GRID_COLOR)
    ax_lead2.tick_params(colors=PURPLE_DARK)
    
    # GradCAM Heatmap Strip (Lead II altinda)
    ax_heatmap = fig.add_subplot(gs_bottom[1, 0])
    ax_heatmap.fill_between(zaman, 0, heatmap, color=PURPLE_LIGHT, alpha=0.6)
    ax_heatmap.plot(zaman, heatmap, color=PURPLE_MID, linewidth=0.8)
    ax_heatmap.set_ylabel('GradCAM\nAktivasyonu', fontsize=8, color=PURPLE_DARK)
    ax_heatmap.set_xlabel('Zaman (s)', fontsize=9, color=PURPLE_DARK)
    ax_heatmap.set_ylim(0, 1.05)
    ax_heatmap.set_xlim(zaman[0], zaman[-1])
    ax_heatmap.set_facecolor(BG_COLOR)
    ax_heatmap.grid(True, alpha=0.15, color=GRID_COLOR)
    ax_heatmap.tick_params(colors=PURPLE_DARK)
    
    # Sinif Olasilik Cubuk Grafigi (Sag alt)
    ax_prob = fig.add_subplot(gs_bottom[:, 1])
    
    sinif_isimleri = [config.LABEL_NAMES[i] for i in range(config.NUM_CLASSES)]
    bar_colors = [CLASS_COLORS[i] for i in range(config.NUM_CLASSES)]
    
    # En yuksek olasiligin kenarini vurgula
    max_idx = np.argmax(olasiliklar)
    edge_colors = ['none'] * config.NUM_CLASSES
    edge_widths = [0] * config.NUM_CLASSES
    edge_colors[max_idx] = PURPLE_DARK
    edge_widths[max_idx] = 2.5
    
    bars = ax_prob.barh(range(config.NUM_CLASSES), olasiliklar, 
                         color=bar_colors, height=0.55, alpha=0.85,
                         edgecolor=edge_colors, linewidth=edge_widths)
    
    ax_prob.set_yticks(range(config.NUM_CLASSES))
    ax_prob.set_yticklabels(sinif_isimleri, fontsize=10, fontweight='bold',
                             color=PURPLE_DARK)
    ax_prob.set_xlim(0, 1.15)
    ax_prob.set_xlabel('Olasılık', fontsize=10, color=PURPLE_DARK)
    ax_prob.set_title('Sınıf Olasılıkları', fontsize=12, fontweight='bold',
                      color=PURPLE_MID, pad=10)
    
    for i, v in enumerate(olasiliklar):
        ax_prob.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10,
                     fontweight='bold' if i == max_idx else 'normal',
                     color=PURPLE_DARK)
    
    ax_prob.set_facecolor(BG_COLOR)
    ax_prob.grid(True, axis='x', alpha=0.2, color=GRID_COLOR)
    ax_prob.tick_params(colors=PURPLE_DARK)
    ax_prob.invert_yaxis()
    
    # Kaydet
    plt.savefig(kayit_yolu, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)


# =============================================================================
# HEDEF KATMAN BULMA
# =============================================================================

def _hedef_katmani_bul(model):
    """Modelin son Conv1d katmanini bulur (SE-ResNet layer4 icindeki son Conv1d)."""
    # Oncelikli: model.layer4 icindeki son Conv1d (SE-ResNet'in en derin katmani)
    if hasattr(model, 'layer4'):
        last_conv = None
        for name, module in model.layer4.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                last_conv = module
        if last_conv is not None:
            print(f"  GradCAM hedef katman: model.layer4 son Conv1d")
            return last_conv
    
    # Fallback: model.layer3
    if hasattr(model, 'layer3'):
        last_conv = None
        for name, module in model.layer3.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                last_conv = module
        if last_conv is not None:
            print(f"  GradCAM hedef katman: model.layer3 son Conv1d (fallback)")
            return last_conv
    
    # Son fallback: tum modeldeki son Conv1d
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            last_conv = module
    if last_conv is not None:
        print(f"  GradCAM hedef katman: son Conv1d (global fallback)")
    return last_conv


# =============================================================================
# ANA PIPELINE
# =============================================================================

def gradcam_pipeline():
    """Her siniftan ornek GradCAM gorsellestirmeleri uretir."""
    _lazy_imports()
    
    print("=" * 70)
    print("BirunAI -- Adim 10: 1D-GradCAM (Aciklanabilir YZ) — 5 Sinif")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Cihaz: {device}")

    # --- 1. Model yukle ---
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    swa_path = os.path.join(config.CHECKPOINT_DIR, "swa_model.pth")

    if os.path.exists(swa_path):
        yuklenecek = swa_path
        print("  [BILGI] swa_model.pth kullaniliyor (SWA modeli).")
    elif os.path.exists(checkpoint_path):
        yuklenecek = checkpoint_path
    else:
        print(f"  [HATA] Model bulunamadi: {config.CHECKPOINT_DIR}")
        print("  Lutfen once model egitimini tamamlayin.")
        return

    model = _CardioFusion5().to(device)
    try:
        state = torch.load(yuklenecek, map_location=device, weights_only=True)
    except TypeError:
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
        test_dataset = _EKGDataset(test_manifest, sinyal_dizini, augment=False)
    except Exception as e:
        print(f"  [HATA] Test veri seti yuklenemedi: {e}")
        gradcam.remove_hooks()
        return

    print(f"  Test kayit sayisi: {len(test_dataset)}")

    # --- 4. GradCAM uret ---
    ornekler_per_sinif = 3  # Her siniftan 3 ornek
    gradcam_dir = os.path.join(config.REPORT_DIR, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    sinif_sayaclari = {i: 0 for i in range(config.NUM_CLASSES)}
    uretilen = 0
    hatalar = 0

    print(f"\n  Her siniftan {ornekler_per_sinif} ornek GradCAM uretiliyor...")
    print(f"  Siniflar: {', '.join([config.LABEL_NAMES[i] for i in range(config.NUM_CLASSES)])}")
    print(f"  (Sinif az ise daha az uretilir)\n")

    for idx in range(len(test_dataset)):
        # Tum siniflar doluysa dur
        if all(v >= ornekler_per_sinif for v in sinif_sayaclari.values()):
            break

        try:
            # EKGDataset: sinyal, wide_features, etiket, aux_etiket, domain
            batch = test_dataset[idx]
            sinyal = batch[0]        # (12, 2500)
            wide_feat = batch[1]     # (8,)
            etiket = batch[2]        # scalar
            sinif = etiket.item()
        except FileNotFoundError:
            continue  # Dosya bulunamadi — sessizce atla
        except Exception as e:
            continue

        if sinif_sayaclari.get(sinif, ornekler_per_sinif) >= ornekler_per_sinif:
            continue

        input_tensor = sinyal.unsqueeze(0).to(device)
        wide_tensor = wide_feat.unsqueeze(0).to(device)

        try:
            heatmap, tahmin, probs = gradcam.generate(
                input_tensor, wide_features=wide_tensor
            )

            ecg_id = test_dataset.ecg_ids[idx]
            sinif_adi = config.LABEL_NAMES[sinif].replace(' ', '_')
            dosya_adi = f"gradcam_{sinif_adi}_{sinif_sayaclari[sinif]+1}.png"
            kayit_yolu = os.path.join(gradcam_dir, dosya_adi)

            gradcam_gorsellestir(
                sinyal.numpy(), heatmap, sinif, tahmin, probs,
                ecg_id, kayit_yolu
            )

            sinif_sayaclari[sinif] += 1
            uretilen += 1
            
            dogru = "✓" if sinif == tahmin else "✗"
            print(f"    [{config.LABEL_NAMES[sinif]:6s}] "
                  f"ECG#{ecg_id} → Tahmin: {config.LABEL_NAMES[tahmin]} {dogru} "
                  f"| {dosya_adi}")

        except Exception as e:
            hatalar += 1
            if hatalar <= 5:
                print(f"    [ATLA] idx={idx}: {e}")
            continue

    gradcam.remove_hooks()

    # --- 5. Ozet ---
    print(f"\n  {'='*50}")
    print(f"  Toplam uretilen : {uretilen} GradCAM gorseli")
    if hatalar > 0:
        print(f"  Atlanan hatalar : {hatalar}")
    print(f"  Kayit yeri      : {gradcam_dir}")

    print(f"\n  Sinif bazli:")
    for sinif in range(config.NUM_CLASSES):
        sayi = sinif_sayaclari.get(sinif, 0)
        bar = '#' * sayi + '.' * (ornekler_per_sinif - sayi)
        print(f"    [{sinif}] {config.LABEL_NAMES[sinif]:6s}: [{bar}] ({sayi}/{ornekler_per_sinif})")

    # --- 6. Confusion Matrix ---
    print(f"\n  Confusion Matrix uretiliyor...")
    try:
        confusion_matrix_olustur(model, test_dataset, device, gradcam_dir)
    except Exception as e:
        print(f"  [UYARI] Confusion Matrix uretimi basarisiz: {e}")

    print(f"\n{'='*70}")
    print(f"Adim 10 tamamlandi.")
    print(f"{'='*70}")


def confusion_matrix_olustur(model, test_dataset, device, output_dir):
    """
    Test seti uzerinde confusion matrix olusturur ve kaydeder.
    5x5 heatmap + sinif bazli precision/recall/f1 tablosu.
    """
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    from torch.utils.data import DataLoader
    
    model.eval()
    tum_tahminler = []
    tum_etiketler = []
    
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in loader:
            signals = batch[0].to(device)
            wide_features = batch[1].to(device)
            labels = batch[2]
            
            try:
                class_logits, _, _ = model(signals, wide_features)
                preds = class_logits.argmax(dim=1).cpu().numpy()
                tum_tahminler.extend(preds)
                tum_etiketler.extend(labels.numpy())
            except Exception:
                continue
    
    if len(tum_tahminler) == 0:
        print("  [UYARI] Hic tahmin uretilemedi, CM olusturulamadi.")
        return
    
    tum_tahminler = np.array(tum_tahminler)
    tum_etiketler = np.array(tum_etiketler)
    
    # Confusion Matrix hesapla
    sinif_isimleri = [config.LABEL_NAMES[i] for i in range(config.NUM_CLASSES)]
    cm = confusion_matrix(tum_etiketler, tum_tahminler, labels=list(range(config.NUM_CLASSES)))
    
    # Yuzde olarak normalize (satir bazli)
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100
    
    # Macro F1
    macro_f1 = f1_score(tum_etiketler, tum_tahminler, average='macro', zero_division=0)
    per_class_f1 = f1_score(tum_etiketler, tum_tahminler, average=None,
                            labels=list(range(config.NUM_CLASSES)), zero_division=0)
    
    # --- Gorsellistirme ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), 
                              gridspec_kw={'width_ratios': [1.2, 0.8]})
    fig.patch.set_facecolor(PURPLE_DARK)
    
    # Sol: Confusion Matrix Heatmap
    ax_cm = axes[0]
    ax_cm.set_facecolor(PURPLE_DARK)
    
    # Mor tonlu colormap
    cm_cmap = LinearSegmentedColormap.from_list(
        'cm_purple', ['#1a1a2e', '#4a148c', '#7b1fa2', '#ce93d8', '#f3e5f5'], N=256)
    
    im = ax_cm.imshow(cm_norm, cmap=cm_cmap, aspect='auto', vmin=0, vmax=100)
    
    # Degerler
    for i in range(config.NUM_CLASSES):
        for j in range(config.NUM_CLASSES):
            color = 'white' if cm_norm[i, j] < 50 else '#1a1a2e'
            ax_cm.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0f}%)", 
                      ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    ax_cm.set_xticks(range(config.NUM_CLASSES))
    ax_cm.set_yticks(range(config.NUM_CLASSES))
    ax_cm.set_xticklabels(sinif_isimleri, color='white', fontsize=11)
    ax_cm.set_yticklabels(sinif_isimleri, color='white', fontsize=11)
    ax_cm.set_xlabel('Tahmin', color='white', fontsize=13, fontweight='bold')
    ax_cm.set_ylabel('Gercek', color='white', fontsize=13, fontweight='bold')
    ax_cm.set_title(f'Confusion Matrix  |  Macro F1: {macro_f1:.4f}', 
                    color='white', fontsize=14, fontweight='bold', pad=15)
    ax_cm.tick_params(colors='white')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.set_ylabel('%', color='white', fontsize=11)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Sag: Sinif bazli metrikler
    ax_metrics = axes[1]
    ax_metrics.set_facecolor(PURPLE_DARK)
    ax_metrics.axis('off')
    
    # Tablo verileri
    from sklearn.metrics import precision_score, recall_score
    precision = precision_score(tum_etiketler, tum_tahminler, average=None,
                                labels=list(range(config.NUM_CLASSES)), zero_division=0)
    recall = recall_score(tum_etiketler, tum_tahminler, average=None,
                          labels=list(range(config.NUM_CLASSES)), zero_division=0)
    
    tablo_baslik = "Sinif Bazli Performans"
    ax_metrics.text(0.5, 0.95, tablo_baslik, transform=ax_metrics.transAxes,
                    ha='center', va='top', color=PURPLE_ACCENT, fontsize=14, fontweight='bold')
    
    header = f"{'Sinif':>8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'N':>8s}"
    ax_metrics.text(0.05, 0.85, header, transform=ax_metrics.transAxes,
                    ha='left', va='top', color='white', fontsize=10, fontfamily='monospace')
    
    ax_metrics.text(0.05, 0.80, "-" * 50, transform=ax_metrics.transAxes,
                    ha='left', va='top', color='#666', fontsize=10, fontfamily='monospace')
    
    for i in range(config.NUM_CLASSES):
        sinif_n = int(cm[i].sum())
        # F1'e gore renk: yesil > 0.8, sari > 0.6, kirmizi < 0.6
        if per_class_f1[i] >= 0.8:
            renk = '#4caf50'
        elif per_class_f1[i] >= 0.6:
            renk = '#ff9800'
        else:
            renk = '#f44336'
        
        satir = f"{sinif_isimleri[i]:>8s} {precision[i]:>10.3f} {recall[i]:>10.3f} {per_class_f1[i]:>10.3f} {sinif_n:>8d}"
        y_pos = 0.73 - i * 0.10
        ax_metrics.text(0.05, y_pos, satir, transform=ax_metrics.transAxes,
                        ha='left', va='top', color=renk, fontsize=11, fontfamily='monospace',
                        fontweight='bold')
    
    # Macro ortalama
    ax_metrics.text(0.05, 0.73 - config.NUM_CLASSES * 0.10 - 0.03, "-" * 50, 
                    transform=ax_metrics.transAxes,
                    ha='left', va='top', color='#666', fontsize=10, fontfamily='monospace')
    
    macro_satir = f"{'MACRO':>8s} {np.mean(precision):>10.3f} {np.mean(recall):>10.3f} {macro_f1:>10.3f} {int(len(tum_etiketler)):>8d}"
    ax_metrics.text(0.05, 0.73 - config.NUM_CLASSES * 0.10 - 0.10, macro_satir, 
                    transform=ax_metrics.transAxes,
                    ha='left', va='top', color='white', fontsize=11, fontfamily='monospace',
                    fontweight='bold')
    
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches='tight', facecolor=PURPLE_DARK)
    plt.close()
    
    print(f"  Confusion Matrix kaydedildi: {cm_path}")
    print(f"  Test Macro F1: {macro_f1:.4f}")
    for i in range(config.NUM_CLASSES):
        print(f"    {sinif_isimleri[i]:8s}: P={precision[i]:.3f} R={recall[i]:.3f} F1={per_class_f1[i]:.3f}")


if __name__ == "__main__":
    gradcam_pipeline()

