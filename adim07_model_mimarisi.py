"""
adim07_model_mimarisi.py — BirunAI EKG Siniflandirma: Adim 7 – Model Mimarisi
================================================================================

Hibrit 1D-CNN + BiLSTM + Self-Attention modeli.

Mimari:
    Girdi: (batch, 12, 2500)
    -> 3x CNN Blok (Conv1d + BN + ReLU + MaxPool + Dropout)
    -> BiLSTM (2 katman, bidirectional)
    -> Self-Attention (weighted sum)
    -> FC (2 katman)
    -> (batch, 3) logits

Toplam parametre: ~1.15M
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# EKG AUGMENTASYON
# =============================================================================

class EKGAugmentation:
    """
    EKG sinyalleri icin veri augmentasyonu.

    Neden augmentasyon?
        - Ritim Bozuklugu sinifi sadece 378 egitim ornegine sahip.
        - Augmentasyon, her epoch'ta farkli varyasyonlar uretiyor
          ve modelin ezberlemesini onluyor.
        - Train/Val F1 gapini kapatmanin en etkili yolu.

    Teknikler:
        1. Time Shift    : Sinyali rastgele kaydirma (circular)
        2. Gaussian Noise: Kucuk rastgele gurultu ekleme
        3. Amplitude Scale: Genlik olcekleme (0.85-1.15x)
        4. Lead Dropout  : Rastgele 1-2 derivasyonu sifirla
    """

    def __init__(self,
                 time_shift_max=200,
                 noise_std=0.05,
                 amplitude_range=(0.85, 1.15),
                 lead_dropout_prob=0.15,
                 p=0.8):
        """
        Args:
            time_shift_max: Maksimum kaydirma miktari (ornek sayisi)
            noise_std: Gaussian gurultu standart sapmasi
            amplitude_range: Genlik olcekleme araligi (min, max)
            lead_dropout_prob: Her derivasyonun sifirlanma olasiligi
            p: Augmentasyonun toplam uygulanma olasiligi
        """
        self.time_shift_max = time_shift_max
        self.noise_std = noise_std
        self.amplitude_range = amplitude_range
        self.lead_dropout_prob = lead_dropout_prob
        self.p = p

    def __call__(self, sinyal):
        """
        Args:
            sinyal: numpy array (12, 2500)

        Returns:
            numpy array (12, 2500) augmente edilmis sinyal
        """
        if random.random() > self.p:
            return sinyal  # Augmentasyon uygulanmadi

        sinyal = sinyal.copy()

        # 1. Time Shift (circular): Sinyali zaman ekseninde kaydır
        if random.random() < 0.5:
            shift = random.randint(-self.time_shift_max, self.time_shift_max)
            sinyal = np.roll(sinyal, shift, axis=1)

        # 2. Gaussian Noise: Kucuk gurultu ekle
        if random.random() < 0.5:
            noise = np.random.normal(0, self.noise_std, sinyal.shape).astype(np.float32)
            sinyal = sinyal + noise

        # 3. Amplitude Scaling: Genlik olcekle
        if random.random() < 0.5:
            scale = random.uniform(*self.amplitude_range)
            sinyal = sinyal * scale

        # 4. Lead Dropout: Bazi derivasyonlari sifirla
        if random.random() < 0.3:
            for ch in range(sinyal.shape[0]):
                if random.random() < self.lead_dropout_prob:
                    sinyal[ch] = 0.0

        return sinyal


# Varsayilan augmentasyon instance'i
DEFAULT_AUGMENTATION = EKGAugmentation()


# =============================================================================
# EKG DATASET
# =============================================================================

class EKGDataset(Dataset):
    """
    Segmente edilmis EKG sinyallerini PyTorch Dataset olarak sarmalar.

    Her __getitem__ cagrisi:
        1. ecg_id'ye gore .npy dosyasini yukler
        2. augment=True ise augmentasyon uygular (sadece train)
        3. torch.FloatTensor'a cevirir (12, 2500)
        4. Etiketi torch.LongTensor olarak dondurur
    """

    def __init__(self, manifest_path, sinyal_dizini, augment=False):
        """
        Args:
            manifest_path: train/val/test_manifest.csv dosya yolu
            sinyal_dizini: segmented_signals/ dizin yolu
            augment: True ise egitim augmentasyonu uygula (sadece train seti)
        """
        self.df = pd.read_csv(manifest_path, index_col="ecg_id")
        self.sinyal_dizini = sinyal_dizini
        self.ecg_ids = self.df.index.tolist()
        self.labels = self.df["label"].values.astype(int)
        self.augment = augment
        self.augmentation = DEFAULT_AUGMENTATION if augment else None

    def __len__(self):
        return len(self.ecg_ids)

    def __getitem__(self, idx):
        ecg_id = self.ecg_ids[idx]
        sinyal_yolu = os.path.join(self.sinyal_dizini, f"{ecg_id}.npy")
        sinyal = np.load(sinyal_yolu)  # (12, 2500) float32

        # Augmentasyon (sadece train)
        if self.augment and self.augmentation is not None:
            sinyal = self.augmentation(sinyal)

        sinyal = torch.FloatTensor(sinyal)       # (12, 2500)
        etiket = torch.LongTensor([self.labels[idx]])[0]
        return sinyal, etiket


# =============================================================================
# SELF-ATTENTION MODULU
# =============================================================================

class SelfAttention(nn.Module):
    """
    Self-Attention mekanizmasi.

    BiLSTM ciktisinin her zaman adimina onem skoru atar,
    sonra agirlikli toplam ile tek bir vektor uretir.

    Bu, modelin EKG sinyalinin hangi bolgelerine odaklandigini
    gosterir -> GradCAM ile birlikte XAI destegi saglar.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)

        Returns:
            context: (batch, hidden_dim) — agirlikli toplam
            attn_weights: (batch, seq_len) — attention skorlari
        """
        attn_scores = self.attention(lstm_output)       # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)    # normalize over time
        context = torch.sum(lstm_output * attn_weights, dim=1)  # (batch, hidden_dim)
        return context, attn_weights.squeeze(-1)


# =============================================================================
# ANA MODEL: 1D-CNN + BiLSTM + ATTENTION
# =============================================================================

class BirunAIModel(nn.Module):
    """
    BirunAI EKG Siniflandirma Modeli.

    Mimari:
        1D-CNN (3 blok) -> BiLSTM (2 katman) -> Self-Attention -> FC

    Neden bu mimari?
        - CNN: Yerel morfoloji (P, QRS, T dalga sekilleri)
        - BiLSTM: Zamansal bagimlilik (ritim duzenliligi, PR iliskisi)
        - Attention: Hangi zaman dilimine odaklanmali? (XAI)
    """

    def __init__(self):
        super().__init__()

        # --- CNN Bloklari ---
        cnn_layers = []
        in_channels = config.NUM_LEADS  # 12

        for out_channels in config.CNN_FILTERS:  # [64, 128, 256]
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=config.CNN_KERNEL_SIZE,
                          padding=config.CNN_KERNEL_SIZE // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(config.CNN_DROPOUT)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # --- BiLSTM ---
        self.lstm = nn.LSTM(
            input_size=config.CNN_FILTERS[-1],     # 256
            hidden_size=config.LSTM_HIDDEN_SIZE,    # 128
            num_layers=config.LSTM_NUM_LAYERS,      # 2
            batch_first=True,
            bidirectional=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0
        )

        lstm_output_size = config.LSTM_HIDDEN_SIZE * 2  # bidirectional -> 256

        # --- Self-Attention ---
        self.attention = SelfAttention(lstm_output_size)

        # --- Fully Connected ---
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, config.FC_HIDDEN_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(config.FC_DROPOUT),
            nn.Linear(config.FC_HIDDEN_SIZE, config.NUM_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 12, 2500) — 12 derivasyon EKG sinyali

        Returns:
            logits: (batch, 3) — sinif skorlari
        """
        x = self.cnn(x)                    # (batch, 256, 312)
        x = x.permute(0, 2, 1)             # (batch, 312, 256)
        x, _ = self.lstm(x)                # (batch, 312, 256)
        x, _ = self.attention(x)           # (batch, 256)
        x = self.fc(x)                     # (batch, 3)
        return x

    def forward_with_attention(self, x):
        """Forward pass + attention weights dondurur (GradCAM/XAI icin)."""
        cnn_out = self.cnn(x)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)
        context, attn_weights = self.attention(lstm_out)
        logits = self.fc(context)
        return logits, attn_weights


# =============================================================================
# FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma = 0 -> Standart CrossEntropy
    gamma = 2 -> Kolay orneklerin katkisi ~0.25x'e duser, zor orneklere odaklanir

    Label Smoothing:
        Gercek etiketi 1.0 yerine (1 - smoothing) yapar,
        diger siniflara smoothing/(num_classes-1) dagitir.
        Bu, modelin asiri guvenliligi (overconfidence) onler ve
        overfitting'i azaltir.

    alpha_t: Sinif agirligi (class_weights.npy'den)
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            if isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha)
            else:
                self.register_buffer('alpha', torch.FloatTensor(alpha))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) — model ciktisi
            targets: (batch,) — gercek etiketler

        Returns:
            scalar: Ortalama focal loss
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


# =============================================================================
# YARDIMCI FONKSIYONLAR
# =============================================================================

def model_ozetini_yazdir(model):
    """Model parametre sayisini ve boyutunu yazdirir."""
    toplam = sum(p.numel() for p in model.parameters())
    egitim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    boyut_mb = toplam * 4 / (1024 * 1024)  # float32

    print(f"\n  Model Ozeti:")
    print(f"    Toplam parametre    : {toplam:,}")
    print(f"    Egitilebilir param  : {egitim:,}")
    print(f"    Model boyutu (GPU)  : ~{boyut_mb:.1f} MB")
    return toplam


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BirunAI -- Adim 7: Model Mimarisi Testi")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Cihaz: {device}")
    if device.type == 'cuda':
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")

    # Model olustur
    model = BirunAIModel().to(device)
    model_ozetini_yazdir(model)

    # Test girdisi
    dummy_input = torch.randn(4, 12, 2500).to(device)
    output = model(dummy_input)
    print(f"\n  Test girdi shape  : {dummy_input.shape}")
    print(f"  Test cikti shape  : {output.shape}")
    print(f"  Beklenen          : (4, 3)")

    # Attention testi
    logits, attn = model.forward_with_attention(dummy_input)
    print(f"  Attention shape   : {attn.shape}")

    # Focal Loss testi
    class_weights = np.load(os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy"))
    criterion = FocalLoss(alpha=class_weights, gamma=config.FOCAL_LOSS_GAMMA).to(device)
    targets = torch.tensor([0, 1, 2, 0]).to(device)
    loss = criterion(output, targets)
    print(f"\n  Focal Loss test   : {loss.item():.4f}")

    # Dataset testi
    train_manifest = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    if os.path.exists(train_manifest):
        dataset = EKGDataset(train_manifest, sinyal_dizini)
        sinyal, etiket = dataset[0]
        print(f"\n  Dataset boyut     : {len(dataset)}")
        print(f"  Sinyal shape      : {sinyal.shape}")
        print(f"  Etiket            : {etiket.item()}")

    print("\n" + "=" * 70)
    print("Adim 7 tamamlandi. Model hazir!")
    print("=" * 70)
