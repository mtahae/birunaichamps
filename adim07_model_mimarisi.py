"""
adim07_model_mimarisi.py — BirunAI EKG Siniflandirma: Adim 7 – Model Mimarisi
================================================================================

CardioFusion-5: SE-ResNet + Transformer + Wide Features + DANN + Multi-Task

PDF'deki TUM yontemlerin profesyonelce uygulandigi versiyon:
    - SE-ResNet (Squeeze-and-Excitation)
    - Transformer Encoder (Zamansal iliskiler)
    - Wide Features (Fizyolojik ozellikler — features.py'den)
    - DANN (Gradient Reversal Layer — Domain Adaptasyonu)
    - Multi-Task Loss (5-sinif main + 3-sinif aux — PDF Bolum 8)
    - Lead-Wise Z-Score (Train istatistikleriyle — PDF Bolum 1)

YASAK OLANLAR (PDF Bolum 6 Kritik Hususlar):
    - np.roll (zaman kaydirma)
    - Global Z-Score
    - Global amplitude scale
    - Random crop
    - 50Hz notch
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
# EKG AUGMENTASYON (PDF Bolum 3 — Yapilaacaklar/Yasaklar)
# =============================================================================

class EKGAugmentation:
    """
    PDF Bolum 3 Augmentasyon Kurallari:
    
    IZIN VERILENLER:
    1. Lead-Wise Amplitude Scale (0.9-1.1)
    2. Gaussian Noise (SNR > 20 dB)
    3. Lead Dropout (1-2 lead sifirla)
    4. Baseline Wander (sinuzoidal drift)
    
    ASLA YAPILMAYACAKLAR:
    1. np.roll (zaman kaydirma) -> P-QRS-T bozulur
    2. Random crop -> P/T dalgasi kaybolur
    3. Global amplitude scale -> V1/V6 orani bozulur
    """
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, sinyal):
        """sinyal: (12, 2500)"""
        if random.random() > self.p:
            return sinyal

        sinyal = sinyal.copy()
        C, L = sinyal.shape

        # 1. Lead-Wise Amplitude Scale (0.9-1.1) — PDF: Her lead bagimsiz
        if random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1, size=(C, 1)).astype(np.float32)
            sinyal *= scale

        # 2. Gaussian Noise (hafif) — PDF: SNR > 20 dB
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.01, sinyal.shape).astype(np.float32)
            sinyal += noise

        # 3. Lead Dropout — PDF: 1-2 lead sifirla
        if random.random() < 0.3:
            n_dropout = random.choice([1, 2])
            dropout_leads = np.random.choice(C, size=n_dropout, replace=False)
            for ch in dropout_leads:
                sinyal[ch] = 0.0

        # 4. Baseline Wander (sinuzoidal drift)
        if random.random() < 0.5:
            t = np.linspace(0, 10, L)
            for ch in range(C):
                if random.random() < 0.3:
                    freq = random.uniform(0.1, 0.5)
                    amp = random.uniform(0.05, 0.3)
                    drift = amp * np.sin(2 * np.pi * freq * t + random.uniform(0, 2*np.pi))
                    sinyal[ch] += drift.astype(np.float32)

        # 5. Time Masking — Rastgele bir zaman segmentini sifirla
        #    SpecAugment'in EKG versiyonu. np.roll veya crop DEGIL, sadece maskeleme.
        if random.random() < 0.4:
            mask_len = random.randint(50, 250)  # 50-250 sample (0.2-1.0 saniye)
            max_start = L - mask_len
            if max_start > 0:
                start = random.randint(0, max_start)
                sinyal[:, start:start + mask_len] = 0.0

        return sinyal

DEFAULT_AUGMENTATION = EKGAugmentation()

# =============================================================================
# EKG DATASET — PDF Bolum 1 (Lead-Wise Z-Score, Train Stats)
# =============================================================================

class EKGDataset(Dataset):
    """
    Multimodal EKG Dataset.
    
    Z-Score Normalizasyonu:
        PDF KRITIK KURAL: "mu ve sigma SADECE train setinden hesaplanir,
        val/test'e uygulanir. Global Z-Score ASLA yapilmaz."
        
        Eger train_stats_path verilmisse, oradan yukler.
        Verilmemisse per-sample lead-wise yapar (fallback).
    """
    def __init__(self, manifest_path, sinyal_dizini, augment=False, train_stats_path=None,
                 wide_features_dir=None):
        self.df = pd.read_csv(manifest_path, index_col="ecg_id")
        self.sinyal_dizini = sinyal_dizini
        self.ecg_ids = self.df.index.tolist()
        self.labels = self.df["label"].values.astype(int)
        
        # Aux labels (Multi-Task) — PDF Bolum 8
        self.aux_labels = np.array([config.MAIN_TO_AUX_LABEL[l] for l in self.labels], dtype=int)
        
        # Domain ID (DANN)
        if "domain_id" in self.df.columns:
            self.domains = self.df["domain_id"].values.astype(int)
        else:
            self.domains = np.zeros(len(self.df), dtype=int)
            
        self.augment = augment
        self.augmentation = DEFAULT_AUGMENTATION if augment else None
        
        # Wide Features dizini ve bellege yukleme (I/O hizlandirma)
        self.wide_features_dir = wide_features_dir
        self.wide_features_array = np.zeros((len(self.ecg_ids), config.WIDE_FEATURE_DIM), dtype=np.float32)
        
        if self.wide_features_dir is not None:
            # Manifest turune gore (train/val) cache dosyasi olustur (Saniyeler icinde yuklenmesi icin)
            is_train = "train" in manifest_path.lower()
            cache_file = os.path.join(self.wide_features_dir, f"{'train' if is_train else 'val'}_wide_features_cache.npy")
            
            if os.path.exists(cache_file):
                print(f"  [EKGDataset] Wide Features Cache'den yukleniyor... ({'Train' if is_train else 'Val'})")
                self.wide_features_array = np.load(cache_file)
            else:
                from tqdm import tqdm
                print(f"  [EKGDataset] Wide Features RAM'e yukleniyor... (Ilk sefere mahsus biraz surebilir)")
                for idx, ecg_id in enumerate(tqdm(self.ecg_ids, desc=f"  Yükleniyor ({'Train' if is_train else 'Val'})", leave=False)):
                    wf_path = os.path.join(self.wide_features_dir, f"{ecg_id}.npy")
                    if os.path.exists(wf_path):
                        self.wide_features_array[idx] = np.load(wf_path)
                # Cache olarak kaydet ki bir sonraki calistirmada 0.1 saniyede acilsin
                np.save(cache_file, self.wide_features_array)
        
        # Train istatistikleri yukle (Z-Score icin)
        self.train_mean = None
        self.train_std = None
        if train_stats_path and os.path.exists(train_stats_path):
            stats = np.load(train_stats_path)
            self.train_mean = stats['mean']  # (12,)
            self.train_std = stats['std']    # (12,)

    def __len__(self):
        return len(self.ecg_ids)

    def __getitem__(self, idx):
        ecg_id = self.ecg_ids[idx]
        sinyal_yolu = os.path.join(self.sinyal_dizini, f"{ecg_id}.npy")
        sinyal = np.load(sinyal_yolu)  # (12, 2500) float32
        
        # Guvenlik: Sinyal tam 2500 sample degilse crop/pad yap
        target_len = config.TARGET_LENGTH
        if sinyal.shape[1] > target_len:
            start = (sinyal.shape[1] - target_len) // 2
            sinyal = sinyal[:, start:start + target_len]
        elif sinyal.shape[1] < target_len:
            padded = np.zeros((sinyal.shape[0], target_len), dtype=sinyal.dtype)
            pad = (target_len - sinyal.shape[1]) // 2
            padded[:, pad:pad + sinyal.shape[1]] = sinyal
            sinyal = padded
        
        # Augmentasyon (sadece train)
        if self.augment and self.augmentation:
            sinyal = self.augmentation(sinyal)

        # Lead-Wise Z-Score Normalizasyonu — PDF BOLUM 1 KRITIK
        if self.train_mean is not None and self.train_std is not None:
            # Train istatistikleriyle normalize et (DOGRU yontem)
            for c in range(sinyal.shape[0]):
                if self.train_std[c] > 1e-6:
                    sinyal[c] = (sinyal[c] - self.train_mean[c]) / self.train_std[c]
                else:
                    sinyal[c] = sinyal[c] - self.train_mean[c]
        else:
            # Fallback: Per-sample lead-wise (train stats yoksa)
            for c in range(sinyal.shape[0]):
                std = np.std(sinyal[c])
                if std > 1e-6:
                    sinyal[c] = (sinyal[c] - np.mean(sinyal[c])) / std

        sinyal_tensor = torch.FloatTensor(sinyal)
        etiket_tensor = torch.LongTensor([self.labels[idx]])[0]
        aux_etiket_tensor = torch.LongTensor([self.aux_labels[idx]])[0]
        domain_tensor = torch.LongTensor([self.domains[idx]])[0]
        
        # Wide Features — RAM'den aninda oku
        wide_features = torch.FloatTensor(self.wide_features_array[idx])

        return sinyal_tensor, wide_features, etiket_tensor, aux_etiket_tensor, domain_tensor

# =============================================================================
# DANN - GRADIENT REVERSAL LAYER
# =============================================================================

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

# =============================================================================
# SE-RESNET BLOKLARI
# =============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: Kanal bazinda agirliklandirma."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        reduced = max(channels // reduction, 4)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = downsample
        self.spatial_dropout = nn.Dropout(0.1)  # Spatial Dropout — CNN overfitting onleme

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.spatial_dropout(out)  # Feature map dropout
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


# =============================================================================
# ANA MODEL: CARDIOFUSION-5 — PDF Bolum 2 (Versiyon A + B Birlesik)
# =============================================================================

class CardioFusion5(nn.Module):
    """
    CardioFusion-5 Unified Model
    
    PDF'deki her iki versiyonun (Efficient + Pro) en iyi ozelliklerini birlestirir:
    - SE-ResNet Feature Extractor (Versiyon A — CNN + SE)
    - Transformer Encoder (Versiyon B — Zamansal)
    - Wide Features (Versiyon A — Precomputed fizyolojik)
    - DANN (Versiyon B — Domain Adversarial)
    - Multi-Task Loss (Her iki versiyon — 5+3 sinif)
    """
    def __init__(self, num_classes=5, num_aux_classes=3, num_domains=2, wide_feature_dim=8):
        super().__init__()
        
        # 1. Feature Extractor (SE-ResNet)
        self.in_channels = 64
        self.conv1 = nn.Conv1d(config.NUM_LEADS, 64, kernel_size=config.CNN_KERNEL_SIZE_1, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2, kernel_size=7)
        self.layer2 = self._make_layer(128, 2, kernel_size=7, stride=2)
        self.layer3 = self._make_layer(256, 2, kernel_size=7, stride=2)
        self.layer4 = self._make_layer(config.TRANSFORMER_DIM, 2, kernel_size=7, stride=2)
        
        # 2. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.TRANSFORMER_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dropout=config.TRANSFORMER_DROPOUT,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.TRANSFORMER_LAYERS)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 3. Main Classifier (Deep + Wide) — PDF: Dense(128) -> Dense(5)
        self.fc_deep = nn.Linear(config.TRANSFORMER_DIM, 128)
        self.fc_deep_drop = nn.Dropout(0.2)  # Deep features dropout
        
        # Wide Features normalizasyonu — overfitting onleme
        self.wide_bn = nn.BatchNorm1d(wide_feature_dim)
        self.wide_drop = nn.Dropout(0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + wide_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
        # 4. Auxiliary Classifier (Multi-Task) — PDF Bolum 8
        # 3-sinif: Normal / Ritim Bozuklugu / Iletim Bozuklugu
        self.aux_classifier = nn.Sequential(
            nn.Linear(config.TRANSFORMER_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_aux_classes)
        )
        
        # 5. Domain Classifier (DANN) — PDF Versiyon B
        self.domain_classifier = nn.Sequential(
            nn.Linear(config.TRANSFORMER_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_domains)
        )

    def _make_layer(self, out_channels, blocks, kernel_size, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, kernel_size, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x, wide_features=None, alpha=1.0):
        """
        Args:
            x: (batch, 12, 2500) — EKG sinyali
            wide_features: (batch, 8) — Fizyolojik ozellikler
            alpha: DANN gradient reversal katsayisi
            
        Returns:
            class_logits: (batch, 5) — Ana sinif tahminleri
            aux_logits: (batch, 3) — Yardimci sinif (Ritim/Iletim/Normal)
            domain_logits: (batch, 2) — Domain tahmini
        """
        # CNN
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Transformer
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        
        # Pooling
        deep_features = self.global_pool(x).squeeze(-1)  # (batch, 384)
        
        # Auxiliary Classification (Multi-Task) — dogrudan deep_features'tan
        aux_logits = self.aux_classifier(deep_features)
        
        # Domain Classification (DANN — Gradient Reversal)
        domain_features = grad_reverse(deep_features, alpha)
        domain_logits = self.domain_classifier(domain_features)
        
        # Main Classification (Deep + Wide)
        deep_out = self.fc_deep_drop(F.relu(self.fc_deep(deep_features)))
        if wide_features is not None:
            wide_norm = self.wide_drop(self.wide_bn(wide_features))  # BN + Dropout
            combined = torch.cat([deep_out, wide_norm], dim=1)
        else:
            dummy_wide = torch.zeros(deep_out.size(0), config.WIDE_FEATURE_DIM, device=deep_out.device)
            combined = torch.cat([deep_out, dummy_wide], dim=1)
            
        class_logits = self.classifier(combined)
        
        return class_logits, aux_logits, domain_logits


# =============================================================================
# FOCAL LOSS — PDF Bolum 6
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    gamma=0 -> Standart CrossEntropy
    gamma=2 -> Kolay orneklerin katkisi ~0.25x'e duser
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
        ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        return focal_loss.mean()


# =============================================================================
# YARDIMCI
# =============================================================================

def model_ozetini_yazdir(model):
    """Model parametre sayisini yazdirir."""
    toplam = sum(p.numel() for p in model.parameters())
    egitim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    boyut_mb = toplam * 4 / (1024 * 1024)
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
    print("BirunAI -- CardioFusion-5 Mimari Testi (Aux Head + DANN)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CardioFusion5().to(device)
    model_ozetini_yazdir(model)

    dummy_input = torch.randn(4, 12, 2500).to(device)
    dummy_wide = torch.randn(4, 8).to(device)
    
    class_logits, aux_logits, domain_logits = model(dummy_input, dummy_wide, alpha=0.1)
    print(f"\n  Main   Logits: {class_logits.shape} (Beklenen: 4, 5)")
    print(f"  Aux    Logits: {aux_logits.shape} (Beklenen: 4, 3)")
    print(f"  Domain Logits: {domain_logits.shape} (Beklenen: 4, 2)")
    print("\n  Mimari testi BASARILI!")
