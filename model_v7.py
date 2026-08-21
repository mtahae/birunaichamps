"""
model_v7.py — BirunAI EKG: CardioFusion-7 "Mixture-of-Experts"
================================================================

Literatur-temelli buyuk mimari revizyon (2026-07-20). Onceki 6+ atriyal-dal
denemesi HEP ayni hatayi yapti: kucuk ozellik vektorlerini TEK fusion
siniflandiricisina ekleyip baskin morfoloji ozelliklerinin yaninda bogdu.
Atriyal sinyal hicbir zaman KENDI BASINA ayirt edici olmaya zorlanmadi.

Arastirma bulgulari (PhysioNet + AFIB/AFL literaturu):
  1. Kazanan yaklasim (F1: AFib .95, AFL .90): AYRI UZMAN aglar, her biri KENDI
     sinifllandirmasini yapip KARAR duzeyinde birlesiyor (ozellik duzeyinde DEGIL).
  2. ConvNeXtV2-1D bloklari (depthwise conv + inverted bottleneck + LayerNorm +
     GELU + Global Response Norm) eski 1D-CNN'i geciyor (F1 .986, 770k param).
  3. Deep supervision: her uzman kendi gradyan sinyalini almali.

Tasarim (Mixture-of-Experts):
  - Morfoloji uzmani  : ConvNeXt1D backbone -> KENDI 5-sinif basligi
  - Ritim/Atriyal uzmani: QRS-baskili atriyal sinyal + spektral + otokorelasyon +
                          temporal conv -> KENDI 5-sinif basligi (AFIB/AFL adanmis)
  - Ogrenilen gate    : her ornek icin uzman logit'lerini agirlikli birlestirir
  - Deep supervision  : her uzman bagimsiz loss alir (atriyal ozellikler ayirt
                        edici olmaya ZORLANIR)

Interface: forward(x, wide, alpha, return_experts=False)
  return_experts=False -> (class_logits, aux_logits, domain_logits)  [drop-in v6 uyumlu]
  return_experts=True  -> (..., morph_logits, rhythm_logits)          [deep supervision icin]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from model_v6 import DropPath, grad_reverse, AttentionPooling, SpectralAtrialBranch


# =============================================================================
# ConvNeXt1D yapi taslari (LayerNorm channels-first + GRN)
# =============================================================================

class LayerNorm1d(nn.Module):
    """Channels-first (B, C, L) uzerinde LayerNorm — kanal ekseninde normalize."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None] * x + self.bias[None, :, None]


class GRN1d(nn.Module):
    """Global Response Normalization (ConvNeXtV2) — kanal-arasi rekabeti tesvik eder."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        # x: (B, C, L). Spatial (L) L2-norm per channel -> (B, C, 1)
        gx = torch.norm(x, p=2, dim=2, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXt1dBlock(nn.Module):
    """ConvNeXtV2-1D blok: depthwise conv -> LN -> pointwise(4x) -> GELU -> GRN -> pointwise."""
    def __init__(self, dim, drop_path=0.0, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.grn = GRN1d(4 * dim)
        self.pwconv2 = nn.Conv1d(4 * dim, dim, 1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return inp + self.drop_path(x)


class Downsample1d(nn.Module):
    """ConvNeXt downsampling: LayerNorm + stride-2 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = LayerNorm1d(in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


# =============================================================================
# RITIM / ATRIYAL UZMANI — kendi siniflandirma basligiyla
# =============================================================================

class RhythmExpert(nn.Module):
    """
    QRS-baskili atriyal sinyal uzerinde calisan, KENDI 5-sinif basligi olan uzman.
    Deep supervision ile dogrudan gradyan alir -> atriyal ozellikler ayirt edici
    olmaya ZORLANIR (onceki denemelerin bogulma sorununu cozer).

    Girdiler:
      - Atriyal leadler (II,III,aVF,V1), QRS-baskilanmis -> temporal ConvNeXt1D
        (atriyal dalga morfolojisi + beat-to-beat zamansal patern)
      - SpectralAtrialBranch: FFT spektrumu (flutter tepesi) + otokorelasyon (duzenlilik)
      - Wide fizyolojik ozellikler (rr_cv vb.)
    """
    def __init__(self, num_classes=5, wide_feature_dim=12, dims=(32, 64, 96)):
        super().__init__()
        self.atrial_leads = [1, 2, 5, 6]
        # QRS baskilama SpectralAtrialBranch'ten yeniden kullanilir (ayni mekanizma)
        self.spectral = SpectralAtrialBranch(out_dim=64)

        # Temporal atriyal CNN (QRS-baskili sinyal uzerinde)
        c0, c1, c2 = dims
        self.stem = nn.Sequential(
            nn.Conv1d(len(self.atrial_leads), c0, kernel_size=15, stride=4, padding=7),
            LayerNorm1d(c0),
        )
        self.stage1 = ConvNeXt1dBlock(c0, drop_path=0.05)
        self.down1 = Downsample1d(c0, c1)
        self.stage2 = ConvNeXt1dBlock(c1, drop_path=0.1)
        self.down2 = Downsample1d(c1, c2)
        self.stage3 = ConvNeXt1dBlock(c2, drop_path=0.1)
        self.temporal_pool = AttentionPooling(c2)

        # Wide MLP
        self.wide_bn = nn.BatchNorm1d(wide_feature_dim)
        self.wide_mlp = nn.Sequential(
            nn.Linear(wide_feature_dim, 32), nn.GELU(), nn.Dropout(0.3)
        )

        feat_dim = c2 + 64 + 32   # temporal + spectral(64) + wide(32)
        self.feat_dim = feat_dim
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 96), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(96, num_classes)
        )

    def forward(self, x, wide_features=None):
        # QRS-baskili atriyal sinyal (SpectralAtrialBranch._suppress_qrs yeniden kullan)
        leads = x[:, self.atrial_leads, :].float()
        leads_supp = self.spectral._suppress_qrs(leads)

        t = self.stem(leads_supp)
        t = self.stage1(t)
        t = self.down1(t)
        t = self.stage2(t)
        t = self.down2(t)
        t = self.stage3(t)
        temporal_feat = self.temporal_pool(t.permute(0, 2, 1))   # (B, c2)

        spec_feat = self.spectral(x)                              # (B, 64) FFT+otokorelasyon

        if wide_features is not None:
            wide = self.wide_mlp(self.wide_bn(wide_features))
        else:
            wide = torch.zeros(x.size(0), 32, device=x.device)

        feat = torch.cat([temporal_feat, spec_feat, wide], dim=1)
        logits = self.head(feat)
        return logits, feat


# =============================================================================
# ANA MODEL: CARDIOFUSION-7
# =============================================================================

class CardioFusion7(nn.Module):
    def __init__(self, num_classes=5, num_aux_classes=3, num_domains=2,
                 wide_feature_dim=12, dims=(48, 96, 192, 256), depths=(1, 1, 2, 2),
                 max_drop_path=0.2):
        super().__init__()
        c0, c1, c2, c3 = dims

        # --- Paylasilan stem: erken katman Instance Norm (domain robustlugu) ---
        self.stem = nn.Sequential(
            nn.Conv1d(config.NUM_LEADS, c0, kernel_size=15, stride=4, padding=7, bias=False),
            nn.InstanceNorm1d(c0, affine=True),
            nn.GELU(),
        )

        # --- Morfoloji uzmani: ConvNeXt1D backbone ---
        dp = torch.linspace(0, max_drop_path, sum(depths)).tolist()
        i = 0
        self.morph_stage1 = nn.Sequential(*[ConvNeXt1dBlock(c0, dp[i + k]) for k in range(depths[0])]); i += depths[0]
        self.morph_down1 = Downsample1d(c0, c1)
        self.morph_stage2 = nn.Sequential(*[ConvNeXt1dBlock(c1, dp[i + k]) for k in range(depths[1])]); i += depths[1]
        self.morph_down2 = Downsample1d(c1, c2)
        self.morph_stage3 = nn.Sequential(*[ConvNeXt1dBlock(c2, dp[i + k]) for k in range(depths[2])]); i += depths[2]
        self.morph_down3 = Downsample1d(c2, c3)
        self.morph_stage4 = nn.Sequential(*[ConvNeXt1dBlock(c3, dp[i + k]) for k in range(depths[3])])
        self.morph_pool = AttentionPooling(c3)
        self.morph_head = nn.Sequential(
            nn.Linear(c3, 128), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # --- Ritim/Atriyal uzmani (kendi basligiyla) ---
        self.rhythm_expert = RhythmExpert(num_classes=num_classes, wide_feature_dim=wide_feature_dim)

        # --- Gate: paylasilan morfoloji ozelliginden uzman agirliklari ---
        self.gate = nn.Sequential(
            nn.Linear(c3, 64), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(64, 2)   # [morfoloji, ritim] agirliklari
        )

        # --- Aux (Multi-Task) + Domain (DANN) — morfoloji deep ozelliginden ---
        self.aux_classifier = nn.Sequential(
            nn.Linear(c3, 64), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(64, num_aux_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(c3, 64), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(64, num_domains)
        )
        self.wide_feature_dim = wide_feature_dim

    def forward(self, x, wide_features=None, alpha=1.0, return_experts=False):
        # Paylasilan stem
        feat = self.stem(x)

        # Morfoloji uzmani
        feat = self.morph_stage1(feat)
        feat = self.morph_down1(feat)
        feat = self.morph_stage2(feat)
        feat = self.morph_down2(feat)
        feat = self.morph_stage3(feat)
        feat = self.morph_down3(feat)
        feat = self.morph_stage4(feat)               # (B, c3, T)
        morph_feat = self.morph_pool(feat.permute(0, 2, 1))   # (B, c3)
        morph_logits = self.morph_head(morph_feat)

        # Ritim uzmani (kendi basligi)
        rhythm_logits, _ = self.rhythm_expert(x, wide_features)

        # Gate: her ornek icin uzman agirliklari (softmax, toplam=1)
        gate_w = F.softmax(self.gate(morph_feat), dim=1)      # (B, 2)
        class_logits = (gate_w[:, 0:1] * morph_logits +
                        gate_w[:, 1:2] * rhythm_logits)        # (B, num_classes)

        # Aux + Domain (paylasilan morfoloji ozelliginden)
        aux_logits = self.aux_classifier(morph_feat)
        domain_logits = self.domain_classifier(grad_reverse(morph_feat, alpha))

        if return_experts:
            return class_logits, aux_logits, domain_logits, morph_logits, rhythm_logits
        return class_logits, aux_logits, domain_logits


def model_ozetini_yazdir(model):
    toplam = sum(p.numel() for p in model.parameters())
    egitim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CardioFusion-7 (MoE) Ozeti:")
    print(f"    Toplam parametre    : {toplam:,}")
    print(f"    Egitilebilir param  : {egitim:,}")
    print(f"    Model boyutu (GPU)  : ~{toplam * 4 / (1024*1024):.1f} MB")
    return toplam


if __name__ == "__main__":
    print("=" * 70)
    print("CardioFusion-7 (Mixture-of-Experts) Mimari Testi")
    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CardioFusion7().to(device)
    model_ozetini_yazdir(model)
    x = torch.randn(4, 12, 2500).to(device)
    w = torch.randn(4, 12).to(device)
    # 3-tuple (inference/val uyumu)
    cl, ax, dl = model(x, w, alpha=0.5)
    print(f"\n  [return_experts=False] Class:{cl.shape} Aux:{ax.shape} Domain:{dl.shape}")
    # 5-tuple (deep supervision)
    cl, ax, dl, ml, rl = model(x, w, alpha=0.5, return_experts=True)
    print(f"  [return_experts=True ] Morph:{ml.shape} Rhythm:{rl.shape}")
    loss = cl.sum() + ax.sum() + dl.sum() + ml.sum() + rl.sum()
    loss.backward()
    print("\n  Forward + Backward BASARILI!")
