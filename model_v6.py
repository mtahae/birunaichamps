"""
model_v6.py — BirunAI EKG: CardioFusion-6 "Lean-Robust"
=========================================================

Overfitting + AFIB/AFL platosunu kirmak icin tasarlanmis YENI mimari.

Felsefe: Daha BUYUK degil, daha YALIN + daha ROBUST + daha iyi GENELLEYEN.
Mevcut CardioFusion-5'in agir 2-katmanli Transformer'i ezberin ana kaynagiydi;
yerine parametre-verimli BiGRU konuldu ve domain-robustlik icin Instance Norm +
Stochastic Depth eklendi.

4 tamamlayici gorus (birbirini tekrar etmez):
    1. Instance-Norm SE-ResNet + Stochastic Depth  -> morfoloji (LBBB/RBBB)
    2. Cross-Lead Attention (V1<->V6, I<->aVL)      -> iletim bloklari
    3. BiGRU Rhythm                                  -> RR duzensizligi (AFIB/AFL)
    4. Wide fizyolojik ozellikler                   -> rr_cv, atrial_rate, p_reg

Interface (adim08 egitim loop'u ile drop-in uyumlu):
    forward(x, wide_features, alpha) -> (class_logits, aux_logits, domain_logits)

YASAKLAR (birunaikeremabi.md Bolum 6) korunur: mimari bunlari IHLAL ETMEZ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# =============================================================================
# STOCHASTIC DEPTH (DropPath) — derin aglar icin guclu regularizasyon
# =============================================================================

class DropPath(nn.Module):
    """Residual dalini egitimde p olasilikla tamamen dusurur (stochastic depth).
    Plain dropout'tan cok daha etkili bir genelleme regularizasyonu."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        # (B, 1, 1) per-sample maske
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep) * mask


# =============================================================================
# GRADIENT REVERSAL LAYER (DANN)
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
# SQUEEZE-AND-EXCITATION
# =============================================================================

class SEBlock(nn.Module):
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


# =============================================================================
# CROSS-LEAD ATTENTION — 12 derivasyon arasi fizyolojik iliski
# =============================================================================

class CrossLeadAttention(nn.Module):
    """
    Her derivasyonu ayri bir "token"a kodlar (paylasilan kucuk CNN ile),
    sonra 12 token arasinda multi-head self-attention uygular.
    V1'in V6'ya, I'nin aVL'ye "bakmasini" ogrenir — LBBB/RBBB icin kritik.

    Cikti: (B, d_lead) havuzlanmis derivasyon-iliski vektoru.
    """
    def __init__(self, d_lead=64, n_heads=4):
        super().__init__()
        # Paylasilan lead encoder — her derivasyon ayni agirliklarla islenir
        self.lead_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7, bias=False),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, d_lead, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(d_lead), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        # Ogrenilebilir lead (pozisyon) embedding — hangi token hangi derivasyon
        self.lead_embed = nn.Parameter(torch.randn(1, config.NUM_LEADS, d_lead) * 0.02)
        self.attn = nn.MultiheadAttention(d_lead, n_heads, dropout=0.2, batch_first=True)
        self.norm = nn.LayerNorm(d_lead)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """x: (B, 12, L)"""
        B, C, L = x.shape
        # Her derivasyonu ayri isle: (B*12, 1, L)
        x_leads = x.reshape(B * C, 1, L)
        tokens = self.lead_encoder(x_leads).squeeze(-1)   # (B*12, d_lead)
        tokens = tokens.view(B, C, -1)                    # (B, 12, d_lead)
        tokens = tokens + self.lead_embed                 # derivasyon kimligi
        attn_out, _ = self.attn(tokens, tokens, tokens)   # (B, 12, d_lead)
        tokens = self.norm(tokens + attn_out)             # residual
        pooled = tokens.mean(dim=1)                       # (B, d_lead)
        return self.dropout(pooled)


# =============================================================================
# SE-RESNET BLOK (Instance-Norm secenegi + Stochastic Depth)
# =============================================================================

class ResNetBlockV6(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, drop_path=0.0,
                 use_instance_norm=False):
        super().__init__()
        pad = kernel_size // 2
        Norm = (lambda c: nn.InstanceNorm1d(c, affine=True)) if use_instance_norm \
            else (lambda c: nn.BatchNorm1d(c))
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, pad, bias=False)
        self.n1 = Norm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, pad, bias=False)
        self.n2 = Norm(out_ch)
        self.se = SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                Norm(out_ch)
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.n1(self.conv1(x)))
        out = self.se(self.n2(self.conv2(out)))
        out = identity + self.drop_path(out)
        return self.relu(out)


# =============================================================================
# TEMPORAL ATTENTION POOLING
# =============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1)
        )

    def forward(self, x):
        """x: (B, T, dim)"""
        w = F.softmax(self.attention(x), dim=1)
        return (x * w).sum(dim=1)


# =============================================================================
# SPECTRAL ATRIAL BRANCH — AFIB vs AFL ayrimi icin frekans-alani dali
# =============================================================================

class SpectralAtrialBranch(nn.Module):
    """
    Atriyal aktivite leadlerinden (II=1, III=2, aVF=5, V1=6) frekans spektrumu cikarir.

    Fizyoloji (AFIB/AFL asil darbogaz):
        AFL  = 250-350/dk DUZENLI flutter -> spektrumda ~4-6 Hz'de KESKIN tepe
        AFIB = kaotik f-wave              -> GENIS BANTLI, tepesiz
        Normal/Sinus = ~1-1.7 Hz P dalgasi tepesi
    Zaman-alani CNN+GRU bu periyodiklik farkini (ozellikle pooling sonrasi)
    zor yakalar; FFT bu ayrimi DOGRUDAN modele verir. Dusuk parametreli,
    fizyolojik indüktif önyargi — overfit'i artirmaz.

    MaxPool (avg degil) ozellikle secildi: flutter'in KESKIN tepesini yakalar.

    QRS-BASKILAMA (kritik duzeltme): 0.5-15Hz bandi sqi.py'nin QRS-guc bandiyla
    (5-15Hz) birebir cakisir. QRS kompleksi genis-bantli, YUKSEK GENLIKLI bir
    transienttir (P/flutter dalgasindan 5-10x daha buyuk) ve bu bandi kolayca
    domine eder — spektral dal aslinda atriyal aktiviteyi degil QRS enerjisini
    goruyor olabilirdi. Cozum: FFT'den ONCE, genlik-zarfi medyan+MAD esigini asan
    (QRS'e ait) pencereler maskelenir (sifirlanir); kalan TQ-segment/baseline
    sinyali (P-dalgasi, flutter testeredisi, f-wave) FFT'ye girer. Esik hesabi
    no_grad icinde (sabit maske); carpma gradyani leads'e normal akar.

    OTOKORELASYON-TABANLI RITIM DUZENLILIGI (2026-07-20 eklendi): adim07b'deki
    R-tepe/P-tepe tabanli el-yapimi ozellikler (find_peaks VE ozenli Pan-Tompkins
    adaptif esik ile TEST EDILDI) AFIB/AFL ayrimini GUVENILMEZ yakaliyor (Cohen's
    d: eski find_peaks ~0.58-0.74, adaptif Pan-Tompkins ~0.17-0.43 — hatta DAHA
    KOTU). Sebep: HERHANGI bir sert "tepe var/yok" karari kirilgan. Otokorelasyon
    (sinyalin kendisiyle zaman-kaydirilmis kopyasina ne kadar benzedigi) hicbir
    tepe karari GEREKTIRMEDEN ayni bilgiyi cok daha saglam verir: duzenli ritim
    (Normal, sabit-blok AFL) fizyolojik RR araliginda (250-1500ms) KESKIN bir
    otokorelasyon tepesi verir; kaotik ritim (AFIB) hizla soner. Olculdu (n=20/
    sinif, TEKNOFEST): Cohen's d=0.807 (QRS-baskili) — TUM el-yapimi alternatiflerden
    daha iyi. Wiener-Khinchin teoremi ile ucretsiz hesaplanir: guc spektrumunun
    ters-FFT'si = otokorelasyon (zaten hesapladigimiz FFT'nin uzerine, ekstra
    R-tepe tespiti YOK).
    """
    def __init__(self, out_dim=48, fs=250, fmin=0.5, fmax=15.0, n_samples=2500,
                 qrs_mad_k=4.0, qrs_dilate_ms=120, rr_lag_min_ms=250, rr_lag_max_ms=1500,
                 use_autocorr=True):
        super().__init__()
        # use_autocorr: False -> otokorelasyon-oncesi (v6 seed42/123/2026 checkpoint
        # uyumu, fc girisi 48). True -> otokorelasyon dahil (v7, fc girisi 52).
        self.use_autocorr = use_autocorr
        self.atrial_leads = [1, 2, 5, 6]  # II, III, aVF, V1
        self.n_samples = n_samples
        freqs = torch.fft.rfftfreq(n_samples, d=1.0 / fs)   # (n_samples//2+1,)
        band = (freqs >= fmin) & (freqs <= fmax)
        self.register_buffer('band_mask', band)
        self.qrs_mad_k = qrs_mad_k
        self.env_k = 9                                      # ~36ms zarf yumusatma @250Hz
        self.dilate_k = max(3, int(qrs_dilate_ms * fs / 1000) | 1)  # tek sayi kernel
        self.ac_lag_min = int(rr_lag_min_ms * fs / 1000)     # fizyolojik RR araligi (samples)
        self.ac_lag_max = min(int(rr_lag_max_ms * fs / 1000), n_samples - 1)
        self.spec_cnn = nn.Sequential(
            nn.Conv1d(len(self.atrial_leads), 32, 7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 48, 5, padding=2, bias=False),
            nn.BatchNorm1d(48), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),   # tepe-yakalama (flutter)
        )
        self.ac_pool = nn.AdaptiveMaxPool1d(1)                # her lead icin en yuksek duzenlilik
        fc_in = 48 + (len(self.atrial_leads) if use_autocorr else 0)  # spec(48) + otokorelasyon(4)
        self.fc = nn.Sequential(nn.Linear(fc_in, out_dim), nn.ReLU(inplace=True), nn.Dropout(0.3))
        self.out_dim = out_dim

    def _suppress_qrs(self, leads):
        """leads: (B, 4, L) -> ayni sekil, QRS pencereleri sifirlanmis."""
        with torch.no_grad():
            env = F.avg_pool1d(leads.abs(), kernel_size=self.env_k, stride=1,
                               padding=self.env_k // 2)
            med = env.median(dim=-1, keepdim=True).values
            mad = (env - med).abs().median(dim=-1, keepdim=True).values + 1e-6
            is_qrs = (env > med + self.qrs_mad_k * mad).float()
            # QRS penceresini tam genislige (~120ms) genislet (dilate)
            qrs_mask = F.max_pool1d(is_qrs, kernel_size=self.dilate_k, stride=1,
                                    padding=self.dilate_k // 2)
            keep = 1.0 - qrs_mask
            # kenar sicramasini (spektral sizinti) yumusat
            keep = F.avg_pool1d(keep, kernel_size=5, stride=1, padding=2)
        return leads * keep

    def forward(self, x):
        """x: (B, 12, L)"""
        leads = x[:, self.atrial_leads, :].float()          # (B, 4, L)
        leads = self._suppress_qrs(leads)                   # QRS'i baskila -> atriyal residual
        # FFT autocast disinda (fp16 desteklemez) — fp32'de hesapla
        with torch.autocast(device_type=x.device.type, enabled=False):
            fft_full = torch.fft.rfft(leads, dim=-1)        # (B, 4, F) complex
            mag = fft_full.abs()

            # --- Spektral yol: flutter'in keskin frekans tepesi ---
            spec = mag[:, :, self.band_mask]                # (B, 4, n_bins)
            spec = torch.log1p(spec)                        # log-magnitude (stabilize)

            # --- Otokorelasyon yolu (opsiyonel): ritim duzenliligi ---
            if self.use_autocorr:
                # Wiener-Khinchin: guc spektrumunun ters-FFT'si = otokorelasyon
                power = mag ** 2
                autocorr = torch.fft.irfft(power, n=self.n_samples, dim=-1)
                autocorr = autocorr / (autocorr[:, :, :1] + 1e-9)            # lag=0'a normalize
                ac_window = autocorr[:, :, self.ac_lag_min:self.ac_lag_max]  # fizyolojik RR araligi

        feat_spec = self.spec_cnn(spec).squeeze(-1)         # (B, 48)
        if self.use_autocorr:
            feat_ac = self.ac_pool(ac_window).squeeze(-1)    # (B, 4)
            feat = torch.cat([feat_spec, feat_ac], dim=1)    # (B, 52)
        else:
            feat = feat_spec                                 # (B, 48) — v6 checkpoint uyumu
        return self.fc(feat)                                 # (B, out_dim)


# =============================================================================
# ANA MODEL: CARDIOFUSION-6
# =============================================================================

class CardioFusion6(nn.Module):
    def __init__(self, num_classes=5, num_aux_classes=3, num_domains=2,
                 wide_feature_dim=12, channels=(64, 128, 192, 256),
                 max_drop_path=0.3, gru_hidden=128):
        super().__init__()
        c1, c2, c3, c4 = channels

        # --- Stem: ilk katmanlar Instance Norm (domain/genlik farkini siler) ---
        self.stem = nn.Sequential(
            nn.Conv1d(config.NUM_LEADS, c1, kernel_size=15, stride=2, padding=7, bias=False),
            nn.InstanceNorm1d(c1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # --- SE-ResNet govdesi: erken katman Instance Norm, gec katman BatchNorm ---
        # Stochastic depth olasiligi derinlikle 0 -> max_drop_path artar
        dp = torch.linspace(0, max_drop_path, 4).tolist()
        self.layer1 = self._stage(c1, c1, stride=1, drop_path=dp[0], inst=True)
        self.layer2 = self._stage(c1, c2, stride=2, drop_path=dp[1], inst=True)
        self.layer3 = self._stage(c2, c3, stride=2, drop_path=dp[2], inst=False)
        self.layer4 = self._stage(c3, c4, stride=2, drop_path=dp[3], inst=False)

        # --- Rhythm: BiGRU (Transformer yerine — hafif, ezberi az) ---
        self.gru = nn.GRU(c4, gru_hidden, num_layers=1, batch_first=True,
                          bidirectional=True)
        self.gru_drop = nn.Dropout(0.3)
        self.temporal_pool = AttentionPooling(2 * gru_hidden)  # BiGRU -> 2*hidden

        # --- Morfoloji: govde ciktisinin dogrudan attention pool'u ---
        self.morph_pool = AttentionPooling(c4)

        # --- Cross-Lead Attention ---
        self.cross_lead = CrossLeadAttention(d_lead=64, n_heads=4)

        # --- Spectral Atrial Branch (AFIB vs AFL) ---
        # use_autocorr=False: seed42/123/2026 checkpoint uyumu (otokorelasyon-oncesi,
        # zaten ise yaramamisti — autocorr777 0.8491). v7 otokorelasyonu kullanir.
        self.spectral = SpectralAtrialBranch(out_dim=48, use_autocorr=False)

        # --- Wide fizyolojik ozellikler (BN + kucuk MLP ile daha etkin) ---
        self.wide_bn = nn.BatchNorm1d(wide_feature_dim)
        self.wide_mlp = nn.Sequential(
            nn.Linear(wide_feature_dim, 32), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )

        # --- Fusion: rhythm(2*gru) + morph(c4) + lead(64) + spectral(48) + wide_mlp(32) ---
        fusion_dim = 2 * gru_hidden + c4 + 64 + self.spectral.out_dim + 32
        self.fusion_dim = fusion_dim
        self.wide_feature_dim = wide_feature_dim

        # --- Main Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 192), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(192, 64), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

        # Aux (Multi-Task) ve Domain (DANN) — deep morfoloji+ritim ozelliginden
        deep_dim = 2 * gru_hidden + c4
        self.aux_classifier = nn.Sequential(
            nn.Linear(deep_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(64, num_aux_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(deep_dim, 64), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(64, num_domains)
        )

    def _stage(self, in_ch, out_ch, stride, drop_path, inst):
        return nn.Sequential(
            ResNetBlockV6(in_ch, out_ch, 7, stride, drop_path, inst),
            ResNetBlockV6(out_ch, out_ch, 7, 1, drop_path, inst),
        )

    def forward(self, x, wide_features=None, alpha=1.0):
        # Cross-Lead Attention — ham sinyalden derivasyon iliskisi
        lead_feat = self.cross_lead(x)                # (B, 64)

        # Morfoloji govdesi
        feat = self.stem(x)
        feat = self.layer1(feat)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        feat = self.layer4(feat)                      # (B, c4, T)

        morph_feat = self.morph_pool(feat.permute(0, 2, 1))   # (B, c4)

        # Ritim: BiGRU zaman ekseninde
        seq = feat.permute(0, 2, 1)                   # (B, T, c4)
        gru_out, _ = self.gru(seq)                    # (B, T, 2*hidden)
        gru_out = self.gru_drop(gru_out)
        rhythm_feat = self.temporal_pool(gru_out)     # (B, 2*hidden)

        # Deep ozellik (aux + domain icin)
        deep = torch.cat([rhythm_feat, morph_feat], dim=1)
        aux_logits = self.aux_classifier(deep)
        domain_logits = self.domain_classifier(grad_reverse(deep, alpha))

        # Spectral atrial ozellik (AFIB/AFL)
        spec_feat = self.spectral(x)                  # (B, 48)

        # Wide ozellikler -> BN -> MLP
        if wide_features is not None:
            wide = self.wide_mlp(self.wide_bn(wide_features))   # (B, 32)
        else:
            wide = torch.zeros(x.size(0), 32, device=x.device)

        fused = torch.cat([rhythm_feat, morph_feat, lead_feat, spec_feat, wide], dim=1)
        class_logits = self.classifier(fused)
        return class_logits, aux_logits, domain_logits


def model_ozetini_yazdir(model):
    toplam = sum(p.numel() for p in model.parameters())
    egitim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  CardioFusion-6 Ozeti:")
    print(f"    Toplam parametre    : {toplam:,}")
    print(f"    Egitilebilir param  : {egitim:,}")
    print(f"    Model boyutu (GPU)  : ~{toplam * 4 / (1024*1024):.1f} MB")
    return toplam


if __name__ == "__main__":
    print("=" * 70)
    print("CardioFusion-6 Mimari Testi")
    print("=" * 70)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CardioFusion6().to(device)
    model_ozetini_yazdir(model)
    x = torch.randn(4, 12, 2500).to(device)
    w = torch.randn(4, 12).to(device)
    cl, ax, dl = model(x, w, alpha=0.5)
    print(f"\n  Class : {cl.shape} (beklenen 4,5)")
    print(f"  Aux   : {ax.shape} (beklenen 4,3)")
    print(f"  Domain: {dl.shape} (beklenen 4,2)")
    # Backward testi
    loss = cl.sum() + ax.sum() + dl.sum()
    loss.backward()
    print("\n  Forward + Backward BASARILI!")
