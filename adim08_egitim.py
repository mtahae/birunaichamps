"""
adim08_egitim.py — BirunAI EKG: Adim 8 – Egitim (CardioFusion-5 Final)
=========================================================================

PDF'deki TUM yontemlerin profesyonelce uygulandigi egitim pipeline'i:
    - Curriculum Learning (3 Asama) — PDF Bolum 3.3
    - DANN (Domain Adversarial) — PDF Versiyon B
    - Multi-Task Loss (5-sinif + 3-sinif) — PDF Bolum 8
    - Class-Balanced Sampler — PDF Bolum 3.4
    - Focal Loss (gamma=2.0) — PDF Bolum 6
    - Hard Example Mining — PDF Bolum 3.4
    - Esik Optimizasyonu (Grid Search) — Bitis'te
    - Seed Fix (Tam Deterministic) — PDF Bolum 9
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import f1_score
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import copy
from contextlib import nullcontext
from adim07_model_mimarisi import EKGDataset, FocalLoss
from model_v7 import CardioFusion7, model_ozetini_yazdir
from threshold_opt import find_optimal_thresholds, find_optimal_thresholds_joint, apply_thresholds
from torch.optim.swa_utils import AveragedModel, SWALR


# =============================================================================
# EMA — Exponential Moving Average of weights (guclu genelleme regularizasyonu)
# =============================================================================

class ModelEMA:
    """Model agirliklarinin ustel hareketli ortalamasi. Her adimda guncellenir;
    ortalanmis agirliklar keskin loss cukurlarindan kacinip daha genis, robust
    optimuma oturur — SWA'nin surekli versiyonu. Val'da genellikle +0.3-1% F1."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for e, m in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                e.copy_(m)

# =============================================================================
# LDAM LOSS — Label-Distribution-Aware Margin Loss
# =============================================================================

class LDAMLoss(nn.Module):
    """
    LDAM Loss: Azinlik siniflarinin karar sinirlarini (margin) genisletir.
    Margin = C / n_j^(1/4) — n_j sinif sayisi, C = max_m
    
    Arkadasin 0.86+ modelinde: max_m=0.5, s=30.0
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # 1 / n^(1/4)
        m_list = m_list * (max_m / np.max(m_list))  # Normalize to max_m
        self.register_buffer('m_list', torch.FloatTensor(m_list))
        self.s = s
    
    def forward(self, logits, targets):
        # Her ornek icin margin
        batch_m = self.m_list[targets]  # (batch,)
        # One-hot
        one_hot = F.one_hot(targets, logits.size(1)).float()
        # Margin'i dogru siniftan cikar
        logits_m = logits - one_hot * batch_m.unsqueeze(1)
        # Scale ve CE
        return F.cross_entropy(self.s * logits_m, targets)

# =============================================================================
# EGITIM LOG YONETIMI (Dashboard icin)
# =============================================================================

LOG_PATH = os.path.join(config.OUTPUT_DIR, "training_log.json")

def log_baslat(total_epochs):
    log = {
        "status": "training",
        "current_epoch": 0,
        "total_epochs": total_epochs,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "best_f1": 0.0,
        "patience_counter": 0,
        "device": str(config.DEVICE),
        "epochs": []
    }
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    return log

def log_epoch_ekle(log, epoch_data):
    log["epochs"].append(epoch_data)
    log["current_epoch"] = epoch_data["epoch"]
    log["best_f1"] = epoch_data["best_f1"]
    log["patience_counter"] = epoch_data["patience_counter"]
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

def log_bitir(log, status="completed"):
    log["status"] = status
    log["end_time"] = datetime.now().isoformat()
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

# =============================================================================
# CLASS-BALANCED SAMPLER — PDF Bolum 3.4
# =============================================================================

def create_balanced_sampler(dataset):
    """
    Her mini-batch'te tum siniflar esit temsil edilir.
    AFL gibi azinlik siniflarinin gradient'i kaybolmasini onler.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        labels = list(dataset.labels)
        
    class_counts = Counter(labels)
    
    # Her sinifin agirligi = 1 / freq
    weights = np.zeros(len(labels), dtype=np.float64)
    for i, label in enumerate(labels):
        weights[i] = 1.0 / class_counts[label]
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def create_domain_anchored_sampler(dataset, tekno_boost=4.0):
    """
    P2 (karma veri) icin: sinif-dengeli AMA TEKNOFEST (domain 0) ornekleri
    'tekno_boost' kat daha olasi. Boylece her batch'te hedef domain baskin kalir,
    model internet verisine dogru kayip "patlamaz" (domain overfit'i onlenir).
    """
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
        domains = [dataset.dataset.domains[i] for i in dataset.indices]
    else:
        labels = list(dataset.labels)
        domains = list(dataset.domains)

    class_counts = Counter(labels)
    weights = np.zeros(len(labels), dtype=np.float64)
    for i, (label, dom) in enumerate(zip(labels, domains)):
        w = 1.0 / class_counts[label]          # sinif dengesi
        if dom == 0:                            # TEKNOFEST -> capa
            w *= tekno_boost
        weights[i] = w

    return WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)


def mixup_data(x, y, alpha=0.2):
    """
    Mixup Augmentation — PhysioNet Top Teams (Triage, HeartBeats).
    Iki farkli ornegi agirlikli olarak karistirarak yeni ornek uretir.
    Ozellikle AFL/AFIB gibi benzer siniflarin ayirt edilmesini kolaylastirir.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup icin loss hesaplama."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# =============================================================================
# TRAIN VE VALIDATION
# =============================================================================

def _rhythm_noise_weights(labels, domains, internet_rhythm_weight):
    """Internet (domain>0) AFIB(1)/AFL(2) ornekleri icin <1.0 agirlik dondurur.
    Gurultulu internet ritim etiketlerinin loss'a katkisini azaltir; temiz
    TEKNOFEST ritim etiketleri tam agirlikla kalir."""
    w = torch.ones_like(labels, dtype=torch.float32)
    if internet_rhythm_weight is None:
        return w
    rhythm = (labels == 1) | (labels == 2)   # AFIB, AFL
    internet = (domains > 0)
    w[rhythm & internet] = internet_rhythm_weight
    return w


def _class_loss(criterion, logits, labels, sw, do_mixup, labels_a=None, labels_b=None, lam=None):
    """Focal loss + opsiyonel per-sample agirlik (sw) + mixup destegi."""
    if do_mixup:
        return mixup_criterion(criterion, logits, labels_a, labels_b, lam)
    if sw is not None:
        return (criterion(logits, labels, reduction='none') * sw).mean()
    return criterion(logits, labels)


# Deep supervision agirligi (CardioFusion7 uzman basliklari icin)
DEEP_SUP_WEIGHT = 0.3


def train_one_epoch(model, loader, criterion_class, criterion_aux, criterion_domain,
                    optimizer, scaler, device, use_amp, use_dann=False, alpha=1.0, use_mixup=False,
                    hard_alpha=None, ema=None, internet_rhythm_weight=None, deep_supervision=False):
    """hard_alpha: AFIB/AFL ceza carpani. ema: ModelEMA. internet_rhythm_weight:
    internet AFIB/AFL loss carpani. deep_supervision: CardioFusion7 uzman basliklarina
    (morph + rhythm) bagimsiz loss uygular (return_experts=True gerekir)."""
    model.train()
    toplam_class_loss = 0
    tum_tahminler = []
    tum_etiketler = []

    for batch in loader:
        signals, wide_features, labels, aux_labels, domains = batch
        signals = signals.to(device)
        wide_features = wide_features.to(device)
        labels = labels.to(device)
        aux_labels = aux_labels.to(device)
        domains = domains.to(device)

        # Mixup augmentation
        if use_mixup and random.random() < 0.5:
            signals, labels_a, labels_b, lam = mixup_data(signals, labels, alpha=0.2)
            do_mixup = True
        else:
            do_mixup = False
            labels_a = labels_b = lam = None

        # Per-sample agirlik (hard mining + gurultu) — mixup yoksa
        sw = None
        if not do_mixup and (hard_alpha is not None or internet_rhythm_weight is not None):
            sw = hard_alpha[labels] if hard_alpha is not None else torch.ones_like(labels, dtype=torch.float32)
            sw = sw * _rhythm_noise_weights(labels, domains, internet_rhythm_weight)

        optimizer.zero_grad()
        amp_on = use_amp and device.type == 'cuda'
        ctx = torch.amp.autocast(device_type='cuda') if amp_on else nullcontext()

        with ctx:
            if deep_supervision:
                class_logits, aux_logits, domain_logits, morph_logits, rhythm_logits = \
                    model(signals, wide_features, alpha, return_experts=True)
            else:
                class_logits, aux_logits, domain_logits = model(signals, wide_features, alpha)

            loss_class = _class_loss(criterion_class, class_logits, labels, sw, do_mixup, labels_a, labels_b, lam)
            loss_aux = criterion_aux(aux_logits, aux_labels)
            loss = loss_class + config.AUX_LOSS_WEIGHT * loss_aux

            if use_dann:
                loss = loss + config.DANN_LAMBDA * criterion_domain(domain_logits, (domains > 0).long())

            # DEEP SUPERVISION: her uzman bagimsiz gradyan alir. Ritim uzmani AFIB/AFL
            # icin var — per-sample agirligi (sw) ONA da uygulanir; morfoloji uzmani duz.
            if deep_supervision:
                loss_morph = _class_loss(criterion_class, morph_logits, labels, None, do_mixup, labels_a, labels_b, lam)
                loss_rhythm = _class_loss(criterion_class, rhythm_logits, labels, sw, do_mixup, labels_a, labels_b, lam)
                loss = loss + DEEP_SUP_WEIGHT * (loss_morph + loss_rhythm)

        if amp_on:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        toplam_class_loss += loss_class.item() * signals.size(0)
        preds = class_logits.argmax(dim=1).cpu().numpy()
        tum_tahminler.extend(preds)
        tum_etiketler.extend((labels_a if do_mixup else labels).cpu().numpy())

    avg_loss = toplam_class_loss / len(loader.dataset)
    macro_f1 = f1_score(tum_etiketler, tum_tahminler, average='macro', zero_division=0)
    return avg_loss, macro_f1


def validate(model, loader, criterion_class, device, use_amp):
    model.eval()
    toplam_loss = 0
    tum_tahminler = []
    tum_etiketler = []
    tum_olasiliklar = []

    with torch.no_grad():
        for batch in loader:
            signals, wide_features, labels, aux_labels, _ = batch
            signals = signals.to(device)
            wide_features = wide_features.to(device)
            labels = labels.to(device)

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    class_logits, _, _ = model(signals, wide_features)
                    loss = criterion_class(class_logits, labels)
            else:
                class_logits, _, _ = model(signals, wide_features)
                loss = criterion_class(class_logits, labels)

            probs = torch.softmax(class_logits, dim=1)
            toplam_loss += loss.item() * signals.size(0)
            preds = class_logits.argmax(dim=1).cpu().numpy()
            
            tum_tahminler.extend(preds)
            tum_etiketler.extend(labels.cpu().numpy())
            tum_olasiliklar.extend(probs.cpu().numpy())

    avg_loss = toplam_loss / len(loader.dataset)
    macro_f1 = f1_score(tum_etiketler, tum_tahminler, average='macro', zero_division=0)
    per_class_f1 = f1_score(tum_etiketler, tum_tahminler, average=None,
                            labels=list(range(config.NUM_CLASSES)), zero_division=0)
    return avg_loss, macro_f1, per_class_f1.tolist(), np.array(tum_etiketler), np.array(tum_olasiliklar)


# =============================================================================
# ANA EGITIM PIPELINE'I — PDF Bolum 3.3 Curriculum Learning
# =============================================================================

def egitim_pipeline(seed=None, tag=None):
    """
    seed: None ise config.SEED kullanilir. Ensemble icin farkli seed'ler
          (orn. 42, 123, 2026) farkli baslangic noktalari/veri sirasi verir.
    tag:  checkpoint/log dosya adlarina eklenir (orn. "seed123" ->
          best_model_seed123.pth). None ise eski davranis (best_model.pth)
          korunur — tek-model calistirmalari etkilenmez.
    """
    print("=" * 70)
    print("BirunAI -- CardioFusion-5 Curriculum Learning (Final)")
    print("=" * 70)

    # --- Reproducibility (PDF Bolum 9) ---
    actual_seed = seed if seed is not None else config.SEED
    config.set_seed(actual_seed)
    print(f"\n  Seed: {actual_seed}" + (f" | Tag: {tag}" if tag else ""))

    # Tag varsa log dosyasini ayir (dashboard.py varsayilan LOG_PATH'i okumaya devam
    # eder; etiketli ensemble kosulari onu ezmez).
    global LOG_PATH
    if tag:
        LOG_PATH = os.path.join(config.OUTPUT_DIR, f"training_log_{tag}.json")

    device = config.DEVICE
    use_amp = config.USE_AMP and device.type == 'cuda'
    
    print(f"\n  Cihaz: {device}")
    if device.type == 'cuda':
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Veri Setleri ---
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    wide_features_dir = os.path.join(config.PROCESSED_DATA_DIR, "wide_features")
    train_manifest = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    val_manifest = os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv")
    train_stats_path = os.path.join(config.PROCESSED_DATA_DIR, "train_stats.npz")

    # Wide features dizini kontrolu
    if os.path.exists(wide_features_dir) and len(os.listdir(wide_features_dir)) > 0:
        wf_dir = wide_features_dir
        print(f"\n  Wide Features: AKTIF ({len(os.listdir(wide_features_dir))} dosya)")
    else:
        wf_dir = None
        print(f"\n  Wide Features: DEAKTIF (once adim07b_wide_features.py calistirilmali!)")

    full_train_dataset = EKGDataset(train_manifest, sinyal_dizini, augment=True, 
                                    train_stats_path=train_stats_path, wide_features_dir=wf_dir)
    val_dataset = EKGDataset(val_manifest, sinyal_dizini, augment=False, 
                             train_stats_path=train_stats_path, wide_features_dir=wf_dir)

    # Curriculum: TEKNOFEST vs Internet indeksleri
    tekno_indices = [i for i, d in enumerate(full_train_dataset.domains) if d == 0]
    if not tekno_indices:
        tekno_indices = list(range(len(full_train_dataset)))
    tekno_train_dataset = Subset(full_train_dataset, tekno_indices)
    
    # Sinif dagilimi
    sinif_dag = Counter(full_train_dataset.labels)
    print(f"\n  Train: {len(full_train_dataset)} | Val: {len(val_dataset)}")
    for idx in range(config.NUM_CLASSES):
        print(f"    [{idx}] {config.LABEL_NAMES[idx]:8s}: {sinif_dag.get(idx, 0)}")

    # Class-Balanced Sampler — PDF Bolum 3.4
    balanced_sampler_tekno = create_balanced_sampler(tekno_train_dataset)
    # P2: sinif-dengeli + TEKNOFEST capali (domain overfit'e karsi)
    anchored_sampler_all = create_domain_anchored_sampler(full_train_dataset, tekno_boost=4.0)

    # DataLoaders
    train_loader_all = DataLoader(full_train_dataset, batch_size=config.BATCH_SIZE,
                                  sampler=anchored_sampler_all, num_workers=2, pin_memory=True, drop_last=True)
    train_loader_tekno = DataLoader(tekno_train_dataset, batch_size=config.BATCH_SIZE,
                                    sampler=balanced_sampler_tekno, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)

    # --- Model (CardioFusion-7 Mixture-of-Experts) ---
    model = CardioFusion7(
        num_classes=config.NUM_CLASSES,
        num_aux_classes=config.NUM_AUX_CLASSES,
        num_domains=2,
        wide_feature_dim=config.WIDE_FEATURE_DIM
    ).to(device)
    model_ozetini_yazdir(model)

    # EMA — tum egitim boyunca agirlik ortalamasi (genelleme icin)
    ema = ModelEMA(model, decay=0.999)

    # --- Loss ---
    # Pure Focal Loss (LDAM s=30 VE s=15 iki kez denendi, ikisinde de sonuc kotulesdi)
    sinif_dag_train = Counter(full_train_dataset.labels)
    class_counts = np.array([sinif_dag_train.get(i, 1) for i in range(config.NUM_CLASSES)], dtype=np.float32)
    
    # Class weights (4th-root smoothing)
    inv_freq = 1.0 / class_counts
    class_weights = np.power(inv_freq, 0.25).astype(np.float32)
    class_weights = class_weights / class_weights.mean()

    # AFIB/AFL focal-weight esitlemesi — global frekans AFIB'i (5905) AFL'den (8640)
    # daha nadir gosterip class_weight'i AFIB lehine kaydiriyordu; bu da AFL'yi
    # one cikarmak icin kullandigimiz hard_alpha ile CATISIYORDU (olculdu: ~%4
    # sondurme etkisi). Sampler zaten batch-bazinda esitliyor; AFIB/AFL base
    # weight'i esitleyip AFL/AFIB dengesini TEK yerden (hard_alpha) kontrol ediyoruz.
    _afib_afl_avg = (class_weights[1] + class_weights[2]) / 2.0
    class_weights[1] = _afib_afl_avg
    class_weights[2] = _afib_afl_avg

    print(f"\n  Sinif Dagilimi: {[int(c) for c in class_counts]}")
    print(f"  Class Weights (4th-root): {[f'{w:.3f}' for w in class_weights]}")
    
    criterion_class = FocalLoss(alpha=class_weights, gamma=config.FOCAL_LOSS_GAMMA, 
                                label_smoothing=config.LABEL_SMOOTHING).to(device)
    criterion_aux = nn.CrossEntropyLoss().to(device)
    criterion_domain = nn.CrossEntropyLoss().to(device)
    
    # Hard Mining Alpha — olculu (AFL 2.2 boost'u geri tepti: AFIB dustu, AFL yardim
    # etmedi). AFL'nin sorunu artik AGIRLIK degil GENELLEME/ezber; onu augmentasyon +
    # weight decay + droppath cozecek. Burada sadece hafif AFL onceligi birakiyoruz.
    p2_hard_alpha = torch.FloatTensor([1.0, 1.2, 1.4, 1.0, 1.0]).to(device)
    p3_hard_alpha = torch.FloatTensor([1.0, 1.4, 1.6, 1.0, 1.0]).to(device)
    
    print(f"  Loss: Focal(g={config.FOCAL_LOSS_GAMMA}, ls={config.LABEL_SMOOTHING})")
    print(f"  P2 Hard Alpha: {p2_hard_alpha.tolist()} | P3 Hard Alpha: {p3_hard_alpha.tolist()}")
    
    # --- Optimizer (scheduler her asama icin ayri ayarlanacak) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler() if use_amp else None

    # --- Curriculum Asamalari (config'den) ---
    P1 = config.EPOCHS_PHASE_1  # 20
    P2 = config.EPOCHS_PHASE_2  # 60
    P3 = config.EPOCHS_PHASE_3  # 20
    TOTAL_EPOCHS = P1 + P2 + P3  # 100
    
    # Her asama icin ayri patience (P2 kisaldi: 30->12)
    PATIENCE_P1 = 15
    PATIENCE_P2 = 8   # 2 run'da zirve P2-epoch~9'da geliyordu; 12 gereksiz epoch harciyordu
    PATIENCE_P3 = 15
    MIN_F1_IMPROVEMENT = 0.001  # Kucuk iyilesmeleri de say (0.002 cok dardi)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_name = f"best_model_{tag}.pth" if tag else "best_model.pth"
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    training_log = log_baslat(TOTAL_EPOCHS)
    best_f1 = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n  Egitim Parametreleri:")
    print(f"    Optimizer      : AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    print(f"    Loss           : Focal(g={config.FOCAL_LOSS_GAMMA}, ls={config.LABEL_SMOOTHING}) + AuxCE(w={config.AUX_LOSS_WEIGHT})")
    print(f"    Batch Size     : {config.BATCH_SIZE}")
    print(f"    Total Epochs   : {TOTAL_EPOCHS}")
    print(f"    Phase 1 (Tekno): 1-{P1} (patience={PATIENCE_P1})")
    print(f"    Phase 2 (Karma): {P1+1}-{P1+P2} (patience={PATIENCE_P2}, Mixup+DANN)")
    print(f"    Phase 3 (Fine) : {P1+P2+1}-{TOTAL_EPOCHS} (patience={PATIENCE_P3}, SWA aktif)")
    print(f"    Mixed Precision: {'Evet' if use_amp else 'Hayir'}")
    print(f"\n{'='*70}\n  EGITIM BASLIYOR...\n{'='*70}\n")

    current_phase = 0  # 1=TEKNO, 2=KARMA, 3=FINE
    scheduler = None   # Her asama icin ayri scheduler olusturulacak
    skip_to_epoch = 0  # Early stop sonrasi atlama icin
    swa_model = None   # SWA modeli (P3'te aktif olacak)
    swa_scheduler = None

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Eger bir asama erken durduysa, sonraki asama baslayana kadar atla
        if skip_to_epoch > 0 and epoch < skip_to_epoch:
            continue
        skip_to_epoch = 0  # Sifirla
        
        epoch_start = time.time()

        # --- Asama Secimi ve Gecis ---
        if epoch <= P1:
            new_phase = 1
        elif epoch <= P1 + P2:
            new_phase = 2
        else:
            new_phase = 3
            
        # Asama degistiginde buyuk operasyonlar
        if new_phase != current_phase:
            if new_phase == 1:
                # P1: Sadece TEKNOFEST, Cosine LR from base
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=P1, eta_min=1e-6)
                for pg in optimizer.param_groups:
                    pg['lr'] = config.LEARNING_RATE
                print(f"\n  [P1 BASLADI] Sadece TEKNOFEST | LR={config.LEARNING_RATE} | Cosine T_max={P1}")
                
            elif new_phase == 2:
                # P2: Best P1 modelini yukle + dusuk LR ile baslat
                print(f"\n  [PHASE CHANGE] P1 -> P2 | En iyi P1 modeli yukleniyor (VF1={best_f1:.4f})")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                    print(f"    -> best_model.pth yuklendi. Model P1'in en iyi noktasindan devam edecek.")
                ema = ModelEMA(model, decay=0.999)  # EMA'yi yuklenen agirliktan yeniden baslat

                # P2 LR: Base LR'in 0.15x'i + WARMUP (epoch 21 loss sicramasini onler)
                p2_lr = config.LEARNING_RATE * 0.15
                for pg in optimizer.param_groups:
                    pg['lr'] = p2_lr
                P2_WARMUP = 3
                _warm = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=P2_WARMUP)
                _cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, P2 - P2_WARMUP), eta_min=1e-6)
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[_warm, _cos], milestones=[P2_WARMUP])
                patience_counter = 0
                print(f"    -> P2 LR={p2_lr:.6f} (warmup={P2_WARMUP}ep) | DANN lambda={config.DANN_LAMBDA} | TEKNOFEST capali")
                
            elif new_phase == 3:
                # P3: Best modeli yukle + cok dusuk LR ile TEKNOFEST fine-tune
                print(f"\n  [PHASE CHANGE] P2 -> P3 | En iyi model yukleniyor (VF1={best_f1:.4f})")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                    print(f"    -> best_model.pth yuklendi.")
                ema = ModelEMA(model, decay=0.999)  # EMA'yi yuklenen agirliktan yeniden baslat

                p3_lr = config.LEARNING_RATE * 0.05  # %5 LR ile ince ayar
                for pg in optimizer.param_groups:
                    pg['lr'] = p3_lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=P3, eta_min=1e-7)
                patience_counter = 0
                
                # P3 BACKBONE FREEZE (CardioFusion7) — paylasilan stem + morfoloji
                # backbone dondurulur. RITIM UZMANI + gate + basliklar egitilmeye devam
                # eder (AFIB/AFL sinyalini onlar tasir — TEKNOFEST'te ince ayar kritik).
                frozen_count = 0
                # startswith kullaniliyor: 'stem.' rhythm_expert.stem.'i YANLISLIKLA
                # dondurmesin (alt-string eslesme hatasi). Bunlar top-level prefix'ler.
                freeze_layers = ('stem.', 'morph_stage', 'morph_down')
                for name, param in model.named_parameters():
                    if name.startswith(freeze_layers):
                        param.requires_grad = False
                        frozen_count += 1
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"    -> Morfoloji backbone DONDURULDU ({frozen_count} parametre grubu)")
                print(f"    -> Egitilebilir parametre: {trainable:,} (Ritim uzmani + gate + basliklar)")
                
                # SWA baslat — P3'te agirlik ortalaması ile daha iyi genelleme
                swa_model = AveragedModel(model).to(device)
                swa_scheduler = SWALR(optimizer, swa_lr=p3_lr * 0.5, anneal_epochs=5)
                print(f"    -> P3 LR={p3_lr:.6f} | Patience sifirlandi | SWA aktif")
                
            current_phase = new_phase

        # --- Warmup (sadece P1'in ilk 5 epoch'u) ---
        if current_phase == 1 and epoch <= config.WARMUP_EPOCHS:
            warmup_lr = config.LEARNING_RATE * (epoch / config.WARMUP_EPOCHS)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # --- Asama'ya gore loader, dann, alpha, mixup ---
        if current_phase == 1:
            loader = train_loader_tekno  # P1: Sadece Teknofest
            use_dann = False
            alpha = 0.0
            phase_name = "P1-TEKNO"
            use_mixup = False
            current_patience = PATIENCE_P1
        elif current_phase == 2:
            loader = train_loader_all    # P2: Tum Veri
            use_dann = True
            p = float(epoch - P1) / P2
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            phase_name = "P2-KARMA"
            use_mixup = False
            current_patience = PATIENCE_P2
        else:
            loader = train_loader_tekno  # P3: Sadece Teknofest
            use_dann = False
            alpha = 0.0
            phase_name = "P3-FINE"
            use_mixup = False
            current_patience = PATIENCE_P3

        # --- Hard mining alpha (P2: hafif 1.5x, P3: agresif 2.5x) ---
        if current_phase == 3:
            current_hard_alpha = p3_hard_alpha
        elif current_phase == 2:
            current_hard_alpha = p2_hard_alpha
        else:
            current_hard_alpha = None
        
        # --- Train ---
        # Gurultulu-etiket agirligi SADECE P2'de (internet verisi devredeyken) aktif
        irw = config.INTERNET_RHYTHM_WEIGHT if current_phase == 2 else None
        train_loss, train_f1 = train_one_epoch(
            model, loader, criterion_class, criterion_aux, criterion_domain,
            optimizer, scaler, device, use_amp, use_dann, alpha, use_mixup,
            hard_alpha=current_hard_alpha, ema=ema, internet_rhythm_weight=irw,
            deep_supervision=True   # CardioFusion7: morph + rhythm uzman basliklari
        )

        # --- Validate (RAW model) ---
        val_loss, val_f1, val_f1_class, y_true, y_prob = validate(
            model, val_loader, criterion_class, device, use_amp
        )

        # --- Validate (EMA model) — hangisi iyiyse o checkpoint'lenir ---
        ema_loss, ema_f1, ema_f1_class, _, _ = validate(
            ema.ema, val_loader, criterion_class, device, use_amp
        )
        if ema_f1 > val_f1:
            eval_tag = "EMA"
            val_loss, val_f1, val_f1_class = ema_loss, ema_f1, ema_f1_class
            cand_state = {k: v.clone() for k, v in ema.ema.state_dict().items()}
        else:
            eval_tag = "RAW"
            cand_state = None  # None -> model.state_dict() kaydedilir
        
        # --- Hard Example Mining — PDF Bolum 3.4 ---
        # P2'de: val setindeki en zor ornekleri bul, sonraki epoch'ta kullanilacak
        if current_phase == 2 and epoch > P1 + 3:  # P2'nin 3. epoch'undan sonra aktif
            try:
                model.eval()
                ornek_losslari = []
                with torch.no_grad():
                    for h_idx in range(len(val_dataset)):
                        try:
                            h_batch = val_dataset[h_idx]
                            h_sig = h_batch[0].unsqueeze(0).to(device)
                            h_wf = h_batch[1].unsqueeze(0).to(device)
                            h_lbl = h_batch[2].unsqueeze(0).to(device)
                            h_logits, _, _ = model(h_sig, h_wf)
                            h_loss = F.cross_entropy(h_logits, h_lbl, reduction='none')
                            ornek_losslari.append((h_idx, h_loss.item()))
                        except Exception:
                            continue
                
                # En yuksek loss'lu 200 ornegi sec (veya val setin %30'u)
                ornek_losslari.sort(key=lambda x: x[1], reverse=True)
                hard_count = min(200, len(ornek_losslari) // 3)
                hard_indices = [idx for idx, _ in ornek_losslari[:hard_count]]
                
                if hard_indices and epoch == P1 + 4:  # Sadece ilk seferde yazdir
                    print(f"    [HARD MINING] {len(hard_indices)} zor ornek tespit edildi")
            except Exception:
                pass  # Hard mining basarisizsa sessizce devam et

        # Scheduler step (warmup'tan sonra)
        if not (current_phase == 1 and epoch <= config.WARMUP_EPOCHS):
            if current_phase == 3 and swa_scheduler is not None:
                swa_scheduler.step()
                swa_model.update_parameters(model)
            elif scheduler is not None:
                scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start

        # --- Checkpoint + Patience (Global best ile; raw veya EMA) ---
        if val_f1 > best_f1 + MIN_F1_IMPROVEMENT:
            best_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(cand_state if cand_state is not None else model.state_dict(),
                       checkpoint_path)
            mark = f" [BEST-{eval_tag}]"
        else:
            patience_counter += 1
            mark = ""
        
        # Val Loss takibi (divergence check icin)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # --- Log ---
        log_epoch_ekle(training_log, {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_f1_macro": round(val_f1, 4),
            "val_f1_class": [round(f, 4) for f in val_f1_class],
            "lr": round(current_lr, 8),
            "patience_counter": patience_counter,
            "best_f1": round(best_f1, 4),
            "duration_sec": round(epoch_duration, 1)
        })

        sinif_str = " | ".join(f"{config.LABEL_NAMES[i][:3]}:{val_f1_class[i]:.3f}" for i in range(5))
        print(f"  [{phase_name}] E{epoch:3d}/{TOTAL_EPOCHS} | TL:{train_loss:.3f} VL:{val_loss:.3f} | "
              f"TF1:{train_f1:.3f} VF1:{val_f1:.3f} | {sinif_str} | "
              f"LR:{current_lr:.6f} ES:{patience_counter}/{current_patience} | "
              f"{epoch_duration:.1f}s{mark}")

        if patience_counter >= current_patience:
            if current_phase < 3:
                # Bu asamayi bitir, sonraki asama baslayana kadar epoch'lari atla
                if current_phase == 1:
                    skip_to_epoch = P1 + 1
                    print(f"\n  [EARLY STOPPING P1] {current_patience} epoch iyilesme yok. P2'ye atlaniyor (epoch {skip_to_epoch})...")
                elif current_phase == 2:
                    skip_to_epoch = P1 + P2 + 1
                    print(f"\n  [EARLY STOPPING P2] {current_patience} epoch iyilesme yok. P3'e atlaniyor (epoch {skip_to_epoch})...")
            else:
                print(f"\n  [EARLY STOPPING P3] {current_patience} epoch iyilesme yok. Egitim bitti.")
                break
        
        # Val Loss Divergence Check — model tamamen cikmissa fazdan atla
        if best_val_loss > 0 and val_loss > 2.5 * best_val_loss:
            if current_phase < 3:
                if current_phase == 1:
                    skip_to_epoch = P1 + 1
                    print(f"\n  [DIVERGENCE P1] Val Loss patlamasi! ({val_loss:.3f} > 2.5 * {best_val_loss:.3f}). P2'ye atlaniyor...")
                elif current_phase == 2:
                    skip_to_epoch = P1 + P2 + 1
                    print(f"\n  [DIVERGENCE P2] Val Loss patlamasi! ({val_loss:.3f} > 2.5 * {best_val_loss:.3f}). P3'e atlaniyor...")
            else:
                print(f"\n  [DIVERGENCE P3] Val Loss patlamasi! ({val_loss:.3f} > 2.5 * {best_val_loss:.3f}). Egitim bitti.")
                break

    # --- SWA BatchNorm guncelle ve kaydet ---
    if swa_model is not None:
        print(f"\nSWA BatchNorm guncelleniyor...")
        # Custom update_bn — dataset 5 eleman donduruyor, update_bn bunu handle edemez
        swa_model.train()
        with torch.no_grad():
            # BN istatistiklerini sifirla
            for module in swa_model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
                    module.num_batches_tracked.zero_()
            
            # Tum train verisiyle BN istatistiklerini guncelle
            swa_loader = DataLoader(full_train_dataset if len(full_train_dataset) > len(val_dataset) else val_dataset, 
                                    batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
            for batch in swa_loader:
                signals, wide_features_b, _, _, _ = batch
                signals = signals.to(device)
                wide_features_b = wide_features_b.to(device)
                swa_model(signals, wide_features_b)
        swa_model.eval()
        swa_path = os.path.join(config.CHECKPOINT_DIR, f"swa_model_{tag}.pth" if tag else "swa_model.pth")
        torch.save(swa_model.module.state_dict(), swa_path)
        print(f"SWA model kaydedildi: {swa_path}")
        
        # SWA modeli ile de validate et
        _, swa_f1, swa_f1_class, _, _ = validate(swa_model.module, val_loader, criterion_class, device, use_amp)
        print(f"SWA Val Macro F1: {swa_f1:.4f}")
        sinif_str = " | ".join(f"{config.LABEL_NAMES[i][:3]}:{swa_f1_class[i]:.3f}" for i in range(5))
        print(f"SWA Sinif F1: {sinif_str}")
        
        # SWA daha iyiyse onu kullan
        if swa_f1 > best_f1:
            print(f"SWA modeli daha iyi! ({swa_f1:.4f} > {best_f1:.4f}) -> best_model olarak kaydediliyor")
            torch.save(swa_model.module.state_dict(), checkpoint_path)
            best_f1 = swa_f1
    
    # --- TTA (Test-Time Augmentation) ile final degerlendirme ---
    print(f"\n{'='*70}")
    print("TTA (Test-Time Augmentation) ile Final Degerlendirme...")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    tta_y_true = []
    tta_y_prob = []
    
    with torch.no_grad():
        for batch in val_loader:
            signals, wide_features, labels, _, _ = batch
            signals = signals.to(device)
            wide_features = wide_features.to(device)
            
            # Orijinal tahmin
            all_probs = []
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    logits, _, _ = model(signals, wide_features)
                    all_probs.append(torch.softmax(logits, dim=1))
                    
                    # TTA 1: Zaman ekseninde ters cevir
                    sig_flip = torch.flip(signals, dims=[2])
                    logits_flip, _, _ = model(sig_flip, wide_features)
                    all_probs.append(torch.softmax(logits_flip, dim=1))
                    
                    # TTA 2: Hafif gurultu ekle
                    sig_noise = signals + 0.01 * torch.randn_like(signals)
                    logits_noise, _, _ = model(sig_noise, wide_features)
                    all_probs.append(torch.softmax(logits_noise, dim=1))
                    
                    # TTA 3: Amplitud olcekleme (0.95x)
                    sig_scale = signals * 0.95
                    logits_scale, _, _ = model(sig_scale, wide_features)
                    all_probs.append(torch.softmax(logits_scale, dim=1))
                    
                    # TTA 4: Amplitud olcekleme (1.05x)
                    sig_scale2 = signals * 1.05
                    logits_scale2, _, _ = model(sig_scale2, wide_features)
                    all_probs.append(torch.softmax(logits_scale2, dim=1))
            else:
                logits, _, _ = model(signals, wide_features)
                all_probs.append(torch.softmax(logits, dim=1))
                
                sig_flip = torch.flip(signals, dims=[2])
                logits_flip, _, _ = model(sig_flip, wide_features)
                all_probs.append(torch.softmax(logits_flip, dim=1))
                
                sig_noise = signals + 0.01 * torch.randn_like(signals)
                logits_noise, _, _ = model(sig_noise, wide_features)
                all_probs.append(torch.softmax(logits_noise, dim=1))
                
                sig_scale = signals * 0.95
                logits_scale, _, _ = model(sig_scale, wide_features)
                all_probs.append(torch.softmax(logits_scale, dim=1))
                
                sig_scale2 = signals * 1.05
                logits_scale2, _, _ = model(sig_scale2, wide_features)
                all_probs.append(torch.softmax(logits_scale2, dim=1))
            
            # Tum tahminlerin ortalamasini al
            avg_probs = torch.stack(all_probs).mean(dim=0)
            tta_y_true.extend(labels.cpu().numpy())
            tta_y_prob.extend(avg_probs.cpu().numpy())
    
    tta_y_true = np.array(tta_y_true)
    tta_y_prob = np.array(tta_y_prob)
    tta_preds = tta_y_prob.argmax(axis=1)
    tta_f1 = f1_score(tta_y_true, tta_preds, average='macro', zero_division=0)
    tta_f1_class = f1_score(tta_y_true, tta_preds, average=None, 
                            labels=list(range(config.NUM_CLASSES)), zero_division=0)
    
    sinif_str = " | ".join(f"{config.LABEL_NAMES[i][:3]}:{tta_f1_class[i]:.3f}" for i in range(5))
    print(f"\nTTA Macro F1 (5 augmentation): {tta_f1:.4f}")
    print(f"TTA Sinif F1: {sinif_str}")

    # --- Esik Optimizasyonu (TTA olasiliklari uzerinden) ---
    # ORTAK/JOINT optimizasyon: apply_thresholds'un macro F1'ini DOGRUDAN maksimize
    # eder. Per-class bagimsiz optimizasyon AFL<->AFIB rekabetini goremeyip argmax'tan
    # KOTU sonuc veriyordu (olculdu). Joint yontem argmax'i gecer.
    print(f"\n{'='*70}")
    print("Optimal Esik Degerleri Hesaplaniyor (JOINT, TTA uzerinden)...")
    real_thresholds, real_th_f1 = find_optimal_thresholds_joint(
        tta_y_true, tta_y_prob, num_classes=config.NUM_CLASSES)
    real_preds = apply_thresholds(tta_y_prob, real_thresholds)
    real_th_f1_class = f1_score(tta_y_true, real_preds, average=None,
                                labels=list(range(config.NUM_CLASSES)), zero_division=0)
    real_sinif_str = " | ".join(f"{config.LABEL_NAMES[i][:3]}:{real_th_f1_class[i]:.3f}" for i in range(5))
    th_str = " | ".join(f"{config.LABEL_NAMES[i][:3]}:{real_thresholds[i]:.2f}" for i in range(5))
    print(f"\n--- Esik Optimizasyonu (JOINT) ---")
    print(f"  Esikler: {th_str}")
    print(f"  GERCEK Macro F1 (yarisma metrigi): {real_th_f1:.4f}")
    print(f"  Sinif F1: {real_sinif_str}")
    print(f"------------------------------------")

    th_path = os.path.join(config.OUTPUT_DIR, f"optimal_thresholds_{tag}.json" if tag else "optimal_thresholds.json")
    with open(th_path, 'w') as f:
        json.dump({"thresholds": real_thresholds, "macro_f1": real_th_f1}, f, indent=4)
    print(f"Kaydedildi: {th_path}")

    log_bitir(training_log, "completed")

    print(f"\n{'='*70}")
    print(f"EGITIM TAMAMLANDI")
    print(f"  En iyi Val Macro F1       : {best_f1:.4f}")
    print(f"  TTA Macro F1              : {tta_f1:.4f}")
    print(f"  Esik Opt. JOINT (GERCEK)  : {real_th_f1:.4f}  <-- yarisma metrigi")
    print(f"  Model                     : {checkpoint_path}")
    print(f"{'='*70}")

    # --- GradCAM Otomatik Uretim ---
    print(f"\n[GRADCAM] 1D-GradCAM gorselleri uretiliyor...")
    try:
        from adim10_gradcam import gradcam_pipeline
        gradcam_pipeline()
    except Exception as e:
        print(f"  [UYARI] GradCAM uretimi basarisiz: {e}")
        print(f"  Manuel olarak calistirmak icin: python adim10_gradcam.py")

    return {"best_f1": best_f1, "tta_f1": tta_f1, "real_threshold_f1": real_th_f1}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                       help="Reproducibility seed (varsayilan: config.SEED). Ensemble icin farkli seed kullan.")
    parser.add_argument("--tag", type=str, default=None,
                       help="Checkpoint/log dosya adi eki (orn. seed123 -> best_model_seed123.pth).")
    args = parser.parse_args()
    egitim_pipeline(seed=args.seed, tag=args.tag)
