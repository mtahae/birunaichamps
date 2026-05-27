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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import f1_score
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import CardioFusion5, EKGDataset, FocalLoss, model_ozetini_yazdir
from threshold_opt import find_optimal_thresholds
from torch.optim.swa_utils import AveragedModel, SWALR

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

def train_one_epoch(model, loader, criterion_class, criterion_aux, criterion_domain, 
                    optimizer, scaler, device, use_amp, use_dann=False, alpha=1.0, use_mixup=False):
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

        # Mixup augmentation (P2'de aktif)
        if use_mixup and random.random() < 0.5:
            signals, labels_a, labels_b, lam = mixup_data(signals, labels, alpha=0.2)
            do_mixup = True
        else:
            do_mixup = False

        optimizer.zero_grad()

        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                class_logits, aux_logits, domain_logits = model(signals, wide_features, alpha)
                
                # Main Loss
                if do_mixup:
                    loss_class = mixup_criterion(criterion_class, class_logits, labels_a, labels_b, lam)
                else:
                    loss_class = criterion_class(class_logits, labels)
                
                # Aux Loss
                loss_aux = criterion_aux(aux_logits, aux_labels)
                
                # Domain Loss (DANN)
                if use_dann:
                    domain_binary = (domains > 0).long()
                    loss_domain = criterion_domain(domain_logits, domain_binary)
                    loss = loss_class + config.AUX_LOSS_WEIGHT * loss_aux + config.DANN_LAMBDA * loss_domain
                else:
                    loss = loss_class + config.AUX_LOSS_WEIGHT * loss_aux
                    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            class_logits, aux_logits, domain_logits = model(signals, wide_features, alpha)
            if do_mixup:
                loss_class = mixup_criterion(criterion_class, class_logits, labels_a, labels_b, lam)
            else:
                loss_class = criterion_class(class_logits, labels)
            loss_aux = criterion_aux(aux_logits, aux_labels)
            if use_dann:
                domain_binary = (domains > 0).long()
                loss_domain = criterion_domain(domain_logits, domain_binary)
                loss = loss_class + config.AUX_LOSS_WEIGHT * loss_aux + config.DANN_LAMBDA * loss_domain
            else:
                loss = loss_class + config.AUX_LOSS_WEIGHT * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            optimizer.step()

        toplam_class_loss += loss_class.item() * signals.size(0)
        if do_mixup:
            preds = class_logits.argmax(dim=1).cpu().numpy()
            tum_tahminler.extend(preds)
            tum_etiketler.extend(labels_a.cpu().numpy())  # Dominant label
        else:
            preds = class_logits.argmax(dim=1).cpu().numpy()
            tum_tahminler.extend(preds)
            tum_etiketler.extend(labels.cpu().numpy())

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

def egitim_pipeline():
    print("=" * 70)
    print("BirunAI -- CardioFusion-5 Curriculum Learning (Final)")
    print("=" * 70)

    # --- Reproducibility (PDF Bolum 9) ---
    config.set_seed(config.SEED)

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
    balanced_sampler_all = create_balanced_sampler(full_train_dataset)
    balanced_sampler_tekno = create_balanced_sampler(tekno_train_dataset)
    
    # DataLoaders
    train_loader_all = DataLoader(full_train_dataset, batch_size=config.BATCH_SIZE, 
                                  sampler=balanced_sampler_all, num_workers=2, pin_memory=True, drop_last=True)
    train_loader_tekno = DataLoader(tekno_train_dataset, batch_size=config.BATCH_SIZE, 
                                    sampler=balanced_sampler_tekno, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=False, num_workers=2, pin_memory=True)

    # --- Model ---
    model = CardioFusion5(
        num_classes=config.NUM_CLASSES,
        num_aux_classes=config.NUM_AUX_CLASSES,
        num_domains=2,
        wide_feature_dim=config.WIDE_FEATURE_DIM
    ).to(device)
    model_ozetini_yazdir(model)

    # --- Loss ---
    class_weights_path = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
    if os.path.exists(class_weights_path):
        class_weights = np.load(class_weights_path)
    else:
        class_weights = np.ones(config.NUM_CLASSES, dtype=np.float32)
        
    criterion_class = FocalLoss(alpha=class_weights, gamma=config.FOCAL_LOSS_GAMMA, 
                                label_smoothing=config.LABEL_SMOOTHING).to(device)
    criterion_aux = nn.CrossEntropyLoss().to(device)
    criterion_domain = nn.CrossEntropyLoss().to(device)
    
    # --- Optimizer (scheduler her asama icin ayri ayarlanacak) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler() if use_amp else None

    # --- Curriculum Asamalari (config'den) ---
    P1 = config.EPOCHS_PHASE_1  # 20
    P2 = config.EPOCHS_PHASE_2  # 60
    P3 = config.EPOCHS_PHASE_3  # 20
    TOTAL_EPOCHS = P1 + P2 + P3  # 100
    
    # Her asama icin ayri patience
    PATIENCE_P1 = 15
    PATIENCE_P2 = 15  # 25'ten dusuruldu — overfitting onleme
    PATIENCE_P3 = 10
    MIN_F1_IMPROVEMENT = 0.002  # F1 artisi bundan azsa "iyilesme" sayilmaz

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    training_log = log_baslat(TOTAL_EPOCHS)
    best_f1 = 0.0
    best_val_loss = float('inf')  # Val Loss divergence takibi
    patience_counter = 0

    print(f"\n  Egitim Parametreleri:")
    print(f"    Optimizer      : AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    print(f"    Loss           : FocalLoss(g={config.FOCAL_LOSS_GAMMA}) + AuxCE(w={config.AUX_LOSS_WEIGHT}) + Mixup(P2)")
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
                
                # P2 LR: Base LR'in 0.3x'i (nazik gecis — PhysioNet HeartBeats stratejisi)
                p2_lr = config.LEARNING_RATE * 0.3
                for pg in optimizer.param_groups:
                    pg['lr'] = p2_lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=P2, eta_min=1e-6)
                patience_counter = 0
                print(f"    -> P2 LR={p2_lr:.6f} | Patience sifirlandi | Cosine T_max={P2} | Mixup+DANN aktif")
                
            elif new_phase == 3:
                # P3: Best modeli yukle + cok dusuk LR
                print(f"\n  [PHASE CHANGE] P2 -> P3 | En iyi model yukleniyor (VF1={best_f1:.4f})")
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                    print(f"    -> best_model.pth yuklendi.")
                
                p3_lr = config.LEARNING_RATE * 0.05  # %5 LR ile ince ayar
                for pg in optimizer.param_groups:
                    pg['lr'] = p3_lr
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=P3, eta_min=1e-7)
                patience_counter = 0
                
                # P3 BACKBONE FREEZE — Sadece Transformer + Classifier egitilir
                # CNN backbone'u dondur (genel EKG paternlerini korumak icin)
                frozen_count = 0
                for name, param in model.named_parameters():
                    if any(layer in name for layer in ['conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.']):
                        param.requires_grad = False
                        frozen_count += 1
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"    -> CNN Backbone DONDURULDU ({frozen_count} parametre grubu)")
                print(f"    -> Egitilebilir parametre: {trainable:,} (Transformer + Classifier)")
                
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
            loader = train_loader_tekno
            use_dann = False
            alpha = 0.0
            phase_name = "P1-TEKNO"
            use_mixup = False
            current_patience = PATIENCE_P1
        elif current_phase == 2:
            loader = train_loader_all
            use_dann = True
            p = float(epoch - P1) / P2
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            phase_name = "P2-KARMA"
            use_mixup = True  # Mixup sadece P2'de aktif
            current_patience = PATIENCE_P2
        else:
            loader = train_loader_all  # P3: Tum veri + donmus backbone = genelleme
            use_dann = False
            alpha = 0.0
            phase_name = "P3-FINE"
            use_mixup = False
            current_patience = PATIENCE_P3

        # --- Train ---
        train_loss, train_f1 = train_one_epoch(
            model, loader, criterion_class, criterion_aux, criterion_domain,
            optimizer, scaler, device, use_amp, use_dann, alpha, use_mixup
        )

        # --- Validate ---
        val_loss, val_f1, val_f1_class, y_true, y_prob = validate(
            model, val_loader, criterion_class, device, use_amp
        )

        # Scheduler step (warmup'tan sonra)
        if not (current_phase == 1 and epoch <= config.WARMUP_EPOCHS):
            if current_phase == 3 and swa_scheduler is not None:
                swa_scheduler.step()
                swa_model.update_parameters(model)
            elif scheduler is not None:
                scheduler.step()
            
        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start

        # --- Checkpoint (Min improvement esigi ile) ---
        if val_f1 > best_f1 + MIN_F1_IMPROVEMENT:
            best_f1 = val_f1
            best_val_loss = val_loss  # Val loss referansini guncelle
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            mark = " [BEST]"
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
        swa_path = os.path.join(config.CHECKPOINT_DIR, "swa_model.pth")
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
    print(f"\n{'='*70}")
    print("Optimal Esik Degerleri Hesaplaniyor (TTA uzerinden)...")
    optimal_thresholds, best_f1s = find_optimal_thresholds(tta_y_true, tta_y_prob, num_classes=config.NUM_CLASSES)
    
    th_macro_f1 = np.mean(best_f1s)
    print(f"\n--- Esik Optimizasyonu Sonuclari ---")
    for i in range(config.NUM_CLASSES):
        print(f"{config.LABEL_NAMES[i]:6s} -> Optimal Esik: {optimal_thresholds[i]:.2f} (F1: {best_f1s[i]:.4f})")
    print(f"Esik Optimizasyonu Sonrasi Macro F1: {th_macro_f1:.4f}")
    print(f"------------------------------------")
    
    th_path = os.path.join(config.OUTPUT_DIR, "optimal_thresholds.json")
    with open(th_path, 'w') as f:
        json.dump({"thresholds": optimal_thresholds, "f1s": best_f1s}, f, indent=4)
    print(f"Kaydedildi: {th_path}")

    log_bitir(training_log, "completed")

    print(f"\n{'='*70}")
    print(f"EGITIM TAMAMLANDI")
    print(f"  En iyi Val Macro F1       : {best_f1:.4f}")
    print(f"  TTA Macro F1              : {tta_f1:.4f}")
    print(f"  Esik Opt. Macro F1 (TTA)  : {th_macro_f1:.4f}")
    print(f"  Model                     : {checkpoint_path}")
    print(f"{'='*70}")

    return {"best_f1": best_f1, "tta_f1": tta_f1, "threshold_f1": th_macro_f1}


if __name__ == "__main__":
    egitim_pipeline()
