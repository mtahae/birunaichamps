"""
adim08_egitim.py — BirunAI EKG Siniflandirma: Adim 8 – Egitim (v4)
====================================================================

v4 Iyilestirmeleri:
    - Offline Oversampling ile dengelenmis veri seti (Ritim: 378 -> 6280)
    - Cosine Annealing LR (Warmup sonrasi) — platoda takilmayi onler
    - Online augmentasyon KAPALI (offline oversampling yeterli)
    - Label Smoothing dusuruluyor (0.05) — veri artik dengeli
    - SWA: Son 15 epoch'ta agirlik ortalamalama
    - WeightedRandomSampler KALDIRILIYOR — veri artik dengeli, sampler gereksiz
    - Focal Loss gamma dusuruluyor (1.0) — kolay/zor ayrim artik gereksiz
    - CosineAnnealingWarmRestarts — LR'yi periyodik olarak dusurur/arttirir

Hedef: Val F1 > 0.85, Val Loss < 0.3

Ciktilar:
    - outputs/checkpoints/best_model.pth
    - outputs/checkpoints/swa_model.pth
    - outputs/training_log.json
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from sklearn.metrics import f1_score
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import BirunAIModel, EKGDataset, FocalLoss, model_ozetini_yazdir


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
        "device": str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
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
# TRAIN VE VALIDATION FONKSIYONLARI
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    toplam_loss = 0
    tum_tahminler = []
    tum_etiketler = []

    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if use_amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(signals)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_MAX_NORM)
            optimizer.step()

        toplam_loss += loss.item() * signals.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        tum_tahminler.extend(preds)
        tum_etiketler.extend(labels.cpu().numpy())

    avg_loss = toplam_loss / len(loader.dataset)
    macro_f1 = f1_score(tum_etiketler, tum_tahminler, average='macro', zero_division=0)
    return avg_loss, macro_f1


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    toplam_loss = 0
    tum_tahminler = []
    tum_etiketler = []

    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(signals)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(signals)
                loss = criterion(outputs, labels)

            toplam_loss += loss.item() * signals.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            tum_tahminler.extend(preds)
            tum_etiketler.extend(labels.cpu().numpy())

    avg_loss = toplam_loss / len(loader.dataset)
    macro_f1 = f1_score(tum_etiketler, tum_tahminler, average='macro', zero_division=0)
    per_class_f1 = f1_score(tum_etiketler, tum_tahminler, average=None,
                            labels=[0, 1, 2], zero_division=0)
    return avg_loss, macro_f1, per_class_f1.tolist()


# =============================================================================
# ANA EGITIM PIPELINE'I
# =============================================================================

def egitim_pipeline():
    print("=" * 70)
    print("BirunAI -- Adim 8: Model Egitimi (v4 — Dengeli Veri + Cosine LR)")
    print("=" * 70)

    # --- Reproducibility ---
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # --- Cihaz ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Cihaz: {device}")
    if device.type == 'cuda':
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  [UYARI] GPU bulunamadi.")

    use_amp = config.USE_AMP and device.type == 'cuda'

    # --- Dataset ve DataLoader ---
    print(f"\n  Veri setleri yukleniyor...")
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")

    # Augmentasyon: Sadece online augmentasyon (hafif), offline oversampling zaten yapildi
    train_dataset = EKGDataset(
        os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv"),
        sinyal_dizini,
        augment=True   # Hafif online augmentasyon hala acik (cesiitlilik icin)
    )
    val_dataset = EKGDataset(
        os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv"),
        sinyal_dizini,
        augment=False
    )

    # Sinif dagilimini goster
    from collections import Counter
    sinif_dag = Counter(train_dataset.labels)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    for idx in range(config.NUM_CLASSES):
        print(f"    [{idx}] {config.LABEL_NAMES[idx]:20s}: {sinif_dag.get(idx, 0)}")

    # Sinif agirliklari
    class_weights = np.load(os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy"))

    # DataLoader — artik WeightedRandomSampler gereksiz (veri dengeli)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'),
        drop_last=True  # Son eksik batch'i at (BN stabilitesi icin)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # --- Model ---
    print(f"\n  Model olusturuluyor...")
    model = BirunAIModel().to(device)
    model_ozetini_yazdir(model)

    # SWA modeli
    swa_model = AveragedModel(model)
    swa_start_epoch = max(config.NUM_EPOCHS - 15, config.NUM_EPOCHS // 2)
    swa_aktif = False

    # --- Loss, Optimizer, Scheduler ---
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=config.FOCAL_LOSS_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # CosineAnnealingWarmRestarts:
    # - Warmup icin ilk 5 epoch'ta LR artarsa, sonra cosine ile duser
    # - T_0=15: Her 15 epoch'ta bir restart
    # - T_mult=2: Her restart sonrasi periyot 2x uzar (15, 30, 60...)
    # - Toplam: Warmup(5) + Cosine(75) = 80 epoch
    warmup_epochs = config.WARMUP_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"\n  Egitim Parametreleri (v4 — Dengeli Veri):")
    print(f"    Optimizer       : AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    print(f"    Scheduler       : Warmup({warmup_epochs}ep) + CosineAnnealingWarmRestarts(T0=15)")
    print(f"    Loss            : FocalLoss(gamma={config.FOCAL_LOSS_GAMMA}, smooth={config.LABEL_SMOOTHING})")
    print(f"    Online Augment  : ACIK (hafif)")
    print(f"    Offline Oversamp: Ritim 378 -> 6280 (Time Shift, Noise, Amplitude)")
    print(f"    SWA             : epoch {swa_start_epoch}+ (son 15 epoch)")
    print(f"    Batch Size      : {config.BATCH_SIZE}")
    print(f"    Epochs          : {config.NUM_EPOCHS}")
    print(f"    Early Stop      : {config.EARLY_STOPPING_PATIENCE} epoch (Val Macro F1)")
    print(f"    Mixed Precision : {'Evet' if use_amp else 'Hayir'}")

    # --- Egitim Dongusu ---
    print(f"\n{'='*70}")
    print(f"  EGITIM BASLIYOR...")
    print(f"{'='*70}\n")

    training_log = log_baslat(config.NUM_EPOCHS)
    best_f1 = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    swa_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "swa_model.pth")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Warmup: Ilk 5 epoch LR lineer artis
        if epoch <= warmup_epochs:
            warmup_lr = config.LEARNING_RATE * (epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # Validate
        val_loss, val_f1, val_f1_class = validate(
            model, val_loader, criterion, device, use_amp
        )

        # Scheduler step (warmup sonrasi)
        if epoch > warmup_epochs:
            scheduler.step(epoch - warmup_epochs)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_duration = time.time() - epoch_start

        # SWA
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_aktif = True

        # Early Stopping (Val Macro F1 bazli)
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            kayit_durumu = " [BEST]"
        else:
            patience_counter += 1
            kayit_durumu = ""

        # Log
        epoch_data = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_f1_macro": round(val_f1, 4),
            "val_f1_class": [round(f, 4) for f in val_f1_class],
            "lr": round(current_lr, 6),
            "patience_counter": patience_counter,
            "best_f1": round(best_f1, 4),
            "duration_sec": round(epoch_duration, 1)
        }
        log_epoch_ekle(training_log, epoch_data)

        # Konsol ciktisi
        sinif_f1_str = " | ".join(
            f"{config.LABEL_NAMES[i][:3]}:{val_f1_class[i]:.3f}"
            for i in range(config.NUM_CLASSES)
        )
        warmup_tag = " [W]" if epoch <= warmup_epochs else ""
        swa_tag = " [SWA]" if epoch >= swa_start_epoch else ""
        print(
            f"  E{epoch:3d}/{config.NUM_EPOCHS} | "
            f"TL:{train_loss:.3f} VL:{val_loss:.3f} | "
            f"TF1:{train_f1:.3f} VF1:{val_f1:.3f} | "
            f"{sinif_f1_str} | "
            f"LR:{current_lr:.6f} ES:{patience_counter}/{config.EARLY_STOPPING_PATIENCE} | "
            f"{epoch_duration:.1f}s{kayit_durumu}{warmup_tag}{swa_tag}"
        )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n  [EARLY STOPPING] {config.EARLY_STOPPING_PATIENCE} epoch iyilesme yok.")
            break

    # --- SWA: BatchNorm guncelle ---
    if swa_aktif:
        print(f"\n  SWA BatchNorm guncelleniyor...")
        bn_loader = DataLoader(
            EKGDataset(
                os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv"),
                sinyal_dizini, augment=False
            ),
            batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        update_bn(bn_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), swa_checkpoint_path)
        print(f"  SWA modeli kaydedildi: {swa_checkpoint_path}")

        # SWA degerlendir
        swa_model.to(device)
        _, swa_f1, swa_f1_class = validate(swa_model, val_loader, criterion, device, use_amp)
        print(f"  SWA Val F1: {swa_f1:.4f} vs Best Val F1: {best_f1:.4f}")
        swa_class_str = " | ".join(
            f"{config.LABEL_NAMES[i][:3]}:{swa_f1_class[i]:.3f}"
            for i in range(config.NUM_CLASSES)
        )
        print(f"  SWA sinif F1: {swa_class_str}")

        if swa_f1 > best_f1:
            torch.save(swa_model.module.state_dict(), checkpoint_path)
            print(f"  SWA modeli daha iyi! best_model.pth guncellendi.")
            best_f1 = swa_f1

    # --- Sonuc ---
    log_bitir(training_log, "completed")

    print(f"\n{'='*70}")
    print(f"EGITIM TAMAMLANDI")
    print(f"{'='*70}")
    print(f"  En iyi Val Macro F1 : {best_f1:.4f}")
    print(f"  Model               : {checkpoint_path}")
    print(f"  Toplam epoch        : {epoch}")
    print(f"{'='*70}")

    return {"best_f1": best_f1, "total_epochs": epoch}


if __name__ == "__main__":
    sonuc = egitim_pipeline()
