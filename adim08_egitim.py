"""
adim08_egitim.py — BirunAI EKG Siniflandirma: Adim 8 – Egitim (v3)
====================================================================

v3 Iyilestirmeleri (Overfitting + Val Loss Dalgalanmasi):
    - Augmentasyon: train seti icin time-shift, noise, scaling, lead dropout
    - SWA (Stochastic Weight Averaging): Val loss dalgalanmasini yumusatir
    - Warmup(5ep) + ReduceLROnPlateau: LR'yi adaptif dusurmek icin
    - Label Smoothing (0.1): Overconfidence onleme
    - Weight Decay (1e-4): Daha guclu regularization
    - Patience 15: Early stop'a yeterince zaman tan

Egitim stratejisi:
    - Focal Loss (gamma=2.0) + sinif agirliklari + label smoothing
    - WeightedRandomSampler: Her batch'te siniflar dengelenir
    - AdamW optimizer + Warmup + ReduceLROnPlateau
    - Gradient Clipping (max_norm=1.0)
    - Mixed Precision (AMP)
    - SWA: Son 20 epoch'ta agirlik ortalamalama
    - Early Stopping: Val Macro F1 bazli

Ciktilar:
    - outputs/checkpoints/best_model.pth
    - outputs/checkpoints/swa_model.pth   (SWA modeli)
    - outputs/training_log.json (dashboard icin)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
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
# WARMUP + REDUCE ON PLATEAU SCHEDULER
# =============================================================================

class WarmupScheduler:
    """
    Ilk warmup_epochs epoch: LR lineer 0 -> base_lr
    Sonrasi: ReduceLROnPlateau (val F1 plato yaparsa 0.5x dusur)
    """
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )

    def step(self, epoch, val_metric=None):
        if epoch <= self.warmup_epochs:
            warmup_lr = self.base_lr * (epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            if val_metric is not None:
                self.plateau_scheduler.step(val_metric)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


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
    print("BirunAI -- Adim 8: Model Egitimi (v3 — Augmentasyon + SWA)")
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

    # augment=True sadece train icin
    train_dataset = EKGDataset(
        os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv"),
        sinyal_dizini,
        augment=True   # <-- Augmentasyon ACIK
    )
    val_dataset = EKGDataset(
        os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv"),
        sinyal_dizini,
        augment=False  # <-- Augmentasyon KAPALI (saf degerlendirme)
    )

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"  Train Augmentasyon: ACIK (time-shift, noise, amplitude, lead-dropout)")

    # WeightedRandomSampler
    class_weights = np.load(os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy"))
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # --- Model ---
    print(f"\n  Model olusturuluyor...")
    model = BirunAIModel().to(device)
    model_ozetini_yazdir(model)

    # SWA modeli: Son SWA_START_EPOCH'tan itibaren agirlik ortalamalama
    swa_model = AveragedModel(model)
    swa_start_epoch = max(config.NUM_EPOCHS - 20, config.NUM_EPOCHS // 2)
    swa_aktif = False
    print(f"  SWA baslangic epoch'u: {swa_start_epoch}")

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

    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        base_lr=config.LEARNING_RATE
    )

    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"\n  Egitim Parametreleri (v3):")
    print(f"    Optimizer       : AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
    print(f"    Scheduler       : Warmup({config.WARMUP_EPOCHS}ep) + ReduceLROnPlateau")
    print(f"    Loss            : FocalLoss(gamma={config.FOCAL_LOSS_GAMMA}, smooth={config.LABEL_SMOOTHING})")
    print(f"    Augmentasyon    : time-shift, noise, amplitude, lead-dropout (p=0.8)")
    print(f"    SWA             : epoch {swa_start_epoch}+ (son 20 epoch ortalamalama)")
    print(f"    Batch Size      : {config.BATCH_SIZE}")
    print(f"    Epochs          : {config.NUM_EPOCHS}")
    print(f"    Early Stop      : {config.EARLY_STOPPING_PATIENCE} epoch")
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

        # Warmup adimi (epoch basinda)
        scheduler.step(epoch, val_metric=None)

        # Train
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # Validate
        val_loss, val_f1, val_f1_class = validate(
            model, val_loader, criterion, device, use_amp
        )

        # Plateau scheduler (warmup sonrasi)
        if epoch > config.WARMUP_EPOCHS:
            scheduler.step(epoch, val_metric=val_f1)

        current_lr = scheduler.get_lr()
        epoch_duration = time.time() - epoch_start

        # SWA — son swa_start_epoch'tan itibaren agirlik topla
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_aktif = True

        # Early Stopping
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
        warmup_tag = " [W]" if epoch <= config.WARMUP_EPOCHS else ""
        swa_tag = " [SWA]" if swa_aktif else ""
        print(
            f"  E{epoch:3d}/{config.NUM_EPOCHS} | "
            f"TL:{train_loss:.3f} VL:{val_loss:.3f} | "
            f"TF1:{train_f1:.3f} VF1:{val_f1:.3f} | "
            f"{sinif_f1_str} | "
            f"LR:{current_lr:.5f} ES:{patience_counter}/{config.EARLY_STOPPING_PATIENCE} | "
            f"{epoch_duration:.1f}s{kayit_durumu}{warmup_tag}{swa_tag}"
        )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n  [EARLY STOPPING] {config.EARLY_STOPPING_PATIENCE} epoch iyilesme yok.")
            break

    # --- SWA: BatchNorm istatistiklerini guncelle ---
    if swa_aktif:
        print(f"\n  SWA BatchNorm guncelleniyor...")
        train_loader_nobug = DataLoader(
            EKGDataset(
                os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv"),
                sinyal_dizini, augment=False
            ),
            batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        update_bn(train_loader_nobug, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), swa_checkpoint_path)
        print(f"  SWA modeli kaydedildi: {swa_checkpoint_path}")

        # SWA ile de val degerlendirmesi yap
        swa_model.to(device)
        _, swa_f1, swa_f1_class = validate(swa_model, val_loader, criterion, device, use_amp)
        print(f"  SWA Val F1: {swa_f1:.4f} vs Best Val F1: {best_f1:.4f}")

        # SWA daha iyi ise best_model olarak kaydet
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


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = egitim_pipeline()
