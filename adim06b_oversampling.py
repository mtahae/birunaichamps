"""
adim06b_oversampling.py — BirunAI EKG Siniflandirma: Adim 6b – Offline Oversampling
=====================================================================================

Ritim Bozuklugu (sinif 1) sadece 378 egitim ornegine sahip.
Normal (6280) ve Iletim (8562) ile karsilastirildiginda bu ciddi
bir sinif dengesizligidir.

Bu modul, Ritim Bozuklugu orneklerini 3 augmentasyon yontemiyle
fiziksel olarak cogaltir (offline oversampling):

    1. Time Shift   : Sinyali 1 saniye saga/sola kaydir (circular)
    2. Gaussian Noise: Hafif elektriksel cizirt ekle (std=0.05)
    3. Amplitude Scale: Voltaji %10 artir/azalt

Her orijinal Ritim orneginden bu 3 teknigin kombinasyonlariyla
yeni .npy dosyalari uretilir. Hedef: Ritim sinifini ~6000 ornege cikarmak.

Onemli: Sadece TRAIN setindeki Ritim orneklerine uygulanir!
Val/Test setleri degistirilmez (safligi korunur).

Ciktilar:
    - Yeni .npy dosyalari: outputs/processed_data/segmented_signals/
    - Guncellenmis train_manifest.csv
    - Yeni class_weights.npy

Kullanim:
    python adim06b_oversampling.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# AUGMENTASYON FONKSIYONLARI
# =============================================================================

def time_shift(sinyal, shift_samples):
    """
    Sinyali zaman ekseninde dairesel olarak kaydirir.

    Args:
        sinyal: numpy array (12, 2500)
        shift_samples: Kaydirma miktari (pozitif=saga, negatif=sola)

    Returns:
        numpy array (12, 2500) kaydirilmis sinyal
    """
    return np.roll(sinyal, shift_samples, axis=1)


def gaussian_noise(sinyal, std=0.05):
    """
    Sinyalin uzerine hafif Gaussian gurultu ekler.

    Gercek dunyada EKG sinyalleri elektriksel cizirtiya (EMG, 50Hz)
    maruz kalir. Bu augmentasyon, modeli bu tur gurutulere karsi
    dayanikli hale getirir.

    Args:
        sinyal: numpy array (12, 2500)
        std: Gurultu standart sapmasi (sinyal z-score normalize
             oldugu icin 0.05 makul bir deger)

    Returns:
        numpy array (12, 2500) gurultulu sinyal
    """
    noise = np.random.normal(0, std, sinyal.shape).astype(np.float32)
    return sinyal + noise


def amplitude_scale(sinyal, scale_factor):
    """
    Sinyalin genligini olcekler.

    Farkli hastalarda ayni ritim bozuklugu farkli voltajlarda
    gorunebilir. Bu augmentasyon, modelin voltaj degisimlerine
    karsi robust olmasini saglar.

    Args:
        sinyal: numpy array (12, 2500)
        scale_factor: Olcekleme faktoru (orn. 1.10 = %10 artis)

    Returns:
        numpy array (12, 2500) olceklenmis sinyal
    """
    return sinyal * scale_factor


# =============================================================================
# OVERSAMPLING PIPELINE
# =============================================================================

def oversampling_pipeline():
    """
    Ritim Bozuklugu orneklerini offline augmentasyon ile cogaltir.

    Strateji:
        Her orijinal ornekten birden fazla augmente kopya uretilir.
        Kopya sayisi, hedef sinif boyutuna ulasmak icin hesaplanir.

    Hedef: Ritim sinifi ~ min(Normal, Iletim) kadar ornege sahip olsun.
    """
    print("=" * 70)
    print("BirunAI -- Adim 6b: Ritim Bozuklugu Offline Oversampling")
    print("=" * 70)

    # --- 1. Mevcut dagilimi oku ---
    train_manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")

    if not os.path.exists(train_manifest_yolu):
        raise FileNotFoundError(
            f"train_manifest.csv bulunamadi: {train_manifest_yolu}\n"
            "Once adim06_veri_bolme.py calistirilmali."
        )

    train_df = pd.read_csv(train_manifest_yolu, index_col="ecg_id")

    print(f"\n[1/5] Mevcut Train Seti Dagilimi:")
    for sinif_idx in range(config.NUM_CLASSES):
        sinif_adi = config.LABEL_NAMES.get(sinif_idx, "?")
        sayi = (train_df["label"] == sinif_idx).sum()
        print(f"  [{sinif_idx}] {sinif_adi:20s}: {sayi:6d}")

    # --- 2. Hedef hesapla ---
    normal_sayisi = (train_df["label"] == 0).sum()
    ritim_sayisi = (train_df["label"] == 1).sum()
    iletim_sayisi = (train_df["label"] == 2).sum()

    # Hedef: Normal sinif sayisina ulasmak (daha buyuk olan Iletim yerine
    # Normal'i seciyoruz — asiri oversampling riskini azaltmak icin)
    hedef_sayisi = normal_sayisi
    uretilecek = hedef_sayisi - ritim_sayisi

    if uretilecek <= 0:
        print(f"\n  Ritim sinifi zaten yeterli ({ritim_sayisi} >= {hedef_sayisi}).")
        print(f"  Oversampling gerekmedi.")
        return

    print(f"\n[2/5] Hedef Hesaplama:")
    print(f"  Ritim mevcut    : {ritim_sayisi}")
    print(f"  Hedef           : {hedef_sayisi}")
    print(f"  Uretilecek      : {uretilecek} yeni ornek")

    # --- 3. Ritim orneklerini bul ---
    ritim_df = train_df[train_df["label"] == 1]
    ritim_ecg_ids = ritim_df.index.tolist()

    print(f"\n[3/5] {len(ritim_ecg_ids)} orijinal Ritim ornegi uzerinden augmentasyon basliyor...")

    # --- 4. Augmentasyon ---
    # Her orijinal ornekten kac kopya uretilecegi
    kopya_per_ornek = uretilecek // len(ritim_ecg_ids) + 1

    # Augmentasyon parametreleri
    augmentasyon_listesi = [
        # (isim, fonksiyon, parametreler)
        ("ts_pos250", lambda s: time_shift(s, 250)),     # 1 saniye saga (250 ornek @ 250 Hz)
        ("ts_neg250", lambda s: time_shift(s, -250)),    # 1 saniye sola
        ("ts_pos125", lambda s: time_shift(s, 125)),     # 0.5 saniye saga
        ("ts_neg125", lambda s: time_shift(s, -125)),    # 0.5 saniye sola
        ("noise_1",   lambda s: gaussian_noise(s, 0.03)),  # Hafif gurultu
        ("noise_2",   lambda s: gaussian_noise(s, 0.05)),  # Orta gurultu
        ("noise_3",   lambda s: gaussian_noise(s, 0.07)),  # Guclu gurultu
        ("amp_110",   lambda s: amplitude_scale(s, 1.10)),  # %10 artis
        ("amp_090",   lambda s: amplitude_scale(s, 0.90)),  # %10 azalis
        ("amp_115",   lambda s: amplitude_scale(s, 1.15)),  # %15 artis
        ("amp_085",   lambda s: amplitude_scale(s, 0.85)),  # %15 azalis
        # Kombinasyonlar
        ("ts_noise1",    lambda s: gaussian_noise(time_shift(s, 250), 0.04)),
        ("ts_noise2",    lambda s: gaussian_noise(time_shift(s, -250), 0.04)),
        ("ts_amp1",      lambda s: amplitude_scale(time_shift(s, 125), 1.10)),
        ("ts_amp2",      lambda s: amplitude_scale(time_shift(s, -125), 0.90)),
        ("noise_amp1",   lambda s: amplitude_scale(gaussian_noise(s, 0.04), 1.10)),
        ("noise_amp2",   lambda s: amplitude_scale(gaussian_noise(s, 0.04), 0.90)),
    ]

    yeni_satirlar = []
    uretilen_sayisi = 0

    for ecg_id in ritim_ecg_ids:
        sinyal_yolu = os.path.join(sinyal_dizini, f"{ecg_id}.npy")

        if not os.path.exists(sinyal_yolu):
            print(f"  [UYARI] {sinyal_yolu} bulunamadi, atlaniyor.")
            continue

        sinyal = np.load(sinyal_yolu)  # (12, 2500)

        # Orijinal kaydin diger bilgilerini al
        kayit_bilgisi = ritim_df.loc[ecg_id]

        for aug_idx, (aug_isim, aug_fn) in enumerate(augmentasyon_listesi):
            if uretilen_sayisi >= uretilecek:
                break

            # Augmente sinyal uret
            aug_sinyal = aug_fn(sinyal).astype(np.float32)

            # Yeni ID: orijinal_id * 100000 + augmentasyon_sira_numarasi
            yeni_ecg_id = ecg_id * 100000 + aug_idx + 1

            # Kaydet
            yeni_dosya_yolu = os.path.join(sinyal_dizini, f"{yeni_ecg_id}.npy")
            np.save(yeni_dosya_yolu, aug_sinyal)

            # Manifest icin satir
            yeni_satir = kayit_bilgisi.copy()
            yeni_satir.name = yeni_ecg_id
            yeni_satirlar.append(yeni_satir)

            uretilen_sayisi += 1

        if uretilen_sayisi >= uretilecek:
            break

    print(f"  Toplam {uretilen_sayisi} yeni augmente ornek uretildi.")

    # --- 5. Manifest guncelle ---
    print(f"\n[4/5] train_manifest.csv guncelleniyor...")

    yeni_df = pd.DataFrame(yeni_satirlar)
    yeni_df.index.name = "ecg_id"

    # Orijinal + yeni
    train_df_yeni = pd.concat([train_df, yeni_df])

    print(f"  Eski train boyutu : {len(train_df)}")
    print(f"  Yeni train boyutu : {len(train_df_yeni)}")

    # Yeni dagilim
    print(f"\n  Yeni Sinif Dagilimi:")
    for sinif_idx in range(config.NUM_CLASSES):
        sinif_adi = config.LABEL_NAMES.get(sinif_idx, "?")
        sayi = (train_df_yeni["label"] == sinif_idx).sum()
        print(f"    [{sinif_idx}] {sinif_adi:20s}: {sayi:6d}")

    # Manifest kaydet
    train_df_yeni.to_csv(train_manifest_yolu)
    print(f"\n  train_manifest.csv guncellendi.")

    # --- 6. Class weights yeniden hesapla ---
    print(f"\n[5/5] Sinif agirliklari yeniden hesaplaniyor...")

    labels = train_df_yeni["label"].values.astype(int)
    siniflar = np.unique(labels)
    toplam = len(labels)
    n_sinif = len(siniflar)

    agirliklar = np.zeros(config.NUM_CLASSES)
    for sinif in siniflar:
        sinif_sayisi = np.sum(labels == sinif)
        agirliklar[int(sinif)] = toplam / (n_sinif * sinif_sayisi)

    agirlik_yolu = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
    np.save(agirlik_yolu, agirliklar)

    print(f"  Yeni sinif agirliklari:")
    for sinif_idx in range(config.NUM_CLASSES):
        sinif_adi = config.LABEL_NAMES.get(sinif_idx, "?")
        print(f"    [{sinif_idx}] {sinif_adi:20s}: {agirliklar[sinif_idx]:.4f}")

    print(f"  Kaydedildi: {agirlik_yolu}")

    # --- Ozet ---
    print(f"\n{'='*70}")
    print(f"OVERSAMPLING TAMAMLANDI")
    print(f"{'='*70}")
    print(f"  Uretilen ornek sayisi : {uretilen_sayisi}")
    print(f"  Yeni train boyutu    : {len(train_df_yeni)}")
    print(f"  Uyg. teknikler       : Time Shift, Gaussian Noise, Amplitude Scale")
    print(f"{'='*70}")

    return train_df_yeni


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    oversampling_pipeline()
