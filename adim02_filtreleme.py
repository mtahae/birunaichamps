"""
adim02_filtreleme.py — BirunAI EKG Siniflandirma: Adim 2 – Filtreleme ve Alt Ornekleme
========================================================================================

Bu modul, ham EKG sinyallerine sinyal isleme adimlari uygular.

Projemizde belirttigimiz islem adimlari:
    1. Ham sinyallerin wfdb ile okunmasi (.dat + .hea dosyalari)
    2. Alt ornekleme (Resampling): 500 Hz -> 250 Hz
       - Nyquist teoremi geregi 250 Hz, 125 Hz'e kadar bilesenleri korur.
       - EKG'nin klinik olarak anlamli frekans bandi 0.5-40 Hz'dir.
       - VRAM tuketimini yariya dusurur (5000 -> 2500 zaman adimi/kayit).
    3. Butterworth Bandpass Filtre: 0.5-40 Hz, order=4
       - 0.5 Hz high-pass: Taban cizgisi kaymasini (baseline wander) eler.
         (Solunum, hasta hareketi kaynakli dusuk frekanslı salinimlar)
       - 40 Hz low-pass: Yuksek frekanslı EMG artefaktlarini eler.
         (Kas seyirmesi, elektronik cihaz gurultusu)
       - 40 Hz ust kesim noktasi, 50 Hz sebeke gurultusunu de dogal olarak engeller.
       - scipy.signal.butter + filtfilt (sifir faz kaymasi) -> QRS zamanlamasi korunur.
    4. Z-score Normalizasyon: Her derivasyonun ortalamasi 0, standart sapmasi 1.

Ciktilar:
    - outputs/processed_data/filtered_signals/  (her kayit icin .npy dosyasi)
    - outputs/processed_data/filtered_manifest.csv

Kullanim:
    python adim02_filtreleme.py
"""

import os
import sys
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# SINYAL ISLEME FONKSIYONLARI
# =============================================================================

def butterworth_bandpass_filtre_olustur(lowcut, highcut, fs, order=4):
    """
    Butterworth bandpass filtre katsayilarini olusturur.

    Projemizde belirttigimiz gibi:
    - 4. derece (order=4) yeterli keskinlik saglar, fazla grup gecikmesi yaratmaz.
    - Butterworth'un maksimum duz frekans yaniti sinyal bozulmasini minimize eder.

    Args:
        lowcut: Alt kesim frekansi (Hz). Varsayilan: 0.5 Hz
        highcut: Ust kesim frekansi (Hz). Varsayilan: 40.0 Hz
        fs: Ornekleme frekansi (Hz).
        order: Filtre derecesi. Varsayilan: 4

    Returns:
        tuple: (b, a) filtre katsayilari
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def sinyal_filtrele(sinyal, fs, lowcut=None, highcut=None, order=None):
    """
    Tek bir EKG sinyaline (tek kanal) bandpass filtre uygular.

    filtfilt kullanimi sifir faz kaymasi saglar -> QRS zamanlamasi korunur.

    Args:
        sinyal: 1D numpy array (zaman adimi,)
        fs: Ornekleme frekansi (Hz)
        lowcut: Alt kesim frekansi. Varsayilan: config.BANDPASS_LOW
        highcut: Ust kesim frekansi. Varsayilan: config.BANDPASS_HIGH
        order: Filtre derecesi. Varsayilan: config.BANDPASS_ORDER

    Returns:
        numpy array: Filtrelenmis sinyal (ayni boyut)
    """
    if lowcut is None:
        lowcut = config.BANDPASS_LOW
    if highcut is None:
        highcut = config.BANDPASS_HIGH
    if order is None:
        order = config.BANDPASS_ORDER

    b, a = butterworth_bandpass_filtre_olustur(lowcut, highcut, fs, order)

    # filtfilt: Ileri-geri filtreleme -> sifir faz kaymasi
    # padlen: Sinyal cok kisaysa padding uzunlugunu ayarla
    padlen = min(3 * max(len(b), len(a)), len(sinyal) - 1)
    if padlen < 1:
        return sinyal  # Cok kisa sinyal, filtreleme yapilamaz

    filtrelenmis = filtfilt(b, a, sinyal, padlen=padlen)
    return filtrelenmis


def cok_kanalli_filtrele(sinyal_2d, fs):
    """
    12 derivasyonlu EKG sinyalinin her kanalina bagimsiz filtreleme uygular.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        fs: Ornekleme frekansi (Hz)

    Returns:
        numpy array: Filtrelenmis sinyal (ayni boyut)
    """
    filtrelenmis = np.zeros_like(sinyal_2d)
    for kanal_idx in range(sinyal_2d.shape[0]):
        filtrelenmis[kanal_idx] = sinyal_filtrele(sinyal_2d[kanal_idx], fs)
    return filtrelenmis


def alt_ornekle(sinyal_2d, orijinal_fs, hedef_fs):
    """
    Sinyali hedef frekansa alt ornekler.

    Projemizde belirttigimiz gibi:
    - 500 Hz -> 250 Hz alt-ornekleme
    - Klinik bilgi kaybi sifir (Nyquist: 125 Hz'e kadar bilesenler korunur)

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        orijinal_fs: Orijinal ornekleme frekansi (Hz)
        hedef_fs: Hedef ornekleme frekansi (Hz)

    Returns:
        numpy array: Alt orneklenmis sinyal (kanal_sayisi, yeni_zaman_adimi)
    """
    if orijinal_fs == hedef_fs:
        return sinyal_2d

    # Yeni zaman adimi sayisini hesapla
    orijinal_uzunluk = sinyal_2d.shape[1]
    yeni_uzunluk = int(orijinal_uzunluk * hedef_fs / orijinal_fs)

    alt_orneklenmis = np.zeros((sinyal_2d.shape[0], yeni_uzunluk))
    for kanal_idx in range(sinyal_2d.shape[0]):
        alt_orneklenmis[kanal_idx] = resample(sinyal_2d[kanal_idx], yeni_uzunluk)

    return alt_orneklenmis


def z_score_normalize(sinyal_2d):
    """
    Her derivasyonun ortalamasini 0, standart sapmasini 1 yapar.

    Projemizde belirttigimiz gibi:
    - Z-score normalizasyonu cihaz bazli genlik farklarini esitler.
    - Farkli veri setleri arasindaki domain shift'i azaltir.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array

    Returns:
        numpy array: Normalize edilmis sinyal (ayni boyut)
    """
    normalize = np.zeros_like(sinyal_2d, dtype=np.float32)
    for kanal_idx in range(sinyal_2d.shape[0]):
        kanal = sinyal_2d[kanal_idx]
        ortalama = np.mean(kanal)
        std = np.std(kanal)
        if std > 1e-8:  # Sifira bolmeyi onle (elektrot kopmasi durumu)
            normalize[kanal_idx] = (kanal - ortalama) / std
        else:
            normalize[kanal_idx] = kanal - ortalama
    return normalize


# =============================================================================
# ANA FILTRELEME PIPELINE'I
# =============================================================================

def filtreleme_pipeline():
    """
    Tum kayitlara filtreleme, alt ornekleme ve normalizasyon uygular.

    Islem Akisi:
        1. raw_manifest.csv okunur.
        2. Sadece gecerli kayitlar (etiket var + dosya mevcut) secilir.
        3. Her kayit icin:
           a. wfdb ile sinyal okunur
           b. 500 Hz -> 250 Hz alt ornekleme
           c. 0.5-40 Hz Butterworth bandpass filtre
           d. Z-score normalizasyon
           e. .npy dosyasi olarak kaydedilir
        4. filtered_manifest.csv kaydedilir.

    Returns:
        pd.DataFrame: Filtrelenmis manifest DataFrame'i.
    """
    print("=" * 70)
    print("BirunAI -- Adim 2: Filtreleme ve Alt Ornekleme")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "raw_manifest.csv")
    print(f"\n[1/4] raw_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"raw_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim01_veri_yukleme.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    print(f"      Toplam kayit: {len(df)}")

    # --- 2. Gecerli kayitlari sec ---
    print(f"\n[2/4] Gecerli kayitlar seciliyor...")
    gecerli = df[(df["label"].notna()) & (df["file_exists"] == True)].copy()
    print(f"      Gecerli kayit: {len(gecerli)}")

    # --- 3. Cikti dizinini olustur ---
    cikti_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    os.makedirs(cikti_dizini, exist_ok=True)

    # --- 4. Filtreleme dongusu ---
    print(f"\n[3/4] Filtreleme isleniyor...")
    print(f"      Orijinal Fs   : {config.ORIGINAL_FS} Hz")
    print(f"      Hedef Fs      : {config.TARGET_FS} Hz")
    print(f"      Bandpass      : {config.BANDPASS_LOW}-{config.BANDPASS_HIGH} Hz")
    print(f"      Filtre Order  : {config.BANDPASS_ORDER}")
    print(f"      Normalizasyon : Z-score")

    basarili = 0
    basarisiz = 0
    hatali_kayitlar = []

    for ecg_id, satir in tqdm(gecerli.iterrows(), total=len(gecerli),
                               desc="      Filtreleme"):
        try:
            # 4a. Sinyal oku
            dosya_yolu = os.path.join(config.PTBXL_ROOT, satir["filename_hr"])
            kayit = wfdb.rdsamp(dosya_yolu)
            sinyal = kayit[0]  # (zaman_adimi, kanal_sayisi)
            meta = kayit[1]

            # Transpoz: (kanal_sayisi, zaman_adimi) formatina cevir
            sinyal = sinyal.T  # (12, 5000) bekle

            # Orijinal ornekleme frekansini belirle
            orijinal_fs = meta.get("fs", config.ORIGINAL_FS)
            if orijinal_fs is None:
                orijinal_fs = config.ORIGINAL_FS

            # 4b. Alt ornekleme: 500 Hz -> 250 Hz
            sinyal = alt_ornekle(sinyal, orijinal_fs, config.TARGET_FS)

            # 4c. Butterworth bandpass filtre
            sinyal = cok_kanalli_filtrele(sinyal, config.TARGET_FS)

            # 4d. Z-score normalizasyon
            sinyal = z_score_normalize(sinyal)

            # 4e. .npy olarak kaydet
            cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
            np.save(cikti_dosyasi, sinyal.astype(np.float32))

            basarili += 1

        except Exception as e:
            basarisiz += 1
            hatali_kayitlar.append((ecg_id, str(e)))

    print(f"\n      Basarili: {basarili}")
    print(f"      Basarisiz: {basarisiz}")

    if basarisiz > 0:
        print(f"\n      Ilk 5 hatali kayit:")
        for ecg_id, hata in hatali_kayitlar[:5]:
            print(f"        ecg_id={ecg_id}: {hata[:80]}")

    # --- 5. Filtrelenmis manifest olustur ---
    print(f"\n[4/4] filtered_manifest.csv kaydediliyor...")

    # Basarili kayitlari isaretlemek icin filtered_signal_path ekle
    basarili_ids = set()
    for ecg_id, _ in gecerli.iterrows():
        cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
        if os.path.exists(cikti_dosyasi):
            basarili_ids.add(ecg_id)

    gecerli["filtered"] = gecerli.index.isin(basarili_ids)
    gecerli["filtered_path"] = gecerli.index.map(
        lambda x: os.path.join("filtered_signals", f"{x}.npy") if x in basarili_ids else None
    )

    manifest_cikti = os.path.join(config.PROCESSED_DATA_DIR, "filtered_manifest.csv")
    gecerli.to_csv(manifest_cikti)
    print(f"      Kaydedildi: {manifest_cikti}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    filtrelenmis = gecerli[gecerli["filtered"] == True]
    print(f"  Toplam islenen      : {len(gecerli)}")
    print(f"  Basarili filtrelen  : {len(filtrelenmis)}")
    print(f"  Hedef sinyal boyutu : ({config.NUM_LEADS}, {config.TARGET_LENGTH})")
    print(f"                        = 12 kanal x {config.TARGET_LENGTH} zaman adimi")

    # Sinif dagilimi (filtrelenmis)
    sinif_dag = filtrelenmis["label"].value_counts().sort_index()
    print(f"\n  Filtrelenmis Sinif Dagilimi:")
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    # Ornek sinyal boyutu kontrolu
    ornek_dosya = os.path.join(cikti_dizini, f"{filtrelenmis.index[0]}.npy")
    if os.path.exists(ornek_dosya):
        ornek = np.load(ornek_dosya)
        print(f"\n  Ornek sinyal shape  : {ornek.shape}")
        print(f"  Ornek sinyal dtype  : {ornek.dtype}")
        print(f"  Ornek sinyal min    : {ornek.min():.4f}")
        print(f"  Ornek sinyal max    : {ornek.max():.4f}")
        print(f"  Ornek sinyal mean   : {ornek.mean():.6f}")
        print(f"  Ornek sinyal std    : {ornek.std():.4f}")

    print("\n" + "=" * 70)
    print("Adim 2 tamamlandi. Sonraki adim: adim03_kalite_kontrol.py")
    print("=" * 70)

    return gecerli


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = filtreleme_pipeline()
