"""
adim00_veri_birlestirme.py — BirunAI EKG Siniflandirma: Adim 0 – Veri Birlestirme
===================================================================================

Bu modul, 5 farkli veri setini tarar, SNOMED-CT kodlarini 3 sinifa esler
ve tek bir unified_manifest.csv dosyasi uretir.

Veri Setleri:
    1. CPSC 2018          — 6,877 kayit
    2. CPSC 2018 Extra    — 3,453 kayit
    3. PTB-XL (Challenge) — 21,837 kayit
    4. Georgia             — 10,344 kayit
    5. ECG Arrhythmia      — 45,152 kayit (Chapman-Shaoxing/Ningbo)

Islem Akisi:
    1. Her veri seti dizinindeki .hea dosyalari taranir
    2. .hea dosyasindan #Dx: satiri okunur -> SNOMED-CT kodlari cikarilir
    3. SNOMED-CT kodlari SNOMED_TO_LABEL tablosuyla 3 sinifa eslenir
    4. Cakisma onceligi: Iletim (2) > Ritim (1) > Normal (0)
    5. Tum setler birlestirilerek unified_manifest.csv uretilir

Ciktilar:
    - outputs/processed_data/unified_manifest.csv

Kullanim:
    python adim00_veri_birlestirme.py
"""

import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# .HEA DOSYASI PARSE FONKSIYONLARI
# =============================================================================

def hea_parse(hea_path):
    """
    .hea (header) dosyasini parse ederek meta verileri cikarir.

    WFDB header formati:
        Satir 1: kayit_adi lead_sayisi fs ornek_sayisi
        Satir 2-13: Her lead icin sinyal bilgisi
        # ile baslayan satirlar: Meta veri (Age, Sex, Dx, Rx, Hx, Sx)

    Args:
        hea_path: .hea dosyasinin tam yolu

    Returns:
        dict: {
            'record_name': str,
            'num_leads': int,
            'fs': int,
            'num_samples': int,
            'age': str,
            'sex': str,
            'dx_codes': list[int],  # SNOMED-CT kodlari
        }
        veya None (dosya okunamaz/bozuksa)
    """
    try:
        with open(hea_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if not lines:
            return None

        # Satir 1: kayit_adi lead_sayisi fs ornek_sayisi
        header_parts = lines[0].strip().split()
        record_name = header_parts[0]
        num_leads = int(header_parts[1])
        fs = int(float(header_parts[2]))  # Bazen float olarak gelebilir
        num_samples = int(header_parts[3]) if len(header_parts) > 3 else 0

        # Meta verileri parse et
        age = "Unknown"
        sex = "Unknown"
        dx_codes = []

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # "# Age: 74" veya "#Age: 74" formatlarini destekle
                content = line.lstrip('#').strip()

                if content.lower().startswith('age:'):
                    age = content.split(':', 1)[1].strip()
                elif content.lower().startswith('sex:'):
                    sex = content.split(':', 1)[1].strip()
                elif content.lower().startswith('dx:'):
                    dx_str = content.split(':', 1)[1].strip()
                    # Virgülle ayrilmis SNOMED-CT kodlari
                    for code_str in dx_str.split(','):
                        code_str = code_str.strip()
                        if code_str.isdigit():
                            dx_codes.append(int(code_str))

        return {
            'record_name': record_name,
            'num_leads': num_leads,
            'fs': fs,
            'num_samples': num_samples,
            'age': age,
            'sex': sex,
            'dx_codes': dx_codes,
        }

    except Exception as e:
        return None


def snomed_to_label(dx_codes):
    """
    SNOMED-CT kodlarindan 3-sinifli etikete donusturur.

    Cakisma mantigi:
        Bir hastada birden fazla SNOMED-CT kodu olabilir.
        Oncelik: Iletim (2) > Ritim (1) > Normal (0)
        En yuksek oncelikli sinif atanir.

    Args:
        dx_codes: list[int] — SNOMED-CT kodlari

    Returns:
        int or None: 0, 1, 2 veya None (eslenemezse)
    """
    if not dx_codes:
        return None

    bulunan_etiketler = set()

    for code in dx_codes:
        if code in config.SNOMED_TO_LABEL:
            bulunan_etiketler.add(config.SNOMED_TO_LABEL[code])

    if not bulunan_etiketler:
        return None

    # Cakisma onceligi: Iletim (2) > Ritim (1) > Normal (0)
    return max(bulunan_etiketler, key=lambda x: config.LABEL_PRIORITY.get(x, 0))


# =============================================================================
# VERI SETI TARAMA
# =============================================================================

def scan_dataset(dataset_name, dataset_root):
    """
    Bir veri seti dizinindeki tum .hea dosyalarini tarar ve manifest satirlari uretir.

    Args:
        dataset_name: str — Veri seti adi (orn: "cpsc_2018")
        dataset_root: str — Veri seti kok dizini

    Returns:
        list[dict]: Her kayit icin bir dict listesi
    """
    if not os.path.exists(dataset_root):
        print(f"  [UYARI] Dizin bulunamadi: {dataset_root}")
        return []

    # Tum .hea dosyalarini bul
    hea_files = glob.glob(os.path.join(dataset_root, "**", "*.hea"), recursive=True)

    if not hea_files:
        print(f"  [UYARI] {dataset_name}: .hea dosyasi bulunamadi")
        return []

    records = []
    eslenmayan_kodlar = set()

    for hea_path in tqdm(hea_files, desc=f"  {dataset_name:18s}", ncols=80):
        meta = hea_parse(hea_path)
        if meta is None:
            continue

        # 12-lead kontrolu
        if meta['num_leads'] != 12:
            continue

        # SNOMED-CT -> 3-sinif
        label = snomed_to_label(meta['dx_codes'])

        # Eslenmayan kodlari kaydet
        for code in meta['dx_codes']:
            if code not in config.SNOMED_TO_LABEL:
                eslenmayan_kodlar.add(code)

        # .mat dosyasinin var olup olmadigini kontrol et
        mat_path = hea_path.replace('.hea', '.mat')
        if not os.path.exists(mat_path):
            continue

        # signal_path: .hea/.mat dosyasinin uzantisiz yolu
        signal_path = hea_path.replace('.hea', '')

        # Benzersiz ecg_id: dataset_prefix + dosya adi
        base_name = os.path.splitext(os.path.basename(hea_path))[0]
        ecg_id = f"{dataset_name}_{base_name}"

        records.append({
            'ecg_id': ecg_id,
            'dataset_source': dataset_name,
            'signal_path': signal_path,
            'label': label,
            'original_fs': meta['fs'],
            'num_samples': meta['num_samples'],
            'num_leads': meta['num_leads'],
            'age': meta['age'],
            'sex': meta['sex'],
            'dx_codes': ','.join(map(str, meta['dx_codes'])),
        })

    if eslenmayan_kodlar:
        print(f"    Eslenmayan SNOMED kodlari ({len(eslenmayan_kodlar)}): "
              f"{sorted(eslenmayan_kodlar)[:10]}...")

    return records


# =============================================================================
# ANA BIRLESTIRME PIPELINE'I
# =============================================================================

def birlestirme_pipeline():
    """
    Tum veri setlerini tarar, etiketler ve unified_manifest.csv uretir.

    Returns:
        pd.DataFrame: unified_manifest DataFrame'i
    """
    print("=" * 70)
    print("BirunAI — Adim 0: Multi-Dataset Birlestirme")
    print("=" * 70)

    # --- 1. Veri setlerini tara ---
    print(f"\n[1/3] Veri setleri taraniyor...")
    print(f"      Toplam {len(config.DATASET_PATHS)} veri seti\n")

    tum_kayitlar = []

    for ds_name, ds_path in config.DATASET_PATHS.items():
        kayitlar = scan_dataset(ds_name, ds_path)
        print(f"    -> {ds_name}: {len(kayitlar)} kayit")
        tum_kayitlar.extend(kayitlar)

    print(f"\n      Toplam taranan: {len(tum_kayitlar)} kayit")

    # --- 2. DataFrame olustur ---
    print(f"\n[2/3] Manifest olusturuluyor...")

    df = pd.DataFrame(tum_kayitlar)

    # Etiketlenemeyen kayitlari ayir
    etiketli = df[df['label'].notna()].copy()
    etiketsiz = df[df['label'].isna()]

    print(f"      Etiketlenen   : {len(etiketli)}")
    print(f"      Etiketlenemeyen: {len(etiketsiz)}")

    # label'i int'e cevir
    etiketli['label'] = etiketli['label'].astype(int)

    # Mukerrer kontrol (ecg_id bazinda)
    mukerrer_sayisi = etiketli['ecg_id'].duplicated().sum()
    if mukerrer_sayisi > 0:
        print(f"      [UYARI] {mukerrer_sayisi} mukerrer ecg_id bulundu, siliniyor...")
        etiketli = etiketli.drop_duplicates(subset='ecg_id', keep='first')

    # --- 3. Kaydet ---
    print(f"\n[3/3] unified_manifest.csv kaydediliyor...")

    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "unified_manifest.csv")
    etiketli.to_csv(cikti_yolu, index=False)
    print(f"      Kaydedildi: {cikti_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    print(f"  Toplam gecerli kayit : {len(etiketli)}")
    print(f"  Etiketlenemeyen      : {len(etiketsiz)}")

    # Veri seti bazli dagilim
    print(f"\n  Veri Seti Bazli Dagilim:")
    ds_dagilim = etiketli['dataset_source'].value_counts()
    for ds_name, sayi in ds_dagilim.items():
        oran = sayi / len(etiketli) * 100
        print(f"    {ds_name:18s}: {sayi:6d} ({oran:5.1f}%)")

    # Sinif dagilimi
    print(f"\n  Sinif Dagilimi:")
    sinif_dag = etiketli['label'].value_counts().sort_index()
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    # Fs dagilimi
    print(f"\n  Ornekleme Frekansi Dagilimi:")
    fs_dag = etiketli['original_fs'].value_counts().sort_index()
    for fs_val, sayi in fs_dag.items():
        print(f"    {int(fs_val)} Hz: {sayi} kayit")

    # Sinif x Veri Seti capraz tablosu
    print(f"\n  Sinif x Veri Seti Capraz Tablosu:")
    cross = pd.crosstab(etiketli['dataset_source'], etiketli['label'])
    cross.columns = [config.LABEL_NAMES.get(c, str(c)) for c in cross.columns]
    print(cross.to_string(index=True))

    print("\n" + "=" * 70)
    print("Adim 0 tamamlandi. Sonraki adim: adim02_filtreleme.py (guncel)")
    print("=" * 70)

    return etiketli


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = birlestirme_pipeline()
