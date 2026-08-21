"""
adim05b_veri_dengeleme.py — BirunAI EKG: Adim 5b – Akilli Veri Dengeleme
=========================================================================

Normal sinifin baskinligini azaltarak Macro F1'i yukseltmeyi hedefler.
PhysioNet 2020 sampiyon takimlarinin yaklasimi: Undersampling + Quality Ranking.

Strateji:
    1. TEKNOFEST verileri DOKUNULMAZ — hepsi kalir
    2. Azinlik siniflari (AFIB, AFL, LBBB, RBBB) DOKUNULMAZ — hepsi kalir
    3. Normal sinif: SQI kalite skor sirasina gore en iyi N tanesi secilir
       - Hedef: Normal sayisi ≈ en buyuk azinlik sinifi kadar
       - TEKNOFEST Normal'leri her zaman korunur

Ciktilar:
    - outputs/processed_data/balanced_manifest.csv

Kullanim:
    python adim05b_veri_dengeleme.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def veri_dengeleme_pipeline():
    print("=" * 70)
    print("BirunAI -- Adim 5b: Akilli Veri Dengeleme (Class Balancing)")
    print("=" * 70)

    # --- 1. Segmented manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"Bulunamadi: {manifest_yolu}\n"
            "Once adim04_segmentasyon.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu)
    
    # Sadece etiketi olan (label != NaN) kayitlari al
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)
    
    print(f"\n[1/4] Mevcut Dagilim:")
    print(f"      Toplam kayit: {len(df)}")
    
    sinif_dag = df['label'].value_counts().sort_index()
    for s, n in sinif_dag.items():
        oran = n / sinif_dag.sum() * 100
        print(f"        [{int(s)}] {config.LABEL_NAMES.get(int(s), '?'):6s}: {n:6d} ({oran:5.1f}%)")

    # --- 2. TEKNOFEST ve Internet ayir ---
    df_tekno = df[df['dataset_source'] == 'TEKNOFEST'].copy()
    df_inter = df[df['dataset_source'] != 'TEKNOFEST'].copy()
    
    tekno_normal = df_tekno[df_tekno['label'] == 0]
    tekno_diger = df_tekno[df_tekno['label'] != 0]
    
    inter_normal = df_inter[df_inter['label'] == 0]
    inter_diger = df_inter[df_inter['label'] != 0]
    
    print(f"\n[2/4] Kaynak Analizi:")
    print(f"      TEKNOFEST  : {len(df_tekno):6d} (Normal: {len(tekno_normal)}, Diger: {len(tekno_diger)})")
    print(f"      Internet   : {len(df_inter):6d} (Normal: {len(inter_normal)}, Diger: {len(inter_diger)})")

    # --- 3. Hedef hesapla ---
    # Tum azinlik siniflarini koru
    azinlik_toplam = len(tekno_diger) + len(inter_diger)
    
    # En buyuk azinlik sinifini bul (RBBB veya AFL genellikle)
    azinlik_siniflar = df[df['label'] != 0]['label'].value_counts()
    en_buyuk_azinlik = azinlik_siniflar.max() if len(azinlik_siniflar) > 0 else 5000
    
    # Hedef Normal sayisi: en buyuk azinlik sinifinin ~1.1 kati
    # (Biraz fazla tutuyoruz ki tam esit olmasin, dogal kalsin)
    hedef_normal = int(en_buyuk_azinlik * 1.1)
    
    # TEKNOFEST Normal'leri her zaman korunur
    tekno_normal_sayisi = len(tekno_normal)
    
    # Internet Normal'lerden secilecek
    inter_normal_hedef = max(0, hedef_normal - tekno_normal_sayisi)
    
    print(f"\n[3/4] Dengeleme Hesabi:")
    print(f"      En buyuk azinlik sinifi  : {en_buyuk_azinlik}")
    print(f"      Hedef Normal sayisi      : {hedef_normal}")
    print(f"      TEKNOFEST Normal (korunan): {tekno_normal_sayisi}")
    print(f"      Internet Normal (secilecek): {inter_normal_hedef} / {len(inter_normal)}")
    
    # --- 4. Akilli Normal secimi ---
    # SQI skoru varsa ona gore sirala, yoksa rastgele sec
    if inter_normal_hedef >= len(inter_normal):
        # Zaten yeterince az, hepsini al
        secilen_inter_normal = inter_normal
        print(f"      -> Internet Normal zaten az, hepsi alindi.")
    else:
        # SQI bazli secim: qc_flat_channels en az olanlari sec (en temiz sinyaller)
        if 'qc_flat_channels' in inter_normal.columns:
            # Duz kanal sayisi en az olan = en kaliteli
            inter_normal_sirali = inter_normal.sort_values('qc_flat_channels', ascending=True)
            secilen_inter_normal = inter_normal_sirali.head(inter_normal_hedef)
            print(f"      -> SQI bazli secim: En kaliteli {inter_normal_hedef} Normal secildi.")
        else:
            # SQI yoksa rastgele sec (seed ile tekrar uretilebilir)
            secilen_inter_normal = inter_normal.sample(n=inter_normal_hedef, random_state=config.SEED)
            print(f"      -> Rastgele secim: {inter_normal_hedef} Normal secildi.")
    
    # --- 5. Dengeli manifest birlestir ---
    # 1. TEKNOFEST hepsi (Normal + Diger)
    # 2. Internet azinlik hepsi
    # 3. Internet Normal (secilmis)
    dengeli_df = pd.concat([
        df_tekno,                  # TEKNOFEST tamami
        inter_diger,               # Internet azinlik siniflari tamami
        secilen_inter_normal,      # Internet Normal (secilmis)
    ], ignore_index=True)
    
    # Karistir
    dengeli_df = dengeli_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    
    # Kaydet
    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "balanced_manifest.csv")
    dengeli_df.to_csv(cikti_yolu, index=False)
    
    # --- Ozet ---
    print(f"\n[4/4] Dengeli Dagilim:")
    print(f"      Toplam: {len(dengeli_df)} (onceki: {len(df)}, cikarilan: {len(df) - len(dengeli_df)})")
    
    sinif_dag_yeni = dengeli_df['label'].value_counts().sort_index()
    for s, n in sinif_dag_yeni.items():
        oran = n / sinif_dag_yeni.sum() * 100
        onceki = sinif_dag.get(s, 0)
        fark = n - onceki
        fark_str = "=" if fark == 0 else f"{fark:+d}"
        print(f"        [{int(s)}] {config.LABEL_NAMES.get(int(s), '?'):6s}: {n:6d} ({oran:5.1f}%) [{fark_str}]")

    print(f"\n      Kaydedildi: {cikti_yolu}")
    
    print("\n" + "=" * 70)
    print("Adim 5b tamamlandi. Sonraki: adim06_veri_bolme.py")
    print("=" * 70)
    
    return dengeli_df


if __name__ == "__main__":
    veri_dengeleme_pipeline()
