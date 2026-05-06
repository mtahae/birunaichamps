"""
baslat.py — BirunAI EKG Siniflandirma: Tek Tikla Baslatici (Multi-Dataset v5)
================================================================================

Bu script tum pipeline'i sirayla calistirir:
    Adim 0: Veri Birlestirme (5 dataset -> unified_manifest.csv)
    Adim 1: Genel Kalite Kontrol (mukerrer tespiti + sinyal dogrulama)
    Adim 2: Filtreleme ve Alt Ornekleme (500 Hz -> 250 Hz, bandpass)
    Adim 3: Sinyal Kalite Kontrol (flat-line, clipping)
    Adim 4: Segmentasyon (-> 12x2500)
    Adim 6: Veri Bolme (Train/Val/Test)
    Adim 6b: SMOTE Oversampling (Ritim sinifi dengeleme)
    Adim 8: Model Egitimi
    Adim 9: Test Degerlendirmesi
    Adim 10: GradCAM

Kullanim:
    python baslat.py                  # Tum pipeline
    python baslat.py --sadece-egitim  # Sadece egitim (veri islem adimlarini atla)

"""

import os
import sys
import time
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def gpu_kontrol():
    """GPU durumunu kontrol et."""
    print("=" * 70)
    print("BirunAI -- Sistem Kontrolu")
    print("=" * 70)

    try:
        import torch
        print(f"\n  Python     : {sys.version.split()[0]}")
        print(f"  PyTorch    : {torch.__version__}")
        print(f"  CUDA       : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  GPU        : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  VRAM       : {props.total_memory / 1e9:.1f} GB")
            print(f"  CUDA Versiyon: {torch.version.cuda}")
        else:
            print(f"\n  [UYARI] GPU bulunamadi!")
            print(f"  CPU ile egitim yapilacak (yavas olabilir).")

        return True

    except ImportError:
        print("\n  [HATA] PyTorch kurulu degil!")
        return False


def veri_isleme_adimlari():
    """Adim 0-6b: Veri isleme pipeline'i."""

    # Adim 0: Birlestirme
    print(f"\n{'='*70}")
    print("  Adim 0: Multi-Dataset Birlestirme")
    print(f"{'='*70}\n")
    from adim00_veri_birlestirme import birlestirme_pipeline
    birlestirme_pipeline()

    # Adim 1: Genel QC
    print(f"\n{'='*70}")
    print("  Adim 1: Genel Kalite Kontrol")
    print(f"{'='*70}\n")
    from adim01_kalite_kontrol_genel import kalite_kontrol_genel
    kalite_kontrol_genel()

    # Adim 2: Filtreleme
    print(f"\n{'='*70}")
    print("  Adim 2: Filtreleme ve Alt Ornekleme")
    print(f"{'='*70}\n")
    from adim02_filtreleme import filtreleme_pipeline
    filtreleme_pipeline()

    # Adim 3: Sinyal QC
    print(f"\n{'='*70}")
    print("  Adim 3: Sinyal Kalite Kontrol")
    print(f"{'='*70}\n")
    from adim03_kalite_kontrol import kalite_kontrol_pipeline
    kalite_kontrol_pipeline()

    # Adim 4: Segmentasyon
    print(f"\n{'='*70}")
    print("  Adim 4: Segmentasyon")
    print(f"{'='*70}\n")
    from adim04_segmentasyon import segmentasyon_pipeline
    segmentasyon_pipeline()

    # Adim 6: Veri Bolme
    print(f"\n{'='*70}")
    print("  Adim 6: Veri Bolme (Train/Val/Test)")
    print(f"{'='*70}\n")
    from adim06_veri_bolme import veri_bolme_pipeline
    veri_bolme_pipeline()

    # Adim 6b: SMOTE
    print(f"\n{'='*70}")
    print("  Adim 6b: SMOTE Oversampling")
    print(f"{'='*70}\n")
    from adim06b_oversampling import smote_pipeline
    smote_pipeline()


def main():
    """Ana baslatma fonksiyonu."""
    print("\n" + "=" * 70)
    print("  BirunAI EKG Siniflandirma -- Multi-Dataset Pipeline v5")
    print("=" * 70)

    # 1. GPU kontrol
    if not gpu_kontrol():
        return

    # Komut satiri argumanlari
    sadece_egitim = "--sadece-egitim" in sys.argv

    # 2. Veri isleme (eger gerekiyorsa)
    if not sadece_egitim:
        veri_isleme_adimlari()
    else:
        print(f"\n  [BILGI] --sadece-egitim modunda. Veri adimları atlaniyor.")
        # SMOTE manifest kontrol
        smote_manifest = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest_smote.csv")
        normal_manifest = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
        if not os.path.exists(smote_manifest) and not os.path.exists(normal_manifest):
            print("  [HATA] Egitim manifesti bulunamadi! Once veri isleme adimlari calistirilmali.")
            return

    # 3. Dashboard baslat
    print(f"\n{'='*70}")
    print("  Dashboard baslatiliyor...")
    print(f"{'='*70}")

    try:
        from dashboard import dashboard_thread_baslat
        dashboard_thread_baslat(5000)
        time.sleep(1)
        webbrowser.open("http://localhost:5000")
        print(f"  Dashboard acildi: http://localhost:5000")
    except Exception as e:
        print(f"  [UYARI] Dashboard baslatilamadi: {e}")
        print(f"  Egitim dashboard'suz devam edecek.")

    # 4. Egitim
    print(f"\n{'='*70}")
    print("  Adim 8: Model Egitimi")
    print(f"{'='*70}\n")

    from adim08_egitim import egitim_pipeline
    egitim_sonuc = egitim_pipeline()

    # 5. Degerlendirme
    print(f"\n{'='*70}")
    print("  Adim 9: Test Degerlendirmesi")
    print(f"{'='*70}\n")

    from adim09_degerlendirme import degerlendirme_pipeline
    metrikler = degerlendirme_pipeline()

    # 6. GradCAM
    print(f"\n{'='*70}")
    print("  Adim 10: GradCAM Uretimi")
    print(f"{'='*70}\n")

    from adim10_gradcam import gradcam_pipeline
    gradcam_pipeline()

    # Ozet
    print(f"\n{'='*70}")
    print(f"  ✅ TUM ISLEMLER TAMAMLANDI!")
    print(f"{'='*70}")
    print(f"\n  Sonuclar:")
    print(f"    En iyi Val F1    : {egitim_sonuc['best_f1']:.4f}")
    print(f"    Test Macro F1    : {metrikler['macro_f1']:.4f}")
    print(f"    Cohen Kappa      : {metrikler['cohen_kappa']:.4f}")
    print(f"\n  Dosyalar:")
    print(f"    Model            : outputs/checkpoints/best_model.pth")
    print(f"    Confusion Matrix : outputs/reports/confusion_matrix.png")
    print(f"    ROC Curves       : outputs/reports/roc_curves.png")
    print(f"    GradCAM          : outputs/reports/gradcam/")
    print(f"    Metrikler        : outputs/reports/test_metrics.json")
    print(f"\n  Dashboard hala acik: http://localhost:5000")
    print(f"{'='*70}\n")

    # Dashboard'un kapanmamasini bekle
    try:
        input("  Cikmak icin Enter'a basin...")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
