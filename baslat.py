"""
baslat.py — BirunAI EKG Siniflandirma: Tek Tikla Baslatici
=============================================================

Bu script su adimlari sirayla calistirir:
    1. GPU kontrolu
    2. Dashboard'u arka planda baslat
    3. Model egitimi (Adim 8)
    4. Test degerlendirmesi (Adim 9)
    5. GradCAM uretimi (Adim 10)

Kullanim:
    python baslat.py
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
            print(f"  GPU icin: pip install torch --index-url https://download.pytorch.org/whl/cu121")

        return True

    except ImportError:
        print("\n  [HATA] PyTorch kurulu degil!")
        print("  Kurulum: pip install torch torchvision")
        return False


def main():
    """Ana baslatma fonksiyonu."""
    print("\n" + "=" * 70)
    print("  ⚡ BirunAI EKG Siniflandirma — Egitim Pipeline")
    print("=" * 70)

    # 1. GPU kontrol
    if not gpu_kontrol():
        return

    # 2. Dashboard baslat
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

    # 3. Egitim
    print(f"\n{'='*70}")
    print("  Model egitimi basliyor...")
    print(f"{'='*70}\n")

    from adim08_egitim import egitim_pipeline
    egitim_sonuc = egitim_pipeline()

    # 4. Degerlendirme
    print(f"\n{'='*70}")
    print("  Test degerlendirmesi basliyor...")
    print(f"{'='*70}\n")

    from adim09_degerlendirme import degerlendirme_pipeline
    metrikler = degerlendirme_pipeline()

    # 5. GradCAM
    print(f"\n{'='*70}")
    print("  GradCAM uretimi basliyor...")
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
