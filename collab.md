🚀 GÜN 1: Ortam Kurulumu ve Veriye İlk Bakış
Sorumlu: TAHA

Bu bölümde projemizin omurgasını oluşturacak çalışma ortamını hazırlıyoruz. Jürinin en çok dikkat ettiği konulardan biri yeniden üretilebilirlik (reproducibility) ilkesidir. Bu nedenle kodumuzun farklı bilgisayarlarda veya jürinin incelemesi sırasında hata vermemesi için:

Gerekli tüm kütüphanelerin belirli sürümlerini kuruyoruz.
Google Drive bağlantımızı statik değil, bir değişken (DATA_PATH) üzerinden dinamik olarak tanımlıyoruz.
Rastgelelik (randomness) içeren işlemlerin her seferinde aynı sonucu vermesi için SEED değerimizi sabitliyoruz.

🗂️ GÜN 2: Büyük Çeviri İşlemi (Sınıf Haritalama) ve Yüzleşme Vakti
Bugünkü Sorumlu: kaan güneeeş Aleyküm Selam Kardeşim

taha babadan bayrağı devraldın. 1. Gün sonunda ortamımızı kurduk, verilerimizi Drive'a çektik ve her şey tıkır tıkır çalışıyor. Bugün sıra sende.

Şu an ne durumdayız? Elimizde ptbxl_database.csv adında devasa bir hasta kayıt defteri var. Bu defterde binlerce hastanın EKG'sine bakıp doktorların aldığı karmaşık tıbbi notlar (SCP kodları) var. Doktorlar "AFIB", "LBBB", "SVT" gibi onlarca farklı ve spesifik kod kullanmış.

Peki bugün senin görevin ne ve bunu NEDEN yapıyoruz? Teknofest Jürisi bizden (1. Aşama modeli için) bu kadar karmaşık detaylar istemiyor. Bizden sadece 3 büyük, üst sepet (sınıf) istiyor:

Normal EKG: Kalbinde sorun olmayanlar.
Ritim Bozuklukları: Kalbi düzensiz, çok hızlı veya çok yavaş atanlar.
İletim Bozuklukları: Kalbindeki elektrik sinyali bir yerlerde takılan veya gecikenler.
Senin bugünkü ana görevin bir "Tıbbi Çevirmen" olmak! Kod yazarak o karmaşık doktor kodlarını okuyup, hastaları bu 3 basit sepete paylaştıracaksın (Biz buna yapay zeka dünyasında Mapping / Haritalama diyoruz).

Bunu yaparken atacağın adımlar (Yol Haritası):

Dosyayı Aç: Pandas kütüphanesiyle ptbxl_database.csv dosyasını okutacaksın.
Kuralı Yaz: Bir fonksiyon yazacaksın. Diyeceksin ki; "Eğer hastada LBBB (Sol Dal Bloğu) kodu varsa, bu hastayı al 'İletim Bozukluğu' sepetine at. Eğer AFIB (Atriyal Fibrilasyon) kodu varsa 'Ritim Bozukluğu' sepetine at. Eğer sadece NORM varsa 'Normal' sepetine at."
Gerçekle Yüzleş (En Önemli Kısım!): Hastaları sepetlere ayırdıktan sonra, hangi sepette kaç hasta olduğunu sayıp bunu şık bir Sütun Grafiği (Bar Chart) ile ekrana çizeceksin.
Neden grafiği çiziyoruz? (Jüri buna bakar!) Grafiği çizdiğinde muhtemelen göreceksin ki; "Normal" sepetinde on binlerce hasta varken, "Ritim Bozukluğu" sepetinde çok daha az hasta var. İşte buna Sınıf Dengesizliği (Class Imbalance) denir. Eğer bunu görmezden gelirsek, yapay zekamız tembelleşir ve "Aman nasıl olsa çoğu kişi sağlıklı, ben herkese Normal diyeyim geçeyim" der. Bu da projemizin elenmesi demektir. Sen bugün bu grafiği çizerek teşhisi koyacaksın, ilerleyen günlerde modelci arkadaşımız bu dengesizliği matematiksel olarak (Focal Loss ile) çözecek.

Hadi bakalım, klavye sende! Önce veriyi oku, sonra sepetlere ayır ve o grafiği karşımıza çıkar. Kolay gelsin!

GÜN 1 VE GÜN 2 BUNLAR YAPILDI.


GÜN 3: Sinyal Ön İşleme ve Gürültü Filtreleme
Sorumlu: Mert

Amaç: Ham EKG sinyallerini model eğitimine uygun hale getirmek. Gerekçe: Ham EKG verilerinde iki ana hata kaynağı bulunur:

Taban Çizgisi Kayması (Baseline Wander): Genellikle solunum kaynaklı, 0.5 Hz altındaki düşük frekanslı değişimlerdir.
EMG ve Şebeke Gürültüsü: Kas aktivitesi ve 50/60 Hz elektrik şebekesi kaynaklı yüksek frekanslı parazitlerdir.
Yöntem: Bu gürültüleri elimine etmek için 0.5 - 40 Hz aralığında 3. dereceden Butterworth bant geçiren (bandpass) filtre uygulanacaktır. EKG dalga morfolojisinde (P, QRS, T dalgaları) faz kaymasını (phase shift) engellemek amacıyla ileri-geri filtreleme (scipy.signal.filtfilt) yöntemi kullanılmalıdır.

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
    Sinyali hedef frekinasa alt ornekler.

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



GÜN 4 : KALİTE KONTROL

"""
adim03_kalite_kontrol.py — BirunAI EKG Siniflandirma: Adim 3 – Kalite Kontrol
==============================================================================

Bu modul, filtrelenmis EKG sinyallerinin kalitesini denetler ve
kullanim disi birakilacak kayitlari isaretler.

Kalite Kriterleri:
    1. Duz Sinyal Tespiti (Flat-line):
       - Standart sapma < 0.01 olan derivasyonlar "duz sinyal" olarak isaretlenir.
       - Tum derivasyonlari duz olan kayitlar elenecek (elektrot kopmasi).

    2. Asiri Genlik (Amplitude Clipping):
       - |z-score| > 20 olan orneklerin orani > %5 ise "clipping" isaretlenir.
       - Cihaz saturation veya hareket artefakti gostergesidir.

    3. PTB-XL Kalite Bayraklari (metadata):
       - ptbxl_database.csv dosyasindaki baseline_drift, static_noise,
         burst_noise, electrodes_problems bayraklari degerlendirilir.
       - electrodes_problems olan kayitlar dogrudan elenecek.

    4. Kayit Uzunlugu Kontrolu:
       - Hedef uzunluktan (2500 ornek = 10sn @ 250Hz) belirgin sapma.

Ciktilar:
    - outputs/processed_data/quality_manifest.csv
      Eklenen sutunlar: qc_flat_channels, qc_clipping, qc_electrode_problem,
                        qc_length_ok, qc_pass (tum kontrollerden geciyor mu)

Kullanim:
    python adim03_kalite_kontrol.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# KALITE KONTROL FONKSIYONLARI
# =============================================================================

def duz_sinyal_tespit(sinyal_2d, esik=0.01):
    """
    Duz sinyal (flat-line) tespiti.

    Elektrot kopmasi veya baglanti sorunlarinda bir veya birden fazla
    derivasyon duz bir cizgi gosterir (standart sapma ~ 0).

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        esik: Standart sapma esigi. Altindaki kanallar "duz" kabul edilir.

    Returns:
        int: Duz sinyal gosteren kanal sayisi (0-12)
    """
    duz_kanal_sayisi = 0
    for kanal_idx in range(sinyal_2d.shape[0]):
        if np.std(sinyal_2d[kanal_idx]) < esik:
            duz_kanal_sayisi += 1
    return duz_kanal_sayisi


def clipping_tespit(sinyal_2d, z_esik=20.0, oran_esik=0.05):
    """
    Asiri genlik (clipping/saturation) tespiti.

    Cihaz saturation veya asiri hareket artefaktlarinda sinyal
    surekli maksimum/minimum degerde kalir.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        z_esik: |z-score| esigi
        oran_esik: Asilan orneklerin toplam ornege orani esigi

    Returns:
        bool: True ise clipping var
    """
    toplam_ornek = sinyal_2d.size
    asiri_ornek = np.sum(np.abs(sinyal_2d) > z_esik)
    oran = asiri_ornek / toplam_ornek
    return oran > oran_esik


def uzunluk_kontrol(sinyal_2d, hedef_uzunluk=None, tolerans=0.1):
    """
    Sinyal uzunlugunun hedef uzunluga uygunlugunun kontrolu.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        hedef_uzunluk: Beklenen zaman adimi sayisi. Varsayilan: config.TARGET_LENGTH
        tolerans: Kabul edilebilir sapma orani (0.1 = %10)

    Returns:
        bool: True ise uzunluk kabul edilebilir
    """
    if hedef_uzunluk is None:
        hedef_uzunluk = config.TARGET_LENGTH

    gercek_uzunluk = sinyal_2d.shape[1]
    sapma = abs(gercek_uzunluk - hedef_uzunluk) / hedef_uzunluk
    return sapma <= tolerans


# =============================================================================
# ANA KALITE KONTROL PIPELINE'I
# =============================================================================

def kalite_kontrol_pipeline():
    """
    Tum filtrelenmis sinyallere kalite kontrol uygular.

    Islem Akisi:
        1. filtered_manifest.csv okunur.
        2. Her sinyal icin kalite metrikleri hesaplanir.
        3. Elektrot problemi bayraklari kontrol edilir.
        4. Nihai QC sonucu belirlenir.
        5. quality_manifest.csv kaydedilir.

    Returns:
        pd.DataFrame: Kalite kontrol edilmis manifest.
    """
    print("=" * 70)
    print("BirunAI -- Adim 3: Kalite Kontrol")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "filtered_manifest.csv")
    print(f"\n[1/3] filtered_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"filtered_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim02_filtreleme.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    filtrelenmis = df[df["filtered"] == True].copy()
    print(f"      Filtrelenmis kayit: {len(filtrelenmis)}")

    # --- 2. Kalite kontrol ---
    print(f"\n[2/3] Kalite kontrol uygulanyor...")

    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")

    qc_duz_kanallar = []
    qc_clipping = []
    qc_uzunluk = []
    qc_elektrot = []

    for ecg_id, satir in tqdm(filtrelenmis.iterrows(), total=len(filtrelenmis),
                               desc="      Kalite Kontrol"):
        # Sinyal yukle
        sinyal_dosyasi = os.path.join(sinyal_dizini, f"{ecg_id}.npy")

        try:
            sinyal = np.load(sinyal_dosyasi)

            # Test 1: Duz sinyal tespiti
            duz_sayisi = duz_sinyal_tespit(sinyal)
            qc_duz_kanallar.append(duz_sayisi)

            # Test 2: Clipping tespiti
            clip = clipping_tespit(sinyal)
            qc_clipping.append(clip)

            # Test 3: Uzunluk kontrolu
            uzunluk_ok = uzunluk_kontrol(sinyal)
            qc_uzunluk.append(uzunluk_ok)

        except Exception as e:
            qc_duz_kanallar.append(12)  # Tum kanallar "kotu"
            qc_clipping.append(True)
            qc_uzunluk.append(False)

        # Test 4: PTB-XL elektrot problemi bayragi
        # Bu sutun NaN, 0, veya elektrot ismi (orn: 'V6', 'aVR') icerebilir
        elektrot_prob = satir.get("electrodes_problems", 0)
        if pd.isna(elektrot_prob):
            has_problem = False
        elif isinstance(elektrot_prob, str):
            # String ise: '0' degeri yok demek, diger degerler problem var demek
            has_problem = elektrot_prob.strip() != "" and elektrot_prob.strip() != "0"
        else:
            has_problem = float(elektrot_prob) > 0
        qc_elektrot.append(has_problem)

    # Sonuclari DataFrame'e ekle
    filtrelenmis["qc_flat_channels"] = qc_duz_kanallar
    filtrelenmis["qc_clipping"] = qc_clipping
    filtrelenmis["qc_length_ok"] = qc_uzunluk
    filtrelenmis["qc_electrode_problem"] = qc_elektrot

    # Nihai QC karari:
    # - Tum kanallar duz degilse (en az 1 gecerli kanal)
    # - Clipping yok
    # - Uzunluk uygun
    # - Elektrot problemi yok
    filtrelenmis["qc_pass"] = (
        (filtrelenmis["qc_flat_channels"] < config.NUM_LEADS) &
        (~filtrelenmis["qc_clipping"]) &
        (filtrelenmis["qc_length_ok"]) &
        (~filtrelenmis["qc_electrode_problem"])
    )

    # --- 3. Sonuclari kaydet ---
    print(f"\n[3/3] quality_manifest.csv kaydediliyor...")
    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "quality_manifest.csv")
    filtrelenmis.to_csv(cikti_yolu)
    print(f"      Kaydedildi: {cikti_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    gecen = filtrelenmis[filtrelenmis["qc_pass"] == True]
    kalan = filtrelenmis[filtrelenmis["qc_pass"] == False]

    print(f"  Toplam kontrol edilen : {len(filtrelenmis)}")
    print(f"  QC GECEN              : {len(gecen)} ({len(gecen)/len(filtrelenmis)*100:.1f}%)")
    print(f"  QC KALAN (elenen)     : {len(kalan)} ({len(kalan)/len(filtrelenmis)*100:.1f}%)")

    # Eleme nedenleri
    duz_eleme = (filtrelenmis["qc_flat_channels"] >= config.NUM_LEADS).sum()
    clip_eleme = filtrelenmis["qc_clipping"].sum()
    uzunluk_eleme = (~filtrelenmis["qc_length_ok"]).sum()
    elektrot_eleme = filtrelenmis["qc_electrode_problem"].sum()

    print(f"\n  Eleme Nedenleri (cakisabilir):")
    print(f"    Tum kanallar duz (flat-line)  : {duz_eleme}")
    print(f"    Asiri genlik (clipping)       : {clip_eleme}")
    print(f"    Uzunluk uyumsuzlugu           : {uzunluk_eleme}")
    print(f"    Elektrot problemi (metadata)  : {elektrot_eleme}")

    # Duz kanal dagilimi
    duz_dagilim = filtrelenmis["qc_flat_channels"].value_counts().sort_index()
    print(f"\n  Duz Kanal Sayisi Dagilimi:")
    for duz_sayi, kayit_sayi in duz_dagilim.items():
        durum = " [ELENECEK]" if duz_sayi >= config.NUM_LEADS else ""
        print(f"    {duz_sayi:2d} duz kanal : {kayit_sayi:6d} kayit{durum}")

    # QC gecen kayitlarin sinif dagilimi
    if len(gecen) > 0:
        sinif_dag = gecen["label"].value_counts().sort_index()
        print(f"\n  QC Gecen Kayitlarda Sinif Dagilimi:")
        for sinif_idx, sayi in sinif_dag.items():
            sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
            oran = sayi / sinif_dag.sum() * 100
            print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 3 tamamlandi. Sonraki adim: adim04_segmentasyon.py")
    print("=" * 70)

    return filtrelenmis


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = kalite_kontrol_pipeline()


SEGMENTASYON


"""
adim04_segmentasyon.py — BirunAI EKG Siniflandirma: Adim 4 – Segmentasyon
==========================================================================

Bu modul, filtrelenmis EKG sinyallerini sabit uzunlukta pencerelere boler.

Projemizde belirttigimiz gibi:
    - 10 saniye sabit pencere: Klinik 12-lead EKG standardi ile uyumlu.
    - 250 Hz x 10 sn = 2500 zaman adimi.
    - PTB-XL kayitlari zaten 10 sn oldugundan, cogu kayit direkt kullanilir.
    - Kisa kayitlar: Sifir-padding (zero-pad) uygulanir.
    - Uzun kayitlar: Ortadan kirpilir (center-crop).

Ciktilar:
    - outputs/processed_data/segmented_signals/  (her kayit icin .npy dosyasi)
    - outputs/processed_data/segmented_manifest.csv

Kullanim:
    python adim04_segmentasyon.py
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# SEGMENTASYON FONKSIYONLARI
# =============================================================================

def sabit_pencere_uygula(sinyal_2d, hedef_uzunluk=None):
    """
    Sinyali sabit uzunlukta pencereye uyarlar.

    Projemizde belirttigimiz gibi:
    - Hedef uzunluk: 2500 ornek (10 sn @ 250 Hz)
    - Kisa sinyaller: Sona sifir-padding eklenir.
    - Uzun sinyaller: Bastan ve sondan esit olarak kirpilir (center-crop).
    - Tam uzunlukta sinyaller: Olduklari gibi korunur.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        hedef_uzunluk: Hedef zaman adimi sayisi. Varsayilan: config.TARGET_LENGTH

    Returns:
        numpy array: (kanal_sayisi, hedef_uzunluk) formatinda numpy array
    """
    if hedef_uzunluk is None:
        hedef_uzunluk = config.TARGET_LENGTH

    kanal_sayisi = sinyal_2d.shape[0]
    mevcut_uzunluk = sinyal_2d.shape[1]

    if mevcut_uzunluk == hedef_uzunluk:
        # Tam uyum — olduklari gibi don
        return sinyal_2d

    elif mevcut_uzunluk < hedef_uzunluk:
        # KISA sinyal — sifir-padding
        padded = np.zeros((kanal_sayisi, hedef_uzunluk), dtype=sinyal_2d.dtype)
        padded[:, :mevcut_uzunluk] = sinyal_2d
        return padded

    else:
        # UZUN sinyal — center-crop
        baslangic = (mevcut_uzunluk - hedef_uzunluk) // 2
        return sinyal_2d[:, baslangic:baslangic + hedef_uzunluk]


# =============================================================================
# ANA SEGMENTASYON PIPELINE'I
# =============================================================================

def segmentasyon_pipeline():
    """
    QC'den gecen tum sinyallere segmentasyon uygular.

    Islem Akisi:
        1. quality_manifest.csv okunur.
        2. Sadece qc_pass=True olan kayitlar secilir.
        3. Her kayit icin sabit pencere uygulanir.
        4. Segmente edilmis sinyaller .npy olarak kaydedilir.
        5. segmented_manifest.csv kaydedilir.

    Returns:
        pd.DataFrame: Segmente edilmis manifest.
    """
    print("=" * 70)
    print("BirunAI -- Adim 4: Segmentasyon (10sn Sabit Pencere)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "quality_manifest.csv")
    print(f"\n[1/3] quality_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"quality_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim03_kalite_kontrol.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    gecerli = df[df["qc_pass"] == True].copy()
    print(f"      QC gecen kayit: {len(gecerli)}")

    # --- 2. Cikti dizinini olustur ---
    kaynak_dizini = os.path.join(config.PROCESSED_DATA_DIR, "filtered_signals")
    cikti_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    os.makedirs(cikti_dizini, exist_ok=True)

    # --- 3. Segmentasyon dongusu ---
    print(f"\n[2/3] Segmentasyon uygulanyor...")
    print(f"      Hedef uzunluk : {config.TARGET_LENGTH} ornek")
    print(f"      = {config.WINDOW_SEC} saniye @ {config.TARGET_FS} Hz")

    basarili = 0
    padded_sayisi = 0
    cropped_sayisi = 0
    tam_sayisi = 0
    hatali = 0

    for ecg_id, satir in tqdm(gecerli.iterrows(), total=len(gecerli),
                               desc="      Segmentasyon"):
        try:
            # Sinyal yukle
            sinyal_dosyasi = os.path.join(kaynak_dizini, f"{ecg_id}.npy")
            sinyal = np.load(sinyal_dosyasi)

            orijinal_uzunluk = sinyal.shape[1]

            # Sabit pencere uygula
            segmente = sabit_pencere_uygula(sinyal)

            # Istatistik
            if orijinal_uzunluk == config.TARGET_LENGTH:
                tam_sayisi += 1
            elif orijinal_uzunluk < config.TARGET_LENGTH:
                padded_sayisi += 1
            else:
                cropped_sayisi += 1

            # Kaydet
            cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
            np.save(cikti_dosyasi, segmente.astype(np.float32))

            basarili += 1

        except Exception as e:
            hatali += 1

    print(f"\n      Basarili     : {basarili}")
    print(f"      Hatali       : {hatali}")
    print(f"      Tam uzunluk  : {tam_sayisi}")
    print(f"      Padding      : {padded_sayisi}")
    print(f"      Cropping     : {cropped_sayisi}")

    # --- 4. Segmente manifest ---
    print(f"\n[3/3] segmented_manifest.csv kaydediliyor...")

    # Basarili kayitlari isaretmek
    basarili_ids = set()
    for ecg_id in gecerli.index:
        cikti_dosyasi = os.path.join(cikti_dizini, f"{ecg_id}.npy")
        if os.path.exists(cikti_dosyasi):
            basarili_ids.add(ecg_id)

    gecerli["segmented"] = gecerli.index.isin(basarili_ids)
    gecerli["segmented_path"] = gecerli.index.map(
        lambda x: os.path.join("segmented_signals", f"{x}.npy") if x in basarili_ids else None
    )

    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    gecerli.to_csv(cikti_yolu)
    print(f"      Kaydedildi: {cikti_yolu}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    segmente_kayitlar = gecerli[gecerli["segmented"] == True]
    print(f"  Toplam segmente edilen : {len(segmente_kayitlar)}")
    print(f"  Sinyal boyutu          : ({config.NUM_LEADS}, {config.TARGET_LENGTH})")

    # Sinif dagilimi
    sinif_dag = segmente_kayitlar["label"].value_counts().sort_index()
    print(f"\n  Sinif Dagilimi:")
    for sinif_idx, sayi in sinif_dag.items():
        sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "Bilinmeyen")
        oran = sayi / sinif_dag.sum() * 100
        print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    # Ornek sinyal dogrulama
    ornek_dosya = os.path.join(cikti_dizini, f"{segmente_kayitlar.index[0]}.npy")
    if os.path.exists(ornek_dosya):
        ornek = np.load(ornek_dosya)
        print(f"\n  Ornek sinyal shape  : {ornek.shape}")
        print(f"  Ornek sinyal dtype  : {ornek.dtype}")

    # Disk kullanimi
    toplam_boyut_mb = 0
    for f in os.listdir(cikti_dizini):
        if f.endswith(".npy"):
            toplam_boyut_mb += os.path.getsize(os.path.join(cikti_dizini, f))
    toplam_boyut_mb /= (1024 * 1024)
    print(f"\n  Toplam disk kullanimi  : {toplam_boyut_mb:.1f} MB")

    print("\n" + "=" * 70)
    print("Adim 4 tamamlandi. Sonraki adim: adim05_ozellik_cikarma.py")
    print("=" * 70)

    return gecerli


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = segmentasyon_pipeline()



ÖZELLİK ÇIKARMA

"""
adim05_ozellik_cikarma.py — BirunAI EKG Siniflandirma: Adim 5 – Ozellik Cikarma
==================================================================================

Bu modul, 1D-CNN + BiLSTM modelimiz icin ek el-yapimi (handcrafted) ozellikler cikarir.
Ana modelin CNN katmanlari otomatik ozellik ogrenir, ancak ek ozellikler
istegebagli olarak model performansini artirabilir.

NOT: Projemizde birincil yaklasim "end-to-end" ogrenimdir.
     Ham sinyal dogrudan modele verilir, CNN kendi ozelliklerini ogenir.
     Bu modul, EDA (Exploratory Data Analysis) ve potansiyel
     feature-augmented model denemeleri icindir.

Cikarilan Ozellikler:
    - Istatistiksel: min, max, mean, std, skewness, kurtosis (her derivasyon)
    - Morfolojik: R-pike algılama (basit esik), RR aralik istatistikleri
    - Frekans alani: Baskın frekans, guc yoğunlugu bantları

Ciktilar:
    - outputs/processed_data/features.csv
    - outputs/processed_data/feature_stats.csv

Kullanim:
    python adim05_ozellik_cikarma.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import find_peaks, welch
from tqdm import tqdm

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# OZELLIK CIKARMA FONKSIYONLARI
# =============================================================================

def istatistiksel_ozellikler(sinyal_2d):
    """
    Her derivasyondan temel istatistiksel ozellikler cikarir.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array

    Returns:
        dict: {ozellik_adi: deger} formati
    """
    ozellikler = {}
    for ch in range(sinyal_2d.shape[0]):
        kanal = sinyal_2d[ch]
        prefix = f"ch{ch}"
        ozellikler[f"{prefix}_mean"] = np.mean(kanal)
        ozellikler[f"{prefix}_std"] = np.std(kanal)
        ozellikler[f"{prefix}_min"] = np.min(kanal)
        ozellikler[f"{prefix}_max"] = np.max(kanal)
        ozellikler[f"{prefix}_skew"] = float(sp_stats.skew(kanal))
        ozellikler[f"{prefix}_kurt"] = float(sp_stats.kurtosis(kanal))
        ozellikler[f"{prefix}_ptp"] = np.ptp(kanal)  # peak-to-peak
        ozellikler[f"{prefix}_rms"] = np.sqrt(np.mean(kanal ** 2))
        # Sifir gecis orani (zero-crossing rate)
        sign_changes = np.diff(np.sign(kanal))
        ozellikler[f"{prefix}_zcr"] = np.sum(sign_changes != 0) / len(kanal)
    return ozellikler


def morfolojik_ozellikler(sinyal_2d, fs=None):
    """
    Lead II (indeks 1) uzerinden R-pike tespiti ve RR araliklarini hesaplar.

    Lead II secilme nedeni: Klinik EKG'de P dalgasi ve QRS kompleksi
    Lead II'de en belirgin gorulur.

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        fs: Ornekleme frekansi. Varsayilan: config.TARGET_FS

    Returns:
        dict: RR-araligi istatistikleri
    """
    if fs is None:
        fs = config.TARGET_FS

    ozellikler = {}

    # Lead II (indeks 1) kullan, yoksa Lead I (indeks 0)
    lead_idx = 1 if sinyal_2d.shape[0] > 1 else 0
    lead = sinyal_2d[lead_idx]

    # R-pike tespiti (basit esik tabanlı)
    # Minimum mesafe: 0.3sn (200 bpm ustu olmaz), minimum yukseklik: 0.5 std
    min_mesafe = int(0.3 * fs)
    min_yukseklik = 0.5 * np.std(lead)

    peaks, properties = find_peaks(lead, distance=min_mesafe, height=min_yukseklik)

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs  # saniye cinsinden
        ozellikler["rr_mean"] = np.mean(rr_intervals)
        ozellikler["rr_std"] = np.std(rr_intervals)
        ozellikler["rr_min"] = np.min(rr_intervals)
        ozellikler["rr_max"] = np.max(rr_intervals)
        ozellikler["rr_range"] = np.ptp(rr_intervals)
        ozellikler["heart_rate_bpm"] = 60.0 / np.mean(rr_intervals)
        ozellikler["num_peaks"] = len(peaks)
        # HRV (Heart Rate Variability) - basit olcum
        ozellikler["rmssd"] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    else:
        ozellikler["rr_mean"] = 0
        ozellikler["rr_std"] = 0
        ozellikler["rr_min"] = 0
        ozellikler["rr_max"] = 0
        ozellikler["rr_range"] = 0
        ozellikler["heart_rate_bpm"] = 0
        ozellikler["num_peaks"] = len(peaks)
        ozellikler["rmssd"] = 0

    return ozellikler


def frekans_alani_ozellikleri(sinyal_2d, fs=None):
    """
    Frekans alaninda guc yogunlugu bantlarini hesaplar.

    Bantlar:
        - VLF (Very Low Frequency): 0.003 - 0.04 Hz
        - LF  (Low Frequency):      0.04 - 0.15 Hz
        - HF  (High Frequency):     0.15 - 0.4 Hz
        - QRS bandi:                 5 - 15 Hz
        - Toplam guc

    Args:
        sinyal_2d: (kanal_sayisi, zaman_adimi) formatinda numpy array
        fs: Ornekleme frekansi

    Returns:
        dict: Frekans alani ozellikleri
    """
    if fs is None:
        fs = config.TARGET_FS

    ozellikler = {}

    # Lead II uzerinden
    lead_idx = 1 if sinyal_2d.shape[0] > 1 else 0
    lead = sinyal_2d[lead_idx]

    # Welch PSD
    freqs, psd = welch(lead, fs=fs, nperseg=min(256, len(lead)))

    # Toplam guc
    ozellikler["total_power"] = np.sum(psd)

    # Bant gucleri
    def bant_gucu(f_low, f_high):
        mask = (freqs >= f_low) & (freqs <= f_high)
        return np.sum(psd[mask]) if np.any(mask) else 0

    ozellikler["power_vlf"] = bant_gucu(0.003, 0.04)
    ozellikler["power_lf"] = bant_gucu(0.04, 0.15)
    ozellikler["power_hf"] = bant_gucu(0.15, 0.4)
    ozellikler["power_qrs"] = bant_gucu(5.0, 15.0)

    # Baskin frekans
    baskin_idx = np.argmax(psd)
    ozellikler["dominant_freq"] = freqs[baskin_idx]

    # LF/HF orani (otonom sinir sistemi gostergesi)
    if ozellikler["power_hf"] > 1e-10:
        ozellikler["lf_hf_ratio"] = ozellikler["power_lf"] / ozellikler["power_hf"]
    else:
        ozellikler["lf_hf_ratio"] = 0

    return ozellikler


# =============================================================================
# ANA OZELLIK CIKARMA PIPELINE'I
# =============================================================================

def ozellik_cikarma_pipeline():
    """
    Tum segmente edilmis sinyallerden ozellik cikarir.

    Returns:
        pd.DataFrame: Ozellik matrisi
    """
    print("=" * 70)
    print("BirunAI -- Adim 5: Ozellik Cikarma (Feature Engineering)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    print(f"\n[1/3] segmented_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"segmented_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim04_segmentasyon.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    segmente = df[df["segmented"] == True].copy()
    print(f"      Segmente kayit: {len(segmente)}")

    # --- 2. Ozellik cikarma ---
    print(f"\n[2/3] Ozellikler cikariliyor...")

    sinyal_dizini = os.path.join(config.PROCESSED_DATA_DIR, "segmented_signals")
    tum_ozellikler = []

    for ecg_id, satir in tqdm(segmente.iterrows(), total=len(segmente),
                               desc="      Ozellik Cikarma"):
        try:
            sinyal = np.load(os.path.join(sinyal_dizini, f"{ecg_id}.npy"))

            # 3 ozellik grubu
            oz = {"ecg_id": ecg_id, "label": satir["label"]}
            oz.update(istatistiksel_ozellikler(sinyal))
            oz.update(morfolojik_ozellikler(sinyal))
            oz.update(frekans_alani_ozellikleri(sinyal))

            tum_ozellikler.append(oz)

        except Exception as e:
            pass

    # DataFrame olustur
    ozellik_df = pd.DataFrame(tum_ozellikler)
    ozellik_df.set_index("ecg_id", inplace=True)

    print(f"\n      Cikarilan ozellik sayisi: {len(ozellik_df.columns) - 1}")
    print(f"      Kayit sayisi: {len(ozellik_df)}")

    # --- 3. Kaydet ---
    print(f"\n[3/3] Kaydediliyor...")

    # Tam ozellik matrisi
    cikti_yolu = os.path.join(config.PROCESSED_DATA_DIR, "features.csv")
    ozellik_df.to_csv(cikti_yolu)
    print(f"      features.csv: {cikti_yolu}")

    # Ozellik istatistikleri
    sayisal_sutunlar = ozellik_df.select_dtypes(include=[np.number]).columns.drop("label", errors="ignore")
    stats_df = ozellik_df[sayisal_sutunlar].describe().T
    stats_cikti = os.path.join(config.PROCESSED_DATA_DIR, "feature_stats.csv")
    stats_df.to_csv(stats_cikti)
    print(f"      feature_stats.csv: {stats_cikti}")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)
    print(f"  Toplam kayit      : {len(ozellik_df)}")
    print(f"  Toplam ozellik    : {len(sayisal_sutunlar)}")
    print(f"  Ozellik gruplari  :")
    print(f"    - Istatistiksel : 12 kanal x 9 ozellik = 108")
    print(f"    - Morfolojik    : 8 ozellik (RR intervalleri, kalp hizi)")
    print(f"    - Frekans alani : 7 ozellik (guc bantlari, baskin frekans)")

    # Sinifa gore bazi ozellikler
    if "heart_rate_bpm" in ozellik_df.columns:
        print(f"\n  Sinifa Gore Ortalama Kalp Hizi (BPM):")
        for sinif_idx in sorted(ozellik_df["label"].unique()):
            sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "?")
            ortalama_hr = ozellik_df[ozellik_df["label"] == sinif_idx]["heart_rate_bpm"].mean()
            print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {ortalama_hr:.1f} BPM")

    print("\n" + "=" * 70)
    print("Adim 5 tamamlandi. Sonraki adim: adim06_veri_bolme.py")
    print("=" * 70)

    return ozellik_df


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    sonuc = ozellik_cikarma_pipeline()


veri bölme

"""
adim06_veri_bolme.py — BirunAI EKG Siniflandirma: Adim 6 – Veri Bolme
======================================================================

Bu modul, segmente edilmis veriyi hasta bazli train/val/test setlerine boler.

Projemizde belirttigimiz kritik tasarim karari:
    - HASTA BAZLI BOLME: Ayni hastanin farkli kayitlari ASLA farkli
      setlerde yer almaz. Bu, veri sizintisini (data leakage) onler.
    - PTB-XL'in kendi strat_fold sutunu kullanilir (1-10 arasi fold).
      -> Fold 9-10: Test seti (sabit, degismez)
      -> Fold 1-8: Train + Validation (kendi icinde %87.5-%12.5 bolunur)

Veri Sizintisi Onleme:
    Eger ayni hasta birden fazla kayda sahipse ve bu kayitlar
    train/test'e dagitilirsa, model hastanin bireysel ozelliklerini
    ogenir (generalizasyon degil, ezberleme). Bu nedenle bolme
    HASTA bazli yapilir.

Sinif Dengesizligi Ele Alma:
    - WeightedRandomSampler icin sinif agirliklari hesaplanir.
    - Agirliklar: 1 / sinif_sayisi (inverse frequency)
    - Adim 8'de Focal Loss ile birlikte kullanilacak.

Ciktilar:
    - outputs/processed_data/train_manifest.csv
    - outputs/processed_data/val_manifest.csv
    - outputs/processed_data/test_manifest.csv
    - outputs/processed_data/class_weights.npy

Kullanim:
    python adim06_veri_bolme.py
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

# Proje kok dizinini Python path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# =============================================================================
# VERI BOLME FONKSIYONLARI
# =============================================================================

def ptbxl_strat_fold_bolme(df, test_folds=(9, 10), val_ratio=0.125):
    """
    PTB-XL'in strat_fold sutununu kullanarak hasta bazli bolme yapar.

    PTB-XL veri seti, kendi icinde stratified fold yapisina sahiptir.
    Bu foldlar hasta bazli olusturulmustur, yani ayni hastanin
    tum kayitlari ayni fold'dadır.

    Args:
        df: Segmente edilmis manifest DataFrame (ecg_id index).
        test_folds: Test seti icin kullanilacak fold numaralari.
        val_ratio: Train setinden validation'a ayrilacak oran.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Test seti: fold 9 ve 10
    test_mask = df["strat_fold"].isin(test_folds)
    test_df = df[test_mask].copy()
    train_val_df = df[~test_mask].copy()

    # Train/Val bolmesi: Hasta bazli
    # Kalan foldlardan (1-8), son %12.5'i validation olarak ayir
    # Fold 8 -> validation, Fold 1-7 -> train
    val_fold = max(f for f in train_val_df["strat_fold"].unique() if f not in test_folds)
    val_mask = train_val_df["strat_fold"] == val_fold
    val_df = train_val_df[val_mask].copy()
    train_df = train_val_df[~val_mask].copy()

    return train_df, val_df, test_df


def sinif_agirliklari_hesapla(labels):
    """
    Sinif dengesizligini ele almak icin sinif agirliklarini hesaplar.

    Agirlik formulu: w_i = toplam_ornek / (sinif_sayisi * sinif_i_ornek)
    Bu formul, sklearn'in class_weight='balanced' yaklasimi ile aynidir.

    Args:
        labels: Etiket dizisi (numpy array veya list)

    Returns:
        numpy array: Her sinif icin agirlik (sinif indeksine gore sirali)
    """
    labels = np.array(labels, dtype=int)
    siniflar = np.unique(labels)
    toplam = len(labels)
    n_sinif = len(siniflar)

    agirliklar = np.zeros(config.NUM_CLASSES)
    for sinif in siniflar:
        sinif_sayisi = np.sum(labels == sinif)
        agirliklar[int(sinif)] = toplam / (n_sinif * sinif_sayisi)

    return agirliklar


# =============================================================================
# ANA VERI BOLME PIPELINE'I
# =============================================================================

def veri_bolme_pipeline():
    """
    Segmente edilmis veriyi train/val/test setlerine boler.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("=" * 70)
    print("BirunAI -- Adim 6: Veri Bolme (Hasta Bazli)")
    print("=" * 70)

    # --- 1. Manifest oku ---
    manifest_yolu = os.path.join(config.PROCESSED_DATA_DIR, "segmented_manifest.csv")
    print(f"\n[1/4] segmented_manifest.csv okunuyor...")

    if not os.path.exists(manifest_yolu):
        raise FileNotFoundError(
            f"segmented_manifest.csv bulunamadi: {manifest_yolu}\n"
            "Once adim04_segmentasyon.py calistirilmali."
        )

    df = pd.read_csv(manifest_yolu, index_col="ecg_id")
    segmente = df[df["segmented"] == True].copy()
    print(f"      Segmente kayit: {len(segmente)}")

    # strat_fold kontrolu
    if "strat_fold" not in segmente.columns:
        raise ValueError("strat_fold sutunu bulunamadi! PTB-XL metadata eksik.")

    print(f"      Fold dagilimi:")
    fold_dag = segmente["strat_fold"].value_counts().sort_index()
    for fold, sayi in fold_dag.items():
        print(f"        Fold {int(fold)}: {sayi} kayit")

    # --- 2. Bolme ---
    print(f"\n[2/4] Hasta bazli bolme uygulanyor...")
    print(f"      Test folds  : 9, 10")
    print(f"      Val fold    : 8")
    print(f"      Train folds : 1-7")

    train_df, val_df, test_df = ptbxl_strat_fold_bolme(segmente)

    print(f"\n      Train : {len(train_df)} kayit ({len(train_df)/len(segmente)*100:.1f}%)")
    print(f"      Val   : {len(val_df)} kayit ({len(val_df)/len(segmente)*100:.1f}%)")
    print(f"      Test  : {len(test_df)} kayit ({len(test_df)/len(segmente)*100:.1f}%)")

    # --- 3. Veri sizintisi kontrolu ---
    print(f"\n[3/4] Veri sizintisi kontrolu...")

    train_hastalar = set(train_df["patient_id"].unique())
    val_hastalar = set(val_df["patient_id"].unique())
    test_hastalar = set(test_df["patient_id"].unique())

    train_val_overlap = train_hastalar & val_hastalar
    train_test_overlap = train_hastalar & test_hastalar
    val_test_overlap = val_hastalar & test_hastalar

    print(f"      Train-Val hasta cakismasi  : {len(train_val_overlap)}")
    print(f"      Train-Test hasta cakismasi : {len(train_test_overlap)}")
    print(f"      Val-Test hasta cakismasi   : {len(val_test_overlap)}")

    if len(train_test_overlap) == 0 and len(train_val_overlap) == 0:
        print(f"      [OK] Veri sizintisi YOK!")
    else:
        print(f"      [UYARI] Veri sizintisi tespit edildi!")

    # --- 4. Sinif agirliklari ---
    print(f"\n[4/4] Sinif agirliklari hesaplaniyor ve kaydediliyor...")

    train_labels = train_df["label"].values
    agirliklar = sinif_agirliklari_hesapla(train_labels)

    print(f"      Sinif agirliklari:")
    for sinif_idx in range(config.NUM_CLASSES):
        sinif_adi = config.LABEL_NAMES.get(sinif_idx, "?")
        print(f"        [{sinif_idx}] {sinif_adi:20s}: {agirliklar[sinif_idx]:.4f}")

    # Kaydet
    agirlik_yolu = os.path.join(config.PROCESSED_DATA_DIR, "class_weights.npy")
    np.save(agirlik_yolu, agirliklar)
    print(f"      Kaydedildi: {agirlik_yolu}")

    # Manifestleri kaydet
    train_yolu = os.path.join(config.PROCESSED_DATA_DIR, "train_manifest.csv")
    val_yolu = os.path.join(config.PROCESSED_DATA_DIR, "val_manifest.csv")
    test_yolu = os.path.join(config.PROCESSED_DATA_DIR, "test_manifest.csv")

    train_df.to_csv(train_yolu)
    val_df.to_csv(val_yolu)
    test_df.to_csv(test_yolu)

    print(f"      train_manifest.csv: {len(train_df)} kayit")
    print(f"      val_manifest.csv  : {len(val_df)} kayit")
    print(f"      test_manifest.csv : {len(test_df)} kayit")

    # --- Ozet ---
    print("\n" + "=" * 70)
    print("OZET ISTATISTIKLER")
    print("=" * 70)

    for isim, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        sinif_dag = subset["label"].value_counts().sort_index()
        print(f"\n  {isim} Seti Sinif Dagilimi ({len(subset)} kayit):")
        for sinif_idx, sayi in sinif_dag.items():
            sinif_adi = config.LABEL_NAMES.get(int(sinif_idx), "?")
            oran = sayi / sinif_dag.sum() * 100
            print(f"    [{int(sinif_idx)}] {sinif_adi:20s}: {sayi:6d} ({oran:5.1f}%)")

    print("\n" + "=" * 70)
    print("Adim 6 tamamlandi. Sonraki adim: adim07_model_mimarisi.py")
    print("=" * 70)

    return train_df, val_df, test_df


# =============================================================================
# ANA CALISTIRMA
# =============================================================================

if __name__ == "__main__":
    train, val, test = veri_bolme_pipeline()
