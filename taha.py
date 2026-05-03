import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os

# 1. Veri yolunu belirle (VS Code Explorer'daki klasör isminle birebir aynı olmalı)
# 'a.py' ile aynı klasörde olduğu için sadece klasör adını yazman yeterli.
base_path = r'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

# 2. Veritabanını yükle
try:
    # os.path.join kullanarak dosya yolunu oluşturuyoruz
    csv_path = os.path.join(base_path, 'ptbxl_database.csv')
    df = pd.read_csv(csv_path, index_col='ecg_id')
    
    # Toplam kayıt sayısı
    x = len(df)
    print(f"Sistemde toplam {x} adet hasta kaydı bulundu.")
    
    # 3. Kullanıcıdan giriş al
    secim = input(f"Lütfen görüntülemek istediğiniz hastanın indeksini girin (0 - {x-1} arası): ")
    
    try:
        secim_idx = int(secim)
        
        if 0 <= secim_idx < x:
            # Seçilen satırdaki dosya yolunu al
            file_path = df.iloc[secim_idx]['filename_lr']
            
            # Sinyal dosyasının tam yolunu oluştur
            # rdsamp uzantı istemediği için olduğu gibi kullanıyoruz
            full_path = os.path.join(base_path, file_path)
            
            # 4. EKG Verisini Oku
            signal, fields = wfdb.rdsamp(full_path)
            
            # 5. Görselleştir
            plt.figure(figsize=(15, 5))
            plt.plot(signal[:, 0], color='blue', linewidth=1) # Lead I
            
            plt.title(f"Hasta Kaydı Sırası: {secim_idx} | Dosya: {file_path}")
            plt.xlabel("Zaman (Örneklem)")
            plt.ylabel("Genlik (mV)")
            plt.grid(True, alpha=0.3)
            plt.show()
            
            print(f"Hastanın Teşhis Kodları: {df.iloc[secim_idx]['scp_codes']}")
            
        else:
            print(f"Hata: Girdiğiniz sayı 0 ile {x-1} aralığında olmalıdır.")
            
    except ValueError:
        print("Hata: Lütfen geçerli bir sayı girin.")

except FileNotFoundError:
    print(f"Hata: Veritabanı dosyası bulunamadı!")
    print(f"Aranan yol: {os.path.abspath(csv_path)}")
    print("Lütfen veri seti klasörünün isminin doğruluğunu kontrol edin.")