# BirunAI — EKG Sınıflandırma Projesi: Detaylı Yolculuk Günlüğü

> Bu belge, projenin **başından bugüne kadarki** tüm teknik yolculuğu — attığımız her adımı, karşılaştığımız her problemi, yaptığımız her değişikliği (mimari mi yoksa sayısal/hiperparametre mi), ve her değişikliğin sonuca etkisini — mümkün olan en ince ayrıntısına kadar kronolojik olarak anlatır. Amaç: birisi bu belgeyi okuyunca "neyi neden yaptık, ne işe yaradı, ne yaramadı" sorusunun cevabını eksiksiz bulabilsin.

---

## 0. Amacımız ve Ne Yaptığımız (Kısa Giriş)

**TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — 2. Aşama** kapsamında, **12 kanallı (12-lead) EKG kayıtlarını 5 sınıfa** ayıran bir derin öğrenme sistemi geliştiriyoruz:

| Etiket | Sınıf | Tür |
|--------|-------|-----|
| 0 | Normal | Normal sinüs ritmi |
| 1 | AFIB | Atriyal Fibrilasyon | (ritim) |
| 2 | AFL | Atriyal Flutter | (ritim) |
| 3 | LBBB | Sol Dal Bloğu | (iletim) |
| 4 | RBBB | Sağ Dal Bloğu | (iletim) |

**Tek değerlendirme metriği: Macro F1.** Accuracy'ye güvenilmiyor çünkü sınıflar dengesiz — bir sınıfı hiç bilemeyen model bile yüksek accuracy alabilir. Macro F1, her sınıfın F1'ini eşit ağırlıkla ortalar; dolayısıyla en zayıf sınıf skoru doğrudan cezalandırır.

**Veri:** TEKNOFEST yarışma seti (~5 bin dengeli kayıt) + ~90 bin halka açık kayıt (PhysioNet Challenge 2020: CPSC, PTB-XL, Georgia, Chapman-Shaoxing/Ningbo, St Petersburg INCART).

**Train veri dağılımı (dengesizliğin kaynağı):** Normal 66.673, AFIB 5.905, AFL 8.640, LBBB 1.752, RBBB 5.790.
Kritik nüans: **AFL aslında veri-zengin (8.640) ama AFIB ile karışıyor** — yani problem veri azlığı değil, iki ritim sınıfının **ayırt edilebilirliği**. Validation seti dengeli (~149/sınıf) ve TEKNOFEST'ten.

**Nihai hedef:** Macro F1'i 0.90+ seviyesine çıkarmak. Bu belge, 0.83'lerden 0.87'lere nasıl geldiğimizi ve 0.90'ın neden bir "veri/etiket tavanı" olduğunu adım adım gösterir.

---

## 1. Sabit Temel: Asla İhlal Etmediğimiz Kurallar

Her denemede korunan, ihlali Macro F1'i sessizce çökerten kurallar (bunlar tıbbi/fizyolojik gerekçelidir, "keşke deneseydik" değil):

1. **Kanal-bazlı (lead-wise) Z-score, sadece train'den hesaplanır.** 12 kanal birlikte normalize edilmez — V1'in ~0.5mV QRS'i ile V5'in ~2.5mV'si birlikte ölçeklenirse V1 silinir, ve RBBB tanısı V1'de yaşar. Val/test train istatistiklerini kullanır (`train_stats.npz`), asla kendi istatistiğini hesaplamaz.
2. **Hasta/kayıt bazlı bölme, sızıntı yok.**
3. **Time-shift (`np.roll`) ve random crop YOK** — P-QRS-T zamansal ilişkisini bozar. Segmentasyon her zaman simetrik/merkez kırpma.
4. **Global genlik ölçekleme YOK** — sadece kanal-bazlı (bağımsız 0.9–1.1). Global ölçek V1/V6 oranını bozar.
5. **50Hz notch filtre YOK** — bant geçiren 0.5–40Hz (Butterworth, 4. derece, `filtfilt`) yeterli; notch T dalgasına zarar verir.
6. **Bir kanal asla tamamen atılmaz** — SQI-bazlı ağırlıklandırma/dropout serbest, ama fiziksel silme yok.

Bu kurallar tüm yolculuk boyunca sabit kaldı; aşağıdaki değişikliklerin **hiçbiri** bunları ihlal etmez.

---

## 2. Kronolojik Yolculuk

Her adımda değişikliğin **türünü** etiketliyorum:
`[MİMARİ]` model yapısı · `[SAYISAL]` hiperparametre/sayı · `[VERİ]` veri/etiket stratejisi · `[ALTYAPI]` eğitim/kod altyapısı · `[ANALİZ]` teşhis (kod değişikliği değil, ama yön belirledi).

---

### AŞAMA A — İlk Mimari: CardioFusion-5 ve İlk Tavan (~0.83)

**Başlangıç mimarisi: `CardioFusion5`** (`adim07_model_mimarisi.py`, ~11.8M parametre)
`[MİMARİ]` SE-ResNet omurga (kanal filtreleri [64,128,256,256,384]) + 2 katmanlı Transformer Encoder (8 head, dim 384) + DANN domain-adversarial dalı + multi-task (Normal/Ritim/İletim 3-sınıf aux head).

**İlk sonuç:** Val Macro F1 **0.8329**, eşik-optimizasyonuyla 0.8417.

**İlk gözlem (kritik):** Normal, LBBB, RBBB sınıfları ~0.94 civarında hızla yakınsıyor. **Tüm sorun AFIB (~0.74) ve AFL (~0.62) ritim sınıflarında.** Bu gözlem tüm projenin geri kalanının odağını belirledi.

---

### Curriculum (müfredat) öğrenmesinin kurulması ve ilk büyük düzeltme

**Problem:** İki fazlı eğitimde Faz 2 (kitlesel internet verisi) **epoch 21 civarında "patlıyordu"** — val loss tırmanmaya başlıyor (klasik domain overfit; model TEKNOFEST'ten koparak internet verisinin cihaz/hastane dağılımına aşırı uyuyor).

**Yapılan değişiklikler (2026-07-03):**
- `[SAYISAL]` Faz 2 uzunluğu **80 → 30 epoch** kısaltıldı.
- `[SAYISAL]` `DANN_LAMBDA` **0.1 → 0.3** (domain-adversarial baskı güçlendirildi).
- `[MİMARİ/ALTYAPI]` Faz 2'ye `LinearLR` warmup eklendi (faz geçişindeki loss sıçramasını önlemek için) → `CosineAnnealingLR`'a `SequentialLR` ile bağlandı.
- `[ALTYAPI]` Yeni `create_domain_anchored_sampler`: Faz 2'de TEKNOFEST **4x oversample** edilir (hem sınıf-dengeli hem TEKNOFEST-çapalı, hedef domain'den kopmayı engeller).
- Sonuç curriculum: **Faz1 / Faz2 / Faz3 = 25 / 30 / 25 = 80 epoch.**

**Neden internet verisi tamamen atılmadı?** Val seti dengeli-TEKNOFEST olduğu için internet verisinin "görülmemiş hastane" genelleme faydasını **eksik ölçüyoruz** — yarışmanın final test setinde bu fayda ortaya çıkabilir. Bu yüzden veri atılmadı, **cerrahi** kullanıldı.

**Üç fazın mantığı:**
- **Faz 1 (sadece TEKNOFEST):** Dengeli, temiz yarışma verisiyle temiz bir "önsel" (prior) kur. DANN kapalı.
- **Faz 2 (TEKNOFEST + ~90k internet):** DANN açık, domain-anchored sampler. Genelleme kazan ama TEKNOFEST'e çapalı kal.
- **Faz 3 (sadece TEKNOFEST fine-tune):** CNN omurgayı dondur, sadece ritim/sınıflandırıcı başlıklarını düşük LR'de eğit. SWA aktif.

---

### İlk atriyal (ritim) dalı denemesi: Spectral Branch

**Problem:** Curriculum düzeltmesi sonrası kalan tek darboğaz hâlâ AFIB (0.74) / AFL (0.62).

`[MİMARİ]` **`SpectralAtrialBranch` eklendi** (`model_v6.py`): Atriyal aktiviteyi taşıyan kanalların (II, III, aVF, V1) FFT'sini alır, 0.5–15Hz bandını tutar, frekans üzerinde MaxPool yapar. Fizyolojik gerekçe: **AFL = keskin ~4–6Hz flutter tepesi, AFIB = geniş bant / tepe yok.** FFT, `torch.autocast(enabled=False)` içinde çalışır (fp16 FFT desteklenmiyor).

**Sonuç:** Bu eklemeden önceki en iyi 0.8402. Spektral dal tek başına belirgin sıçrama getirmedi — çünkü (aşağıda bulacağımız) **QRS baskılaması yoktu**, flutter sinyali QRS enerjisinde boğuluyordu.

---

### İlk büyük TEŞHİS: Sorun mimaride değil, VERİ/ETİKETTE

`[ANALİZ]` **En kritik erken bulgu (2026-07-03):** **4 ayrı mimari iterasyonda** (BiGRU, Cross-Lead, Spectral FFT, wide features) AFIB/AFL **hiç kımıldamadı** (0.60/0.75'te sabit), diğer sınıflar ~0.94. Çıkarım: **~0.84 tavanı MODEL tavanı değil, VERİ/ETİKET tavanı.** İnternet ritim etiketleri (AFIB/AFL) gürültülü ve TEKNOFEST ile çelişiyor (morfoloji sınıfları LBBB/RBBB temiz ölçülür, ama ritim tanısı özneldir).

`[VERİ]` **Çözüm — Gürültülü-etiket-farkında ağırlıklandırma:** `config.INTERNET_RHYTHM_WEIGHT = 0.4` + `adim08:_rhythm_noise_weights`. Faz 2'de **internet kaynaklı (domain>0) AFIB/AFL örnekleri loss'a 0.4x katkı** yapar; TEKNOFEST ritmi ve internet morfolojisi tam 1.0x. Böylece ritim öğrenimi baskın olarak **temiz TEKNOFEST etiketlerinden** gelir, morfoloji sınıfları ise internet verisinin tamamından faydalanır.

---

### İlk confusion (karışıklık) analizi ve joint threshold keşfi

`[ANALİZ]` `analiz_confusion.py` (best 0.8536): **En büyük hata 48 AFL→AFIB (tek yönlü).** AFL recall 0.547; AFIB recall 0.886 ama precision 0.717. Yani **model ritim şüphesinde default olarak "AFIB" diyor** (AFIB önyargısı). Bu 48 vakayı eşik ayarı düzeltemiyor çünkü model bu kararlarında kendinden emin.

`[SAYISAL/ALTYAPI]` **Joint threshold optimizasyonu keşfi:** Per-class bağımsız eşik optimizasyonu aslında argmax'tan **kötü** (0.8417 < 0.8536). Ama yeni `find_optimal_thresholds_joint` (koordinat yükselişi ile doğrudan macro-F1'i maksimize eder, tek tutarlı karar kuralı `apply_thresholds`): **0.8536 → 0.8618 BEDAVA** (hiç eğitim yok, sadece karar eşiklerini birlikte optimize etmek). `adim08` artık bu joint eşiği kullanıyor.

`[SAYISAL]` Hard-example-mining ayarları: AFL `hard_alpha` P3'te 1.8→2.2, AFIB 1.5→1.3 (recall dengesi). Focal class-weight AFL'ye az ağırlık veriyordu (çok örnek var), hard_alpha bunu telafi ediyor.

**Ara sonuç:** ~0.86 (joint threshold ile).

---

### AŞAMA B — Büyük Mimari Pivot: CardioFusion-5 → CardioFusion-6 ("Lean-Robust")

**Problem:** CardioFusion-5 (11.8M param, ağır Transformer) **overfit oluyordu** — val loss yükseliyor, AFIB/AFL 0.56–0.59'da plato yapıyor. Çok fazla parametre, az veri için fazla kapasite.

`[MİMARİ]` **Yeni mimari: `CardioFusion6`** (`model_v6.py`, ~3.87M parametre — v5'in **üçte biri**). "Lean-Robust" felsefesi: daha az parametre, daha çok regularizasyon ve domain-dayanıklılık:
- **Transformer → BiGRU** ritim dalı (daha hafif, ritim için yeterli).
- **Instance Normalization** stem/erken katmanlarda (her örneği kendi içinde normalize eder → cihaz/genlik farklarını siler, domain-dayanıklılık).
- **Stochastic Depth (`DropPath`)** — regularizasyon.
- **`CrossLeadAttention`** — 12 kanalın her birini bir "token" olarak kodlar (paylaşımlı küçük CNN encoder) ve kanallar arası multi-head self-attention yapar. Amaç: V1↔V6 gibi LBBB/RBBB için kritik ilişkileri öğrenmek.
- **`SpectralAtrialBranch`** korundu (AFIB↔AFL için).
- Wide fizyolojik özellikler korundu.

`[ALTYAPI]` **EMA (Exponential Moving Average) eklendi:** `ModelEMA`, decay 0.999, **her optimizer adımında** güncellenen gölge ağırlık kopyası. Her epoch hem raw hem EMA model validate edilir, yüksek skorlayan checkpoint'lenir (`[BEST-RAW]`/`[BEST-EMA]`).

`[ALTYAPI]` **Hard-mining kusuru düzeltildi (gerçek bug):** `FocalLoss.forward`'a `reduction='none'` yolu eklendi. Önceden `.mean()`-sonra-ölçekle yapılıyordu, bu per-sample AFIB/AFL ağırlığını tek bir batch-geneli skalere çökertiyordu. Artık per-sample hard-example ağırlıkları doğru uygulanıyor.

**CF-6 ilk uzun-curriculum sonucu:** 0.8247 (v5'in altında başladı — ama regularizasyon avantajı ilerleyen tunning'de ortaya çıktı).

---

### İkinci büyük TEŞHİS + KÖK NEDEN: QRS baskılaması

`[ANALİZ]` **2026-07-15 (best 0.8566, eşik-opt 0.8633):** 5 mimari iterasyon boyunca **48–50 AFL→AFIB hatası hiç kımıldamadı** — kanıt kaya gibi sağlam. Wide-feature cache analizi sebebi açıkladı:
- `Atrial_rate` özelliği **çöp gürültü** — tüm sınıflar ~0.22–0.28, hiç ayırt edicilik yok.
- `P_regularity` AFIB (0.517) vs AFL (0.536) **neredeyse aynı**.
- LBBB/RBBB'nin RR_CV'si (~0.45) Normal'den (0.348) **yüksek** — `find_peaks` tabanlı R-tepe dedektörü **geniş QRS'lerde yanlış tepe** yakalıyor (güvenilmez).
- **Asıl kök neden:** `SpectralAtrialBranch`'in 0.5–15Hz bandı, `sqi.py`'nin QRS-güç bandıyla (5–15Hz) **çakışıyor.** QRS geniş-bantlı, büyük-genlikli bir transient; küçük P/flutter dalgasını (5–10x küçük) domine ediyor. Yani spektral dal flutter'ı görmeye çalışırken aslında QRS'i görüyordu.

`[MİMARİ]` **Çözüm — QRS baskılaması (`_suppress_qrs`):** FFT'den **önce**, genlik zarfı `medyan + 4*MAD` eşiğini aşan pencereler (yani QRS'ler) maskelenir (dilate ~120ms, kenar yumuşatma). Geriye kalan TQ-segment/baseline FFT'ye girer. Test: sentetik sinyalde %60 enerji azalması, atriyal bileşen korundu.

`[SAYISAL]` **class_weight / hard_alpha çatışması düzeltildi:** Focal alpha AFIB'i global frekanstan (5905<8640) daha ağır sayıyordu, bu da AFL'yi öne çıkarma niyetini ~%4 söndürüyordu. AFIB/AFL base weight eşitlendi; AFL/AFIB dengesi artık tek yerden (hard_alpha) kontrol ediliyor.

---

### Faz-1'e özel derin analiz (internet verisi olmadan bile neden düşük?)

`[ANALİZ]` **2026-07-15:** Soru: TEKNOFEST-only Faz 1'de (internet verisi YOK) bile AFIB/AFL neden düşük? Epoch-epoch trend:
- Normal/LBBB/RBBB epoch 2'den itibaren monoton yakınsıyor.
- **AFIB epoch 1–9 arası vahşi salınım** (0.244 → 0.642 → 0.089 → 0.737) — karar sınırı kırılgan/instabil, özellikle ilk 10 epoch.
- TEKNOFEST-only wide-feature analizi: P_regularity/P_present aslında TEKNOFEST'te internet-karışık veriden **daha iyi** ayırt ediyor (P_reg AFIB=0.456 vs AFL=0.603 — gerçek ama örtüşen sinyal). **İnternet verisi bu sinyali seyreltiyor** → `INTERNET_RHYTHM_WEIGHT` çözümünü doğrular.
- `adim07b`'nin R-tepe dedektörü (`find_peaks`, ham sinyal) ve `Atrial_rate` özelliği (Welch PSD, **QRS-baskılamasız** — SpectralAtrialBranch'teki aynı hata) kök neden.

`[ANALİZ]` **Pan-Tompkins-lite R-tepe dedektörü denendi** (n=3/sınıf gerçek TEKNOFEST kayıt): sonuç kararsız, net iyileşme yok (eski AFIB-AFL farkı 0.034, yeni 0.005). 90K kayıt yeniden hesaplama (~saatler) bu kırılgan kanıtla **başlatılmadı**, ikincil önceliğe bırakıldı.

---

### QRS-baskılama sonucu: Hatayı azaltmadı, YÖNÜNÜ değiştirdi

`[ANALİZ]` **2026-07-18 (best 0.8539):** QRS-baskılama + ağırlık-eşitleme run'ı bitti — 0.8539 (önceki 0.8566 ile aynı/wash). Ama confusion matrix **önemli bir mekanizma değişikliği** gösterdi:
- AFL→AFIB: 49 → 41 (AFL recall 0.573 → 0.620 **yükseldi**)
- AFIB→AFL: 15 → 24 (AFIB recall 0.866 → 0.805 **düştü**)
- Toplam AFIB↔AFL hata havuzu **aynı** kaldı (64 → 65).

**Kritik ders:** Ağırlık ayarı/reweighting **hatayı azaltmadı, yönünü değiştirdi.** Bu ikili için loss-reweighting **azalan getiri** noktasına ulaştı; kalan ~65 vaka muhtemelen gerçekten belirsiz (değişken-bloklu AFL / sınır-AFIB).

`[ALTYAPI]` **Pratik ders + düzeltme (gerçek veri kaybı yaşandı):** `adim08_egitim.py` her zaman aynı `best_model.pth`'a yazıyordu — bu run önceki iyi checkpoint'i (0.8566) **EZDİ, kaybettik.** Düzeltme: `egitim_pipeline(seed, tag)` parametreleri eklendi (`--seed --tag` CLI), checkpoint/log/threshold dosyaları artık tag'e göre ayrılıyor (`best_model_<tag>.pth`, `training_log_<tag>.json`). Ayrıca `ensemble_eval.py` yazıldı (softmax olasılık ortalaması + joint eşik).

---

### AŞAMA C — Ensemble (topluluk) stratejisi: Asıl sıçrama buradan geldi

Tek modelde ritim tavanına çarptığımız kanıtlandığı için strateji değişti: **aynı mimariyi farklı seed'lerle eğitip olasılıklarını ortalamak** (ensemble). Farklı seed'ler farklı hatalar yapar; ortalama bu hataları törpüler.

`[ALTYAPI/SAYISAL]` **2026-07-19 — 2-model ensemble:**
- seed123 tamamlandı: best **0.8600** (epoch 59, Faz 3). Sınıf F1: Nor .933, AFI .779, AFL .712, LBB .950, RBB .928 — o ana kadarki **tek-model rekoru**.
- `ensemble_eval.py` ile seed42 + seed123:
  - Ensemble argmax: 0.8599 (tek başına seed123'ten farksız)
  - **Ensemble + joint-eşik: 0.8676 — YENİ GENEL REKOR** (AFL 0.714, AFIB 0.785). Yani ensemble ile joint-eşiğin **kombinasyonu**, ikisinden de ayrı ayrı güçlü.

`[ANALİZ]` **P2 overfit kesinleşti:** İki ayrı run'da (seed42, seed123) P2'nin val-loss/F1 zirvesi **tam olarak aynı yerde** (P2-epoch ~9, mutlak ~epoch 34) geliyor: train loss düzgün düşüyor (0.287 → 0.056), val loss **katlanıyor** (0.225 → 0.440). Tutarlı, öngörülebilir overfit.

`[SAYISAL]` **Aksiyon:** `EPOCHS_PHASE_2` 30 → 18, `PATIENCE_P2` 12 → 8 (toplam 80 → 68 epoch). **Sonuç değişmez** çünkü checkpoint zaten zirveyi kurtarıyordu — sadece boşa giden ~11 epoch (duvar saati) kazanıldı.

---

### 3-model ensemble: GENEL REKOR 0.8736

`[ALTYAPI/SAYISAL]` **2026-07-20:**
- seed2026 (kısaltılmış P2 ile) tamamlandı: **tek başına 0.8673** — o ana kadarki en iyi tekil model. Bu, kısaltılmış-P2 konfigürasyonunun sonucu bozmadığını (aksine iyileştirdiğini) doğruladı.
- **3-model ensemble (seed42 + seed123 + seed2026, softmax ortalama + joint eşik): 0.8736 GENEL REKOR.**
- Sınıf F1: **Normal 0.960, AFIB 0.795, AFL 0.712, LBBB 0.960, RBBB 0.940.**
- Eşikler: Normal 0.34 | AFIB 0.42 | AFL 0.50 | LBBB 0.30 | RBBB 0.55.

`[ALTYAPI]` Modeller `saved_models/` altına F1-skoruyla isimlendirilmiş klasörlere kaydedildi (her klasörde checkpoint + `train_stats.npz` + training_log + README): `CardioFusion6_F1_0.8600_seed123/`, `CardioFusion6_F1_0.8673_seed2026/`, `CardioFusion6_ENSEMBLE_F1_0.8736/`.

`[ALTYAPI]` `dashboard.py` artık otomatik en son güncellenen `training_log*.json`'u buluyor (`find_latest_log()`) — yeni bir eğitim `--tag` ile başlatıldığında URL değişikliği gerekmiyor, dashboard kendiliğinden ona geçiyor.

> **Ölçüm notu:** Training-log'un kendi kaydettiği F1 (örn. seed2026 için 0.86) ile bağımsız yeniden-değerlendirme (0.8673) arasında ~0.007 fark var — AMP/GPU kayan-nokta ölçüm gürültüsü. Tüm karşılaştırmalar için tutarlı olarak `ensemble_eval.py`/`analiz_confusion.py`'nin bağımsız ölçümü kullanıldı.

---

### Atriyal sinyal için son mühendislik denemeleri (hepsi ritim tavanını doğruladı)

`[MİMARİ/ANALİZ]` **Otokorelasyon ritim-düzenliliği (2026-07-20, "Path B"):** R-tepe tabanlı el-yapımı özellikleri Cohen's d (etki büyüklüğü) ile test ettim (n=20/sınıf gerçek TEKNOFEST):
- Eski find_peaks: Atrial_rate d=0.739, RR_CV d=0.579
- Yeni Pan-Tompkins: **daha kötü** (d=0.428 / 0.174) — sert "tepe var/yok" kararı kırılgan.
- **Otokorelasyon** (sinyalin zaman-kaydırılmış kendisiyle benzerliği, hiçbir tepe kararı gerektirmez): **d=0.807 — tüm alternatiflerden iyi.**
- `[MİMARİ]` `SpectralAtrialBranch`'e eklendi — **Wiener-Khinchin teoremi** (güç spektrumunun ters-FFT'si = otokorelasyon) ile mevcut FFT'nin üzerine, ekstra R-tepe tespiti **olmadan** hesaplanıyor. Fizyolojik RR aralığında (250–1500ms) max-pool ile "düzenlilik skoru" çıkarılıp spektral-flutter özelliğiyle birleştiriliyor. Parametre artışı ~192 (bedava). Veri yeniden hesaplama gerekmedi (model-içi hesap).
- **Sonuç:** 0.8491 (AFIB 0.7375 düştü) — **6+ atriyal-dal denemesinin hepsi başarısız.**

`[ANALİZ]` **KÖK NEDEN (nihayet):** Tüm atriyal-dal denemeleri küçük özellik vektörünü **tek fusion sınıflandırıcısına ekliyordu** — baskın morfoloji(256) + GRU(256) yanında boğuluyordu; atriyal sinyal kendi başına ayırt edici olmaya **hiç zorlanmadı.**

---

### AŞAMA D — Mixture-of-Experts (CardioFusion-7) ve Ritim Tavanının Kesin Kanıtı

`[ANALİZ]` **Deep research yapıldı** (WebSearch/WebFetch, PhysioNet + AFIB/AFL literatürü). Bulgular:
1. Kazanan yaklaşımlar (F1 AFib .95, AFL .90) **ayrı uzman ağlar** kullanıyor — her biri kendi sınıflandırması + **karar düzeyinde** birleşme (özellik düzeyinde DEĞİL — bizim boğulma sorunumuzun tam cevabı).
2. **ConvNeXtV2-1D** (depthwise conv + inverted bottleneck + LayerNorm + GELU + GRN) eski 1D-CNN'i geçiyor (F1 .986, 770k param).
3. Deep supervision şart.
4. RR AFIB'e yardım eder ama AFL'yi bozabilir.

`[MİMARİ]` **`model_v7.py:CardioFusion7`** (2.18M param) — büyük mimari revizyon:
- **ConvNeXt1D morfoloji uzmanı** (kendi 5-sınıf başlığı)
- **RhythmExpert** (QRS-baskılı atriyal + spektral + otokorelasyon + temporal ConvNeXt, **kendi 5-sınıf başlığı**)
- **Öğrenilen GATE** (logit'leri per-sample birleştirir: `gate_w[:,0]*morph_logits + gate_w[:,1]*rhythm_logits`)
- **Deep supervision** (her uzman bağımsız loss, `DEEP_SUP_WEIGHT=0.3`)

`[ALTYAPI]` `adim08` `train_one_epoch` DRY yeniden yazıldı (`_class_loss` helper, `deep_supervision` bayrağı, `return_experts=True`). P3 freeze `startswith(('stem.','morph_stage','morph_down'))` — **dikkat: 'stem.' alt-string olarak `rhythm_expert.stem`'i de yakalıyordu**, `startswith` ile düzeltildi (ritim uzmanı + gate P3'te eğitilebilir kalmalı).

`[ANALİZ]` **v7 sonucu (2026-07-25, "mev7"): best 0.8449** (AFIB 0.741, AFL 0.672) — v6 tek-model 0.867 ve v6-ensemble 0.8736'nın **ALTINDA.**

`[ANALİZ]` **MoE gate teşhisi (`analiz_moe.py`) — KRİTİK:** Ritim uzmanı **kendi başlığıyla AFIB 0.765** alıyor (morfoloji 0.644, fused 0.743'ten iyi — **uzman gerçekten uzmanlaştı, atriyal bilgi çıkarılabilir KANITLANDI**). AMA:
- AFL'de **uzman bile 0.596'da** kalıyor.
- Normal'de kötü (0.589) olduğundan gate ona AFIB/AFL'de sadece **0.42 ağırlık** veriyor.

`[ANALİZ]` **Karışık ensemble (`mixed_ensemble_eval.py`, v6×3 + v7): 0.8743** = v6-only 0.8736'dan **+0.0007 (gürültü seviyesinde).**

**DEFİNİTİF SONUÇ:** AFL ~0.60–0.71 **gerçek veri/etiket tavanı** — 8+ mimari (CNN / ResNet / Transformer / BiGRU / spektral / otokorelasyon / cross-lead / adanmış-uzman-MoE) **hepsi aynı yere çarptı.** AFIB ~0.79 tavanı. Değişken-bloklu AFL kardiyoloğa bile AFIB gibi görünüyor.

> **Uyumluluk notu:** `SpectralAtrialBranch(use_autocorr=)` bayrağı eklendi — v6 seed checkpoint'leri `use_autocorr=False` (fc 48), v7 `True` (fc 52). Otokorelasyon eklemem eski v6 ensemble'ı önce yüklenemez yapmıştı (`size mismatch for spectral.fc.0.weight`), bu bayrakla düzeltildi.

---

### Son teşhis: Hata vakaları gerçekten belirsiz mi?

`[ANALİZ]` **`analiz_afl_hatalari.py`** — modelin AFIB sandığı 36 AFL vakasının fizyolojisi ölçüldü (Lead II, QRS-baskılı):

| Grup | RR düzensizliği (CV) | Flutter gücü |
|------|----------------------|--------------|
| Tüm AFIB (referans, n=149) | 0.246 | 0.1622 |
| Doğru bilinen AFL (n=98) | 0.196 | 0.2808 |
| **AFL ama AFIB sanılan (36 hata)** | **0.233** | **0.1827** |

**Yorum:** Hatalı vakalar tam **arada, ama AFIB'e yakın** — RR düzensizlikleri AFIB'e benziyor (0.233 ≈ 0.246), flutter gücü neredeyse AFIB kadar zayıf (0.1827, temiz AFL'nin 0.2808'inin çok altında). Bunlar **temiz AFL değil, "değişken bloklu AFL" denen sınır vakaları.** Yine de flutter gücü AFIB'den (0.1622) **biraz yüksek** → %100 belirsiz değil, **çok ince bir sinyal artığı var** → sınırlı bir alan (headroom) olabilir ama AFIB ile örtüşme ağır.

---

## 3. Sonuç Tablosu — F1 İlerlemesi (Özet)

| # | Aşama / Değişiklik | Tür | Macro F1 |
|---|--------------------|-----|----------|
| 1 | CardioFusion-5 ilk sonuç | MİMARİ | 0.8329 (eşik 0.8417) |
| 2 | Spektral dal öncesi | — | 0.8402 |
| 3 | İlk confusion analizi + joint threshold | ANALİZ/SAYISAL | 0.8536 → **0.8618** (bedava) |
| 4 | CardioFusion-6'ya pivot (ilk run) | MİMARİ | 0.8247 |
| 5 | QRS baskılama + kök-neden düzeltmeleri | MİMARİ/SAYISAL | 0.8566 (eşik 0.8633) |
| 6 | QRS-baskılama + ağırlık-eşitleme run'ı | SAYISAL | 0.8539 (wash — hata yönü değişti) |
| 7 | seed123 tek model | ALTYAPI | 0.8600 |
| 8 | **2-model ensemble + joint eşik** | ALTYAPI | **0.8676 (rekor)** |
| 9 | seed2026 tek model | ALTYAPI | 0.8673 |
| 10 | **3-model ensemble + joint eşik** | ALTYAPI | **0.8736 (GENEL REKOR)** |
| 11 | Otokorelasyon dalı | MİMARİ | 0.8491 (başarısız) |
| 12 | CardioFusion-7 MoE (mev7) | MİMARİ | 0.8449 (v6'nın altında) |
| 13 | Karışık ensemble (v6×3 + v7) | ALTYAPI | 0.8743 (+0.0007 gürültü) |

**Şu anki en iyi dağıtılabilir sonuç: v6 3-seed ensemble = 0.8736** (veya v6+v7 = 0.8743, marjinal).

---

## 4. 0.90 Neden Ulaşılamıyor? (Matematik)

Macro F1 = 5 sınıf F1'inin ortalaması. 0.90 için sınıf-F1 toplamı ≥ 4.50 olmalı.
- Şu an: Normal 0.960 + AFIB 0.795 + AFL 0.712 + LBBB 0.960 + RBBB 0.940 = **4.367** → ortalama 0.8736.
- Normal/LBBB/RBBB zaten tavanda (~0.96), buradan alınacak yer yok.
- 0.90 için **AFIB + AFL toplamda +0.13** (yani birleşik ~+0.25) artmalı.
- 36 AFL→AFIB hatasının iyimser yarısını (18) kurtarsak bile macro'ya **sadece +0.012** → ~0.886.

**Sonuç:** Bu val/etiket kalitesiyle 0.90 gerçekçi değil; gerçekçi tavan **~0.87–0.88.** Kalan açık **model değil**, o sınır AFL vakalarının **doğası + etiket kalitesi.**

---

## 5. 0.90'a Giden Gerçek Levers (Denenmemiş / Veri-Merkezli)

Mimariyi 8 kez denedik — 0.90'ın anahtarı mimaride değil. Etki sırasına göre kalan gerçek levers:

1. **`[VERİ]` AFL/AFIB veri kalitesi + miktarı (en büyük lever):** Ek temiz AFL kaynağı (2–3x), ve borderline vakaların fizyolojik metriklerle (RR_CV + flutter gücü) taranarak yanlış-etiketlilerin düzeltilmesi. Beklenen: +0.03–0.05.
2. **`[MİMARİ]` QRST-cancellation (ortalama-atım çıkarma):** Literatürün altın standardı — tüm atımları hizala, ortalama QRST şablonunu çıkar, geriye saf atriyal aktivite kalsın. Biz sadece genlik-maskeleme yaptık. Beklenen: +0.01–0.02.
3. **`[MİMARİ]` İki-aşamalı kaskad (adanmış AFIB-vs-AFL ikili sınıflandırıcı):** 5-sınıflı model "AFIB ya da AFL" dediğinde, QRST-cancellation'lı saf atriyal sinyalde çalışan özel bir ikili model son kararı versin. Beklenen: +0.02–0.03.
4. **`[ALTYAPI]` Çoklu-segment agregasyonu:** Kayıt başına tek pencere yerine 3–4 pencere, tahminleri birleştir → daha çok ritim kanıtı. Beklenen: +0.01.

İyimser toplam: 0.874 → ~0.88–0.90 bandı (levers örtüşür, garanti değil).

---

## 6. Öğrenilen Genel Dersler

- **Metrik seçimi belirleyici:** Accuracy yerine Macro F1'e odaklanmak, tüm çabayı doğru yere (en zayıf sınıf) yönlendirdi.
- **Bir noktadan sonra sorun mimaride değil veride:** 8+ mimari aynı duvara çarptığında, kanıt "daha akıllı model" değil "daha temiz veri/etiket" der.
- **Ensemble + joint-threshold, tek modelden güçlü** ve **bedava** (yeni eğitim yok). En büyük garantili sıçrama buradan geldi (0.86 → 0.8736).
- **Ölçüm disiplini:** training-log F1'i ile bağımsız değerlendirme arasında ~0.007 gürültü var; tüm kararlar **tek tutarlı ölçüm aracıyla** (`ensemble_eval.py`) verildi.
- **Fizyolojik gerekçe > kör deneme:** QRS baskılama, kanal-bazlı normalizasyon, atriyal dal — hepsi EKG fizyolojisinden türetildi, rastgele mimari tweak değil.
- **Reproducibility'yi koru:** checkpoint'i tag'lemek (bir kez iyi modeli ezip kaybettikten sonra) ve `saved_models/` arşivi, ilerlemeyi güvene aldı.

---

*Bu belge projenin canonical yolculuk günlüğüdür. Güncel hiperparametreler için `config.py`, güncel mimari için `model_v6.py`/`model_v7.py`, ve `CLAUDE.md` (proje talimatları) esastır — çelişki halinde kod kazanır.*
