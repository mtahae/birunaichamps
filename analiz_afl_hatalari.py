"""
analiz_afl_hatalari.py — AFL->AFIB hata vakalari GERCEKTEN belirsiz mi?
=======================================================================
Kesin soru: modelin AFIB sandigi AFL vakalari fizyolojik olarak AFIB'e mi
benziyor (asilamaz etiket-tavani), yoksa net flutter'li vakalar mi (model
kacirici, hala alan var)?

Yontem: v6 ensemble tahminleri -> true=AFL,pred=AFIB vakalarini bul. Her biri
icin (QRS-baskili atriyal sinyalden) olc:
  - RR duzensizligi (CV): AFL=dusuk(duzenli), AFIB=yuksek(kaotik)
  - Flutter tepe gucu (4-6Hz / toplam): AFL=yuksek(keskin tepe), AFIB=dusuk(genis)
Ve karsilastir: dogru-AFL, dogru-AFIB, ve AFL->AFIB HATALARI.
"""
import os, sys, numpy as np, torch
from torch.utils.data import DataLoader
from scipy.signal import welch, find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from adim07_model_mimarisi import EKGDataset
from model_v6 import CardioFusion6

device = config.DEVICE
PDD = config.PROCESSED_DATA_DIR
FS = config.TARGET_FS

val_ds = EKGDataset(os.path.join(PDD, "val_manifest.csv"),
                    os.path.join(PDD, "filtered_signals"), augment=False,
                    train_stats_path=os.path.join(PDD, "train_stats.npz"),
                    wide_features_dir=os.path.join(PDD, "wide_features"))
val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

# v6 ensemble tahminleri
CKPTS = ["best_model_seed42.pth", "best_model_seed123.pth", "best_model_seed2026.pth"]
all_probs = []
y_true = None
for ck in CKPTS:
    m = CardioFusion6(num_classes=5, num_aux_classes=config.NUM_AUX_CLASSES,
                      num_domains=2, wide_feature_dim=config.WIDE_FEATURE_DIM).to(device)
    m.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, ck), map_location=device, weights_only=True))
    m.eval()
    yt, yp = [], []
    with torch.no_grad():
        for sig, wf, lbl, _, _ in val_loader:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits, _, _ = m(sig.to(device), wf.to(device))
            yp.append(torch.softmax(logits.float(), 1).cpu().numpy()); yt.append(lbl.numpy())
    if y_true is None:
        y_true = np.concatenate(yt)
    all_probs.append(np.concatenate(yp))
y_pred = np.mean(all_probs, axis=0).argmax(1)


def qrs_suppress(lead, fs=FS):
    env = np.abs(lead); w = int(0.036*fs) | 1
    env = np.convolve(env, np.ones(w)/w, mode='same')
    med = np.median(env); mad = np.median(np.abs(env-med))+1e-6
    mask = env > med + 4.0*mad
    dil = int(0.12*fs) | 1
    mask = np.convolve(mask.astype(float), np.ones(dil), mode='same') > 0
    out = lead.copy(); out[mask] = 0.0
    return out

def rr_cv(lead, fs=FS):
    peaks, _ = find_peaks(lead, distance=int(0.3*fs), height=0.5*np.std(lead))
    if len(peaks) < 4: return np.nan
    rr = np.diff(peaks)/fs
    return np.std(rr)/(np.mean(rr)+1e-9)

def flutter_power(lead, fs=FS):
    supp = qrs_suppress(lead)
    f, psd = welch(supp, fs=fs, nperseg=min(512, len(supp)))
    flutter = np.sum(psd[(f>=4)&(f<=6)])
    total = np.sum(psd[(f>=0.5)&(f<=15)])+1e-9
    return flutter/total

# Her val ornegini isle (ham sinyalden — dataset normalize edilmis ama goreli metrikler ok)
metrics = {'rrcv': [], 'flutter': []}
idx = 0
for i in range(len(val_ds)):
    ecg_id = val_ds.ecg_ids[i]
    sig = np.load(os.path.join(PDD, "filtered_signals", f"{ecg_id}.npy"))
    lead_ii = sig[1]
    metrics['rrcv'].append(rr_cv(lead_ii))
    metrics['flutter'].append(flutter_power(lead_ii))
rrcv = np.array(metrics['rrcv']); flut = np.array(metrics['flutter'])

def summ(mask, tag):
    r = rrcv[mask]; fl = flut[mask]
    r = r[~np.isnan(r)]
    print(f"  {tag:28s} (n={mask.sum():3d}): RR_CV med={np.median(r):.3f}  flutter_guc med={np.median(fl):.4f}")

true_afl = y_true == 2
true_afib = y_true == 1
afl_correct = (y_true == 2) & (y_pred == 2)
afl_as_afib = (y_true == 2) & (y_pred == 1)   # HATALAR

print("=== FIZYOLOJIK METRIKLER (Lead II) ===")
print("  (AFL: RR_CV DUSUK + flutter YUKSEK | AFIB: RR_CV YUKSEK + flutter DUSUK)")
summ(true_afib, "Tum AFIB (referans)")
summ(afl_correct, "Dogru bilinen AFL")
summ(afl_as_afib, "AFL ama AFIB sanilan (HATA)")

print("\n=== YORUM ===")
r_err = rrcv[afl_as_afib]; r_err = r_err[~np.isnan(r_err)]
r_ok = rrcv[afl_correct]; r_ok = r_ok[~np.isnan(r_ok)]
f_err = flut[afl_as_afib]; f_ok = flut[afl_correct]; f_afib = flut[true_afib]
print(f"  Hatali AFL'lerin RR_CV medyani ({np.median(r_err):.3f}) vs dogru AFL ({np.median(r_ok):.3f})")
print(f"  Hatali AFL'lerin flutter gucu ({np.median(f_err):.4f}) vs dogru AFL ({np.median(f_ok):.4f}) vs AFIB ({np.median(f_afib):.4f})")
if np.median(r_err) > np.median(r_ok) * 1.3 and np.median(f_err) < np.median(f_ok) * 0.8:
    print("  -> HATALI AFL'ler fizyolojik olarak AFIB'e BENZIYOR (yuksek RR_CV + dusuk flutter)")
    print("     => BELIRSIZ ETIKET TAVANI. Hicbir model bunlari ayiramaz. 0.90 ulasilamaz.")
else:
    print("  -> Hatali AFL'lerde flutter sinyali VAR ama model kaciriyor => modelde ALAN olabilir")
