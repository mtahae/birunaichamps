# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması — 2. Aşama. Classifies 12-lead ECG recordings into 5 classes (Normal, AFIB, AFL, LBBB, RBBB) using a competition dataset (TEKNOFEST, ~5k balanced records) merged with ~90k public-dataset records (PhysioNet Challenge 2020: CPSC, PTB-XL, Georgia, Chapman-Shaoxing/Ningbo, St Petersburg INCART). The single evaluation metric is **Macro F1** — accuracy is explicitly not trusted (see `ozet.md` §1.2). Full architecture/methodology narrative lives in `ozet.md`; treat it as the canonical design doc but verify against actual code before relying on specific numbers (see "Docs vs. current code" below).

## Environment

- Windows, PowerShell primary shell. A venv exists at `env/`.
- Dependencies in `requirements.txt` (numpy, pandas, wfdb, scipy, tqdm, matplotlib, torch/torchvision/torchaudio) — this list is incomplete; the code also imports `sklearn`, `neurokit2`, `h5py`, `imblearn`, `flask`, `tqdm`. Install what's missing as needed rather than trusting the file blindly.
- No test suite, linter, or build step exists in this repo. There is no CI config. Validation happens empirically by running training and reading Macro F1 / per-class F1 off the dashboard or console output — there is no `pytest`/`npm test` equivalent to invoke.

## Commands

```powershell
# Full pipeline from raw datasets (rarely needed — processed data already exists in outputs/)
python baslat.py

# Train only, skipping data pipeline (the normal day-to-day command)
python baslat.py --sadece-egitim

# Equivalent, more direct — this is what's actually used in practice
python adim08_egitim.py

# Live training dashboard (Flask, http://localhost:5000) — run in a separate terminal
python dashboard.py

# Individual pipeline stages (only needed when reprocessing raw data — see Data Pipeline below)
python adim00_veri_birlestirme.py    # merge datasets -> unified_manifest.csv
python adim01_kalite_kontrol_genel.py
python adim02_filtreleme.py          # resample 500->250Hz, bandpass filter
python adim03_kalite_kontrol.py
python adim04_segmentasyon.py        # -> fixed (12, 2500) windows
python adim06_veri_bolme.py          # train/val/test split + DANN domain_id assignment
python adim07b_wide_features.py      # precompute physiological features -> outputs/processed_data/wide_features/
python threshold_opt.py              # standalone self-test of threshold optimization (has __main__ block)
python model_v6.py                   # architecture smoke test: forward+backward shape check, no data needed
```

There's no single-test-runner concept here; the closest equivalent to "run one test" is invoking a stage script directly, or running a model file's `__main__` block for a forward/backward sanity check (see `model_v6.py`'s `if __name__ == "__main__"`).

## Architecture

### Data pipeline (adim00 → adim07b)

Sequential stages, each reading the previous stage's manifest CSV from `outputs/processed_data/` and writing its own:

`adim00` (merge 7 datasets, SNOMED-CT/AHA → label harmonization) → `adim01` (cross-dataset dedup + integrity) → `adim02` (resample 500→250Hz, Butterworth bandpass 0.5–40Hz via `filtfilt`, **no 50Hz notch** — see Critical Rules) → `adim03` (flat-line/clipping QC) → `adim04` (segment to fixed `(12, 2500)` windows, center-crop or symmetric zero-pad) → `adim06` (train/val/test split + DANN `domain_id` assignment per source dataset) → `adim07b` (precompute the 12-dim wide feature vector per record, cached as `.npy` for fast RAM loading in the `EKGDataset`).

`adim05_ozellik_cikarma.py` and `adim05b_veri_dengeleme.py` / `adim06b_oversampling.py` are **not part of the active path** — see "Docs vs. current code" below.

All stages read paths and hyperparameters exclusively from `config.py`; there is no other place hyperparameters should be hardcoded.

### Training (`adim08_egitim.py`)

Three-phase curriculum learning in a single run (phase boundaries and hyperparameters live in `config.py` as `EPOCHS_PHASE_1/2/3`, currently 25/30/25 = 80 total):

- **Phase 1 (TEKNOFEST only)**: balanced competition data only, DANN off, establishes a clean prior.
- **Phase 2 (TEKNOFEST + ~90k internet records)**: DANN on (`DANN_LAMBDA`), uses `create_domain_anchored_sampler` — class-balanced *and* TEKNOFEST-anchored (currently 4x oversampled) so the model doesn't drift into internet-data domain overfit. Loads the best Phase-1 checkpoint before starting, gets its own LR + a short `LinearLR` warmup into `CosineAnnealingLR` (`SequentialLR`) to avoid a loss spike at the phase transition.
- **Phase 3 (TEKNOFEST fine-tune)**: reloads best checkpoint, freezes the CNN backbone (layer names to freeze are architecture-specific — see the `freeze_layers` list in `adim08_egitim.py`, must be kept in sync with whichever model class is active), trains only the rhythm/classifier heads at low LR, SWA active.

Each phase transition reloads `checkpoint_path` and reduces LR by a fixed factor; patience counters are per-phase (`PATIENCE_P1/P2/P3`) and an early stop in phase 1 or 2 skips straight to the next phase's first epoch rather than ending training. A val-loss divergence check (`> 2.5x` best) also forces an early phase transition.

`ModelEMA` maintains a shadow exponential-moving-average copy of the weights, updated every optimizer step; at each epoch, both the raw and EMA model are validated and whichever scores higher gets checkpointed (`[BEST-RAW]`/`[BEST-EMA]` in logs). After training, SWA batch-norm stats are recomputed, TTA (5 augmented views) is run, and both the *independently-optimal-per-class* threshold F1 (an inflated upper bound) and the *realistic* threshold F1 (via `apply_thresholds`, one consistent decision rule) are reported — treat the latter as the number that matters, not the former.

Live progress is written to `outputs/training_log.json` and polled by `dashboard.py`.

### Model architectures (two exist — know which is live)

`adim08_egitim.py` currently imports **`CardioFusion6`** from `model_v6.py`, not `CardioFusion5` from `adim07_model_mimarisi.py`. `CardioFusion5` (SE-ResNet + 2-layer Transformer + DANN + multi-task, ~11.8M params) is kept as a fallback but is not the training entry point. `CardioFusion6` ("Lean-Robust", ~3.9M params) was introduced specifically to fight overfitting seen with v5: it replaces the Transformer with a lighter BiGRU rhythm branch, adds Instance Norm in early stem/stage layers (domain-robustness — normalizes away device/amplitude differences per-sample) plus Stochastic Depth (`DropPath`) for regularization, and adds an explicit `CrossLeadAttention` module that encodes each of the 12 leads as a token (shared small CNN encoder) and runs multi-head self-attention across leads — meant to let the model learn relationships like V1↔V6 that matter for LBBB/RBBB. It also has a `SpectralAtrialBranch` that takes the FFT of the atrial-activity leads (II, III, aVF, V1), keeps the 0.5–15Hz band, and max-pools over frequency — a physiologically-motivated, low-parameter branch aimed squarely at the AFIB↔AFL confusion (AFL flutter = a sharp ~4–6Hz peak, AFIB = broadband/no peak). The FFT runs inside `torch.autocast(enabled=False)` because half-precision FFT is unsupported. `SpectralAtrialBranch` and the wide-feature MLP are deliberately **not** in the Phase-3 `freeze_layers` list so they keep fine-tuning on TEKNOFEST (they carry the AFIB/AFL signal).

Both models share `EKGDataset` and `FocalLoss` from `adim07_model_mimarisi.py`. `FocalLoss.forward` supports `reduction='none'` specifically so per-sample hard-example-mining weights (`hard_alpha`, AFIB/AFL boosted) can be applied correctly — do not silently drop that reduction path if refactoring, since a plain `.mean()`-then-scale collapses the per-sample weighting into a single batch-wide scalar (a real bug that existed here before and was fixed).

Whichever model is active, `forward(x, wide_features, alpha)` must return `(class_logits, aux_logits, domain_logits)` — this exact interface is assumed by `train_one_epoch`/`validate` in `adim08_egitim.py`. If you swap in a new architecture, also update the `freeze_layers` name list used in the Phase-3 backbone freeze, since it matches against `named_parameters()` string prefixes specific to the old/new module names.

### Physiological feature injection

Beyond raw signal, a 12-dim "wide features" vector (`adim07b_wide_features.py` / `features.py`) is concatenated into the classifier: HR mean/std, RR mean/std/CV, NN50/pNN50, age, gender, plus AFIB/AFL-specific P-wave presence/regularity and atrial rate. `EKGDataset` caches these as `.npy` in `outputs/processed_data/wide_features/{train,val}_wide_features_cache.npy` for fast reload — delete the cache if the underlying feature extraction logic changes, since it won't auto-invalidate.

### Docs vs. current code — don't trust docs blindly

Several markdown docs in the repo root (`ozet.md`, `BirunAI_Architecture.md`, `mertozet.md`, `model_architecture_and_training_report.md`, `birunaikeremabi.md`, `gelistirilecekyonler.md`, `collab.md`) describe methodology and historical results at different points in time. `ozet.md` is the most complete narrative but reflects `CardioFusion5`/older `config.py` values (e.g. it documents `EPOCHS_PHASE_2 = 80`, `DANN_LAMBDA = 0.1` — current `config.py` has `30` and `0.3` after a fix for Phase-2 domain-overfit "loss spike" behavior observed on the dashboard). When a doc's stated hyperparameter/architecture conflicts with `config.py` or the active model file, **the code wins** — treat docs as historical rationale, not current ground truth.

Known stale/inactive code paths (don't assume these run in the current workflow):
- `adim09_degerlendirme.py` and parts of `adim10_gradcam.py` reference/import `BirunAIModel` or fall back to `CardioFusion5` — they have not been updated for `CardioFusion6` and will break or silently evaluate the wrong architecture if invoked as-is. `adim10`'s GradCAM call at the end of `adim08_egitim.py` is wrapped in a `try/except` specifically because of this.
- `adim06b_oversampling.py` (SMOTE-based) and `adim05b_veri_dengeleme.py` implement class-balancing strategies from an earlier 3-class (stage 1) design and are **not** invoked by the current `adim08_egitim.py` training path (which does its own balancing via `create_balanced_sampler`/`create_domain_anchored_sampler`). `adim06b_oversampling.py` also contains a `time_shift` augmentation helper — this directly violates the project's own forbidden-augmentation rule (see below) and should not be reintroduced into the active pipeline.
- `baslat.py` orchestrates the full legacy pipeline including these stale stages; in practice, since `outputs/processed_data/` is already populated, `python adim08_egitim.py` is run directly instead.

## Critical domain rules (violating these silently destroys Macro F1)

These are asserted repeatedly across the docs and enforced in code — preserve them in any preprocessing/augmentation change:

1. **Lead-wise Z-score only, computed from train set only.** Never normalize across all 12 leads together (V1's ~0.5mV QRS vs V5's ~2.5mV means global normalization erases V1 — and RBBB diagnosis lives in V1). Val/test must reuse train-computed `train_stats.npz` mean/std, never recompute their own (`EKGDataset` in `adim07_model_mimarisi.py` implements this correctly — treat it as the reference).
2. **Per-patient / per-record split with no leakage** between train/val/test.
3. **No time-shift (`np.roll`) and no random crop augmentation** — both break the P-QRS-T temporal relationship. Segmentation crop is always symmetric/center, never random.
4. **No global amplitude scaling** — only lead-wise (0.9–1.1 independently per lead), since global scaling distorts the V1/V6 ratio that RBBB/LBBB diagnosis depends on.
5. **No 50Hz notch filter** — bandpass 0.5–40Hz (Butterworth, order 4, `filtfilt`) is considered sufficient; a notch filter risks damaging the T wave.
6. **Never fully drop a lead** for quality reasons — SQI-based weighting/dropout is fine (and used as an augmentation — lead dropout), but a lead is never physically removed from the 12-lead input.

Full context on why these rules exist is in `birunaikeremabi.md` and `ozet.md` §11 if deeper justification is ever needed for a design decision.

## Config

`config.py` is the single source of truth for hyperparameters, dataset paths, label mappings (SNOMED-CT/AHA → 5-class), and reproducibility (`set_seed(42)`, deterministic cuDNN). Do not hardcode hyperparameters elsewhere — add them here.
