# Changelog

All notable changes to this project are documented here.

## [v3.2.0] — 2026-10-04

### Added
- **Speech gate** (`src/gate.py`, `scripts/train_speech_gate.py`): a logistic regression on the WavLM embedding rejects
  music, noise and other non-speech instead of inventing an emotion. It accepts 99.97% of speech (99.96% of angry or
  fearful speech) and 97.5% of speech in loud noise, and rejects 100% of music and synthetic audio and 97.7% of
  environmental sounds, all on unseen sources. Silero VAD was evaluated and rejected: it turned away 1.7% of angry
  and fearful speech.
- The app shows a "No clear speech" card. The API returns `speech_detected: false` with a message, and the browser
  demo displays it.
- **Calibration analysis:** ECE, reliability diagram, and accuracy at high and low confidence in `train.py`; ECE and mean
  confidence for every cross-corpus condition. In-domain ECE is 1.6%; cross-corpus it is 21–27% (overconfident).
- Demo GIF in the README.

### Fixed
- Browser demo resampled audio with linear interpolation (aliasing), which flipped borderline predictions. It now
  sends native-rate audio and the server resamples.
- Space deploy: short description now fits Hugging Face's 60-character limit.

## [v3.1.0] — 2026-10-04

### Added
- **CREMA-D** (91 actors, 7,441 usable clips) via `src/corpora.py`, a uniform loader for both corpora with a shared
  6-emotion label set, namespaced speaker IDs, cached extraction that skips bad files, and checkpointing.
- **Cross-corpus study** (`src/cross_corpus.py`, `results/cross_corpus/`). A RAVDESS-only model drops from 81% to 38% UAR
  on CREMA-D. Per-corpus normalisation recovers 8–12 points. Hand-crafted features fall below chance across corpora.
- `scripts/download_cremad.py` (resumable) and tests for corpus parsing.
- Per-speaker accuracy plot that scales to 115 speakers, coloured by sex, with per-sex means.

### Changed
- **Shipped model is now trained on RAVDESS + CREMA-D** (6 emotions): 75.2% accuracy / 75.3% UAR on 115 unseen
  speakers, with fold spread ±1.8. The male/female gap narrowed from 10 to 4 points.
- RAVDESS *calm* is merged into *neutral*, which cuts the share of calm voices flagged as stressed from 61% to 18%.
  *Surprised* is no longer predicted.
- `train.py` takes `--corpora` and reports UAR, per-corpus and per-sex metrics.
- App, Space description and browser demo read the label set and dataset stats from the model bundle.
- Example clips: *neutral* and *disgust* replace *calm* and *surprised*. All are still from held-out actors 23 and 24.

## [v3.0.0] — 2026-10-03

### Fixed
- **The live demo now runs a real model.** The Hugging Face Space shipped without a model file and silently
  fell back to hand-written rules. The Space is now generated from the repo (`scripts/build_space.py`) with
  the trained model included.
- **Train/serve skew.** `app/app.py` re-implemented feature extraction differently from `src/features.py`
  (raw vs dB mel, 5 s vs 3 s, different chroma input). All code paths now share `src/features.py` / `src/predict.py`.
- **Inflated accuracy.** v2's ~68% came from a random clip split that leaks actors between train and test.
  Speaker-independent evaluation of v2 gives 40.9%. All reported numbers are now actor-grouped.
- README images pointed to plots that were never committed.
- Audio is now silence-trimmed and peak-normalised, so microphone gain no longer drives predictions.

### Added
- WavLM-base-plus embeddings + logistic regression: **79.0% unseen-speaker accuracy** (was 40.9%).
- Improved handcrafted feature set (376 features: mean+std pooling, pitch, spectral shape): 56.2%, no torch needed.
- Speaker-independent `train.py` (6-fold GroupKFold) writing `metrics.json`, confusion matrix, ROC, and per-actor accuracy.
- Layer probe for every WavLM layer (`results/layer_probe.json`).
- Redesigned Gradio app: stress gauge, pitch contour, emotion-over-time timeline, acoustic summary, JSON API (`/analyze`),
  and example clips from actors held out of the final model.
- GitHub Pages demo now calls the real model through the Space API (it used a 4-feature heuristic before).
- Stress index grounded in the arousal/valence circumplex model (`src/stress.py`).
- `src/predict.py` CLI, versioned model bundles that refuse stale feature versions.
- Test suite (`tests/`) and GitHub Actions CI.
- MIT `LICENSE` file (the README referenced one that didn't exist).

### Removed
- Unused, untrained CNN definition in `model.py`.
- Rule-based "fallback" predictions.

## [v2.0.0] — 2026-04

### Added
- Live microphone recording in browser demo (Web Audio API / getUserMedia)
- Real-time waveform visualisation during recording
- Confidence bar chart for all 8 emotion classes in browser demo
- Stress level indicator (low / moderate / high) based on emotion cluster
- Extracted feature display (pitch, RMS, ZCR, spectral centroid) in demo
- Polished Gradio app with confidence chart and stress output (`app/app.py`)
- HuggingFace Spaces deployment guide (`README_SPACES.md`)
- Design decisions document (`APPROACH.md`)
- Model comparison table in README (SVM vs RF vs GB)
- Confusion matrix and ROC curve images in `results/`

### Changed
- Demo no longer requires file upload — mic recording works out of the box
- README restructured: problem statement → architecture → results → quick start

### Fixed
- Feature extraction now handles audio clips < 0.5s gracefully (error message instead of crash)

---

## [v1.0.0] — 2026-04 (initial release)

### Added
- Feature extraction pipeline: 182-dim vector (MFCC 40 + delta 40 + delta² 40 + chroma 12 + spectral contrast 7 + mel 40 + ZCR 1 + RMS 2)
- Support for SVM, Random Forest, and Gradient Boosting classifiers
- 5-fold cross-validation training script (`src/train.py`)
- Evaluation plots: confusion matrix, ROC curves, feature importance (`src/evaluate.py`)
- Feature caching via `.npy` files (skip re-extraction on subsequent runs)
- Static browser demo with file upload (`demo/index.html`)
- Jupyter walkthrough notebook (`notebooks/emotion_detection.ipynb`)
- Training on RAVDESS dataset (1440 samples, 8 emotion classes, 24 actors)
- 65–70% test accuracy reported (SVM, 8-class). Note: measured on a random clip split; see v3.0.0
