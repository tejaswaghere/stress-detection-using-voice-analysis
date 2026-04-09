# Changelog

All notable changes to this project are documented here.

## [v2.0.0] — 2025-XX-XX

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

## [v1.0.0] — 2025-XX-XX  ← Initial release

### Added
- Feature extraction pipeline: 182-dim vector (MFCC 40 + delta 40 + delta² 40 + chroma 12 + spectral contrast 7 + mel 40 + ZCR 1 + RMS 2)
- Support for SVM, Random Forest, and Gradient Boosting classifiers
- 5-fold cross-validation training script (`src/train.py`)
- Evaluation plots: confusion matrix, ROC curves, feature importance (`src/evaluate.py`)
- Feature caching via `.npy` files (skip re-extraction on subsequent runs)
- Static browser demo with file upload (`demo/index.html`)
- Jupyter walkthrough notebook (`notebooks/emotion_detection.ipynb`)
- Training on RAVDESS dataset (1440 samples, 8 emotion classes, 24 actors)
- 65–70% test accuracy (SVM, 8-class classification)
