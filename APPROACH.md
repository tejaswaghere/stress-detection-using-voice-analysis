# Design Decisions & Approach

This document explains the technical choices made in this project — what was tried, what worked, what didn't, and what comes next.

## Problem framing

Detecting stress and emotion from voice is fundamentally a **sequence classification problem**. A short audio clip (3–5 seconds) must be mapped to one of 8 emotional states.

The challenge: human speech is high-dimensional (22,050 samples/second × 5s = ~110,000 raw values), but most of that signal is irrelevant to emotion. The job is feature engineering — compressing 110,000 values down to a small vector that preserves emotional content.

## Dataset choice: RAVDESS

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) was chosen because:

- Controlled conditions — professional actors, consistent recording quality
- 8 emotion classes with matching intensity levels (normal + strong)
- Widely used in published research, so accuracy figures are comparable
- 1440 samples across 24 actors (12M, 12F) — good gender balance

**Limitation acknowledged:** RAVDESS is acted emotion, not naturalistic. A model trained only on RAVDESS may generalise poorly to real-world stressed speech (e.g., a tired developer at 2am). Adding CREMA-D and MSP-Podcast would help. This is on the roadmap.

## Feature engineering: why 182 dimensions

The v1 notebook used only 13 mean-pooled MFCCs. This was a significant limitation.

**What was added and why:**

| Feature | Dims | Reasoning |
|---|---|---|
| MFCCs (40 coeff) | 40 | Vocal tract shape. 40 > 13 captures higher-frequency formants important for distinguishing disgust/surprised. |
| Delta-MFCCs | 40 | *How* tone changes over time — critical for angry (fast rise) vs sad (slow decay). |
| Delta²-MFCCs | 40 | Acceleration of change. Fearful speech often has jittery second derivatives. |
| Chroma | 12 | Pitch class energy. Happy speech tends toward major-key pitch distributions. |
| Spectral Contrast | 7 | Energy ratio across frequency bands. Angry speech has high contrast; calm has low. |
| Mel Spectrogram | 40 | Perceptually-weighted frequencies. Humans are more sensitive to lower frequencies — mel scale reflects this. |
| ZCR | 1 | High ZCR correlates with consonant-dense, tense speech (angry, fearful). |
| RMS mean + std | 2 | Loudness and loudness variation. Angry = high mean + high std. Calm = low mean + low std. |

**Total: 182 features.** This is deliberately compact — a single sklearn pipeline can fit in under a minute on a CPU.

## Model selection

Three classifiers were evaluated with 5-fold cross-validation:

- **SVM (RBF kernel):** Best accuracy (~68%). SVMs work well on high-dimensional feature spaces with limited data. The RBF kernel implicitly captures non-linear emotion boundaries.
- **Random Forest:** Slightly lower accuracy (~64%) but much faster inference and interpretable via feature importance. ZCR and RMS consistently ranked as top features.
- **Gradient Boosting:** Comparable to SVM on accuracy (~67%) but 4× slower to train and much slower at inference.

SVM was chosen as the default. For production, Random Forest is arguably better — it's faster, lighter, and you can explain to a stakeholder *which features drove a prediction*.

## What didn't work

- **Mean-pooling everything:** Averaging MFCCs over the full clip throws away temporal dynamics. The phrase "I'm FINE" said through gritted teeth has the same mean MFCCs as a calm "I'm fine." Delta features partially recover this.
- **Raw waveform as input:** Tested a simple 1D CNN on raw audio. Without extensive data augmentation it overfit immediately on 1440 samples.
- **PCA before SVM:** Reducing to 50 components degraded accuracy by ~5%. The SVM handles 182 dimensions without dimensionality reduction — there's no curse of dimensionality issue here because SVM is a max-margin classifier.

## Known limitations

1. **Small dataset:** 1440 samples for 8 classes = 180 samples/class. This is why simple classifiers beat neural networks here — CNNs need 10,000+ samples per class to outperform SVM.
2. **Acted vs naturalistic emotion:** RAVDESS actors exaggerate emotions. Real-world stressed speech is subtler.
3. **English only:** All RAVDESS speakers are North American English speakers. Emotion prosody differs across languages and cultures.
4. **No speaker normalisation:** A naturally loud speaker may be mis-classified as angry. Speaker-normalised features (e.g., relative pitch rather than absolute) would help.

## Roadmap

- [ ] **CREMA-D integration** — 7,442 clips, more diverse speaker pool
- [ ] **CNN on 2D Mel spectrograms** — likely to push accuracy to 75–80% with more data
- [ ] **LSTM on MFCC sequences** — captures temporal dynamics natively instead of via delta features
- [ ] **Real-time mic streaming** — pyaudio + sliding window inference
- [ ] **Speaker normalisation** — z-score per speaker before extracting global features
- [ ] **Explainability** — SHAP values per prediction

## References

- Livingstone & Russo (2018). [RAVDESS dataset](https://zenodo.org/record/1188976). *PLOS ONE.*
- McFee et al. (2015). [librosa](https://librosa.org). *Proceedings of SciPy.*
- Schuller et al. (2018). *The INTERSPEECH 2018 Computational Paralinguistics Challenge.* (Benchmark for emotion recognition accuracy ranges.)
