# Design Decisions & Approach

What was tried, what was measured, and why the project looks the way it does.
Every number here comes from 6-fold cross-validation **grouped by actor**, so test speakers are never in the training set.

## Problem framing

Map a short utterance to one of 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised),
then derive a stress indicator. The signal is in *prosody* and *voice quality*, not lexical content. RAVDESS
actors say the same two sentences in every emotion, so the words carry no information at all.

## Lesson 1: evaluate on unseen speakers

v2 used `train_test_split(..., stratify=y)` over clips. RAVDESS has 60 clips per actor, so every test actor
also appeared in training, and a model could score well by recognising *who* was speaking.

| Pipeline | Random clip split | Unseen speakers |
|---|---|---|
| v2: 182 features, SVM | 62.4% | **40.9%** |
| v3: 376 features, SVM | 68.7% | **56.2%** |
| v3: WavLM + logistic regression | 88.8% | **79.0%** |

A deployed demo only ever meets unseen speakers, so the unseen-speaker column is the honest one.
`train.py` uses `GroupKFold` over actors and reports the random-split number only as a labelled comparison.

## Lesson 2: train/serve skew silently breaks a model

In v2, `app/app.py` re-implemented feature extraction and drifted from `src/features.py`: raw vs dB mel
energies, 5 s vs 3 s clips, and chroma computed from different inputs. The deployed Space also had no model file,
so it ran a hand-written rule-based fallback. v3 fixes both structurally:
- there is one `preprocess` + feature/embedding path in `src/`, imported by training, CLI and app;
- model bundles carry a feature-version tag and refuse to load against mismatched code;
- the Space is generated from the repo by `scripts/build_space.py` instead of being edited by hand.

## Handcrafted features (v3, 56%)

Changes from v2 and why:
- **Trim silence.** RAVDESS clips start with ~1 s of silence. Mic recordings have arbitrary amounts, and without
  trimming, silence ratio leaks into every mean.
- **Peak-normalise.** Otherwise "loud microphone" becomes a proxy for "angry". `tests/test_pipeline.py` checks that features are gain-invariant.
- **Mean + std pooling.** The std of MFCCs and energy captures how much the voice *moves*, which v2's means discarded.
- **Pitch.** pYIN log-F0 mean, std, range, voiced ratio and a jitter proxy. Pitch level and range are among the
  strongest arousal cues in the literature.
- Spectral centroid, bandwidth, roll-off and flatness for voice brightness and breathiness.

This is still kept (`--features handcrafted`) because it needs no torch, trains in seconds, and is
interpretable (`results/handcrafted_svm/feature_importance.png`).

## Pretrained speech embeddings (v3 default, 79%)

Frozen `microsoft/wavlm-base-plus`, with hidden states mean-pooled over time. A per-layer linear probe
([results/layer_probe.json](results/layer_probe.json)) gave:

```
layer   0     1     2     3     4     5     6     7     8     9    10    11    12
acc   .55   .67   .70   .73   .77   .78   .79   .76   .72   .71   .70   .70   .69
```

The middle layers win. The bottom layers are close to acoustics, and the top layers specialise in phonetic
content, which is useless here because every emotion uses the same sentences. Variants tried:

| Variant | Accuracy |
|---|---|
| layer 6, mean | 78.3% |
| layer 6, mean + std | 78.1% |
| **average of layers 4–7, mean (chosen)** | **79.1%** |
| concatenation of layers 4–7 (3072-d) | 79.9% |
| layer 6 + RBF SVM | 75.3% |

The differences between the top rows are within one fold's standard deviation (±4–5%), so the compact 768-d
average was chosen. The classifier is L2 logistic regression (C = 0.01) with balanced class weights. It is
linear, fast, gives sensible probabilities, and saves to 42 KB.

The selection was made on the same CV folds that report the final score, which adds a small optimistic bias.
A nested CV or an external test set (e.g. CREMA-D) would remove it.

## Stress index

RAVDESS has no stress labels, so stress is a function of the emotion posterior. On the circumplex model,
stress lives in the high-arousal, negative-valence quadrant. Weights: angry/fearful 1.0, disgust 0.7, sad 0.45,
surprised 0.35, happy 0.1, neutral/calm 0. Using the full probability vector, rather than only the arg-max,
makes the index degrade gracefully when the classifier is unsure. The thresholds (30, 55) are design choices,
not fitted values.

## Emotion timeline

For recordings longer than 4 s, the app runs WavLM once over the whole clip, then pools 3 s windows with a 0.5 s hop
from the frame-level hidden states. The timeline therefore costs no extra forward passes. Windows see the
full-utterance context through self-attention, which differs slightly from training on isolated clips, but
it lets you watch a voice move from calm to angry.

## Demo examples

The final model is fitted on 22 actors. Actors 23 (male) and 24 (female) are excluded, and their clips are the app's
examples, chosen by a fixed rule (statement 1, repetition 1, strong intensity) rather than picked for correct
predictions. One of them (happy, actor 23) is misclassified as surprised, which is representative.

## Known limitations

1. **Acted, English-only, two sentences.** Natural speech is subtler. Expect lower real-world accuracy.
2. **Speaker variance.** Per-actor accuracy ranges from 52% to 92%. Male voices average 74% vs 84% for female voices.
3. **Closed-set and overconfident.** Non-speech or very noisy input still gets a confident label.
4. **No speaker normalisation.** Each prediction sees one clip with no baseline for that speaker's neutral voice.

## Next steps

- Train on RAVDESS + CREMA-D (91 speakers), evaluate cross-corpus.
- Fine-tune WavLM's upper layers with a gradient-reversal speaker head.
- Temperature-scale the classifier and add a speech/non-speech gate.
- Per-user calibration: record a neutral baseline, then score deviations from it.
