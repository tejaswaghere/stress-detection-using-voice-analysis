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

## Pretrained speech embeddings (RAVDESS-only: 79%)

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

## Lesson 3: unseen speakers ≠ unseen recording conditions (v3.1)

Speaker-independent CV still keeps the studio, the microphone, the sentences and the acting direction fixed.
A demo user changes all of them. To measure that, CREMA-D was added: 91 actors aged 20–74 from diverse
backgrounds, 12 sentences, recorded separately from RAVDESS. Both corpora share 6 emotions. All numbers below
are UAR (mean per-class recall) on unseen speakers, with chance at 16.7% (`src/cross_corpus.py`).

| Train → test | WavLM | Hand-crafted |
|---|---|---|
| RAVDESS → RAVDESS | 81.3% | 59.1% |
| CREMA-D → CREMA-D | 75.3% | 55.1% |
| RAVDESS → CREMA-D | 37.7% | 14.3% |
| CREMA-D → RAVDESS | 58.3% | 23.9% |
| … with per-corpus z-scoring | 45.3% / 70.4% | 32.7% / 37.5% |
| pooled → RAVDESS / CREMA-D | 77.9% / 75.3% | 53.4% / 55.7% |

**Diagnosis.** The RAVDESS-only WavLM model labels 51% of CREMA-D clips *fearful* (true share: 17%), so it isn't
making random errors. The whole CREMA-D distribution sits in the region RAVDESS associates with fear. Standardising
each corpus by its own mean and standard deviation (no labels needed) cuts that to 18% and recovers 8–12 points UAR.
So most of the transfer gap is a channel and recording offset, not a lack of emotion knowledge. Hand-crafted features
fail almost completely: RAVDESS → CREMA-D is below chance with macro-F1 0.07, a near-total collapse onto one class.

**Decision: ship a pooled model.** Training on both corpora matches the CREMA-D-only model on CREMA-D (75.3%) and
costs 3 points on RAVDESS. It also narrows the male/female gap from 10 to 4 points UAR, and its fold-to-fold
spread drops from ±3.7 to ±1.8 because 115 speakers give a much more stable estimate than 24. Regularisation
was re-checked for the larger dataset: C = 0.01 is still best (C ∈ {0.003, 0.01, 0.03, 0.1}).
Per-corpus normalisation isn't used at inference, because a single user clip has no "corpus statistics" to
normalise with. A running per-user baseline is on the roadmap.

**The calm problem.** CREMA-D has no *calm*. Three options were compared with out-of-fold predictions:

| Option | RAVDESS UAR | CREMA-D UAR | Unseen calm clips scored ≥ 30 stress |
|---|---|---|---|
| drop calm | 78.0% | 75.3% | 61% (read as *sad*) |
| calm as a 7th class | 74.0% | 75.4% | 5% |
| **merge calm → neutral** | **78.3%** | **75.2%** | **18%** |

The 7-class model looks best on calm clips, but it predicted *calm* for exactly 0.0% of CREMA-D clips, including
CREMA-D's quiet neutral and sad ones. It had learned "RAVDESS recording conditions = calm", which would never fire for
a user on their own microphone. Merging calm into neutral teaches "calm voice → no stress" in a way that doesn't
depend on recording conditions, so that is what ships.

**Data quality.** One CREMA-D file (`1076_MTI_SAD_XX.wav`) is silent. The extractor skips unusable clips and
lists them in the cache, and it checkpoints every 500 clips. The first full run crashed on that file after 16 minutes.

## Lesson 4: refuse to answer when the input isn't speech (v3.2)

**Speech gate.** The classifier always outputs an emotion, so non-speech gets confident nonsense. First attempt: Silero
VAD. Its frame-level speech probabilities were collected for every clip and a grid of rules was swept (frame threshold
× minimum speech duration). No setting got both jobs right: rules that rejected 97–99% of music and environmental
sounds also rejected 1.7–2.5% of *angry and fearful* speech, because shouting and panicky delivery look unlike
the conversational speech a VAD is tuned for.

Second attempt: a logistic regression on the WavLM embedding the emotion model already computes. With grouped
cross-validation (no speaker, sound category or music track in both train and test) it accepted 99.95% of speech and
rejected 100% of non-speech. But a robustness test on held-out speakers with *unseen* noise files showed it rejected
5.5% of speech at 10 dB SNR and 30% at 0 dB: it had learned "clean studio audio = speech". Adding 2,400 copies of
speech mixed with background sounds and music (0–10 dB SNR) as positives fixed that (99.75% / 97.5% accepted) at a
small cost on environmental sounds (100% → 97.7% rejected).

**Calibration.** On unseen speakers from the training corpora the model is already well calibrated (ECE 1.6%; temperature
scaling fitted T = 1.03 and changed nothing, so it isn't used). Across corpora calibration collapses (ECE 21–27%:
79% confident, 58% correct). The honest takeaway: the confidence shown in the app is meaningful for audio resembling
the training data and optimistic otherwise. The UI therefore hedges below 50% confidence, where accuracy is 46%.

**Remaining gap.** The gate is noise-robust, but the emotion model isn't: at 5 dB SNR a neutral clip read as sad. Noise
augmentation for the emotion model is the obvious next step.

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

1. **Acted, English-only, fixed sentences.** Natural speech is subtler, and the cross-corpus results show that a new
   recording setup alone can cost a lot. Expect lower real-world accuracy.
2. **Speaker variance.** With the pooled model, per-speaker accuracy ranges from 46% to 94%. Male voices score 73.5% UAR
   against 77.3% for female voices.
3. **Closed-set and overconfident.** Non-speech or very noisy input still gets a confident label.
4. **No speaker normalisation.** Each prediction sees one clip with no baseline for that speaker's neutral voice.

## Next steps

- Test on a third, never-used corpus (e.g. SAVEE, TESS, or naturalistic MSP-Podcast).
- Per-user baseline: z-score a user's clips against their own neutral recording (the inference-time analogue of
  the per-corpus normalisation that recovered 8–12 points).
- Fine-tune WavLM's upper layers with a gradient-reversal speaker head.
- Temperature-scale the classifier and add a speech/non-speech gate.
- Per-user calibration: record a neutral baseline, then score deviations from it.
