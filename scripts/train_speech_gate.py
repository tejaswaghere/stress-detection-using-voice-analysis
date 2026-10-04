"""
train_speech_gate.py — Train and evaluate the speech gate (speech vs. non-speech on WavLM embeddings).

Why: the emotion classifier always outputs an emotion, even for silence, music or a door slam.
The gate decides whether a clip contains speech at all, reusing the WavLM embedding the emotion
model already computes (so it adds ~0 ms at inference).

Why not an off-the-shelf voice-activity detector? Silero VAD was tried first: at settings that
reject 97–99% of music and environmental sounds it also rejected 1.7% of *angry and fearful*
speech (shouting and panicky voices), exactly the voices a stress detector must not turn away.

Data
  positives: all RAVDESS + CREMA-D speech, plus copies mixed with background noise/music at 0–10 dB SNR
             ("speech with noise behind it is still speech")
  negatives: ESC-50 environmental sounds (clips 1–4 of each of 50 categories), instrumental music
             (librosa examples, 5 s chunks), synthetic noise/tones

Evaluation (all on data the gate never trained on)
  1. Grouped CV (10 folds): speakers, sound categories and music tracks never appear in both train and test.
  2. Robustness: 20 held-out speakers' clips with *unseen* noise files (ESC-50 clip 5 of each category)
     at 10 dB and 0 dB SNR, and phone-band audio (300–3400 Hz, 8 kHz).

    python scripts/train_speech_gate.py           # ~15 min CPU (embeds ~3.5k augmented clips)
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from corpora import load_features  # noqa: E402
from features import SAMPLE_RATE, load_audio, preprocess  # noqa: E402
from gate import GATE_THRESHOLD, build_gate  # noqa: E402

ESC = ROOT / 'data' / 'nonspeech' / 'esc50'
ESC_URL = 'https://raw.githubusercontent.com/karolpiczak/ESC-50/master'
HUMAN_VOCAL = {'crying_baby', 'laughing', 'coughing', 'sneezing', 'breathing', 'snoring'}
MUSIC = ['brahms', 'nutcracker', 'trumpet', 'vibeace', 'pistachio', 'sweetwaltz', 'humpback']
N_AUGMENT = 2400
N_ROBUST = 400
rng = np.random.default_rng(0)


# ── Non-speech sources ───────────────────────────────────────────────────────

def esc50(clip_index: int) -> list[tuple[str, np.ndarray]]:
    """The clip_index-th (sorted) ESC-50 file of every category, downloaded on demand."""
    import librosa

    ESC.mkdir(parents=True, exist_ok=True)
    meta = ESC / 'esc50.csv'
    if not meta.exists():
        urllib.request.urlretrieve(f'{ESC_URL}/meta/esc50.csv', meta)
    by_cat: dict[str, list[str]] = {}
    for r in csv.DictReader(open(meta)):
        by_cat.setdefault(r['category'], []).append(r['filename'])
    out = []
    for cat in sorted(by_cat):
        name = sorted(by_cat[cat])[clip_index]
        f = ESC / name
        if not f.exists():
            urllib.request.urlretrieve(f'{ESC_URL}/audio/{name}', f)
        out.append((cat, librosa.load(f, sr=SAMPLE_RATE, mono=True)[0]))
    return out


def music() -> list[tuple[str, np.ndarray]]:
    import librosa

    out = []
    for name in MUSIC:
        y, _ = librosa.load(librosa.ex(name), sr=SAMPLE_RATE, mono=True, duration=60)
        step = 5 * SAMPLE_RATE
        out += [(name, y[i:i + step]) for i in range(0, len(y) - step + 1, step)]
    return out


def synthetic() -> list[tuple[str, np.ndarray]]:
    n = 4 * SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    white = rng.standard_normal(n)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    pink, brown = np.fft.irfft(spec / np.sqrt(np.maximum(f, 1)), n), np.fft.irfft(spec / np.maximum(f, 1), n)
    out = []
    for amp in (0.01, 0.1, 0.5):
        out += [('white noise', amp * white), ('pink noise', amp * pink / np.abs(pink).max()),
                ('brown noise', amp * brown / np.abs(brown).max())]
    for hz in (100, 220, 440, 1000, 3000):
        out.append(('pure tone', 0.3 * np.sin(2 * np.pi * hz * t)))
        out.append(('harmonic pulsed tone', 0.3 * sum(np.sin(2 * np.pi * k * hz * t) / k for k in range(1, 6))
                    * 0.5 * (1 + np.sin(2 * np.pi * 4 * t))))
    out.append(('chirp', 0.3 * np.sin(2 * np.pi * (100 + 1000 * t) * t)))
    out.append(('mains hum', 0.3 * np.sin(2 * np.pi * 50 * t) + 0.01 * white))
    return out


# ── Augmentation ─────────────────────────────────────────────────────────────

def mix(y: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = np.resize(noise, len(y))
    py, pn = np.mean(y ** 2), np.mean(n ** 2) + 1e-12
    return (y + n * np.sqrt(py / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def phone(y: np.ndarray) -> np.ndarray:
    import librosa
    import scipy.signal as sg

    b, a = sg.butter(4, [300, 3400], btype='band', fs=SAMPLE_RATE)
    y8 = librosa.resample(sg.lfilter(b, a, y), orig_sr=SAMPLE_RATE, target_sr=8000)
    return librosa.resample(y8, orig_sr=8000, target_sr=SAMPLE_RATE).astype(np.float32)


def main():
    from embeddings import SSLEmbedder
    from sklearn.model_selection import GroupKFold, cross_val_predict

    t0 = time.time()
    emb = SSLEmbedder('microsoft/wavlm-base-plus', (4, 5, 6, 7))

    def embed(y):
        try:
            return emb(preprocess(np.asarray(y, dtype=np.float32), SAMPLE_RATE))
        except ValueError:  # silent / too short: rejected before the gate anyway
            return None

    # Speech (cached emotion-model embeddings)
    Xr, cr = load_features('ravdess')
    Xc, cc = load_features('cremad')
    Xs, clips = np.vstack([Xr, Xc]), cr + cc
    spk = np.array([c.speaker for c in clips])
    emo = np.array([c.emotion for c in clips])

    # Non-speech
    print('[gate] embedding non-speech ...', flush=True)
    neg = [(f'esc:{c}', 'vocal' if c in HUMAN_VOCAL else 'environmental', y) for c, y in esc50(0) + esc50(1) + esc50(2) + esc50(3)]
    neg += [(f'music:{n}', 'music', y) for n, y in music()]
    neg += [(f'synth:{n}', 'synthetic', y) for n, y in synthetic()]
    keep = [(g, k, e) for g, k, y in neg if (e := embed(y)) is not None]
    Xn = np.stack([e for *_, e in keep])
    gn = np.array([g for g, *_ in keep])
    kind = np.array([k for _, k, _ in keep])

    # Held-out speakers for the robustness test (excluded from all gate training)
    speakers = np.unique(spk)
    held = set(rng.choice([s for s in speakers if s.startswith('ravdess')], 5, replace=False)) | \
        set(rng.choice([s for s in speakers if s.startswith('cremad')], 15, replace=False))
    held_mask = np.isin(spk, list(held))

    # Augmented positives: speech + background (ESC-50 non-vocal clips 1–4, music) at 0–10 dB
    print(f'[gate] embedding {N_AUGMENT} noisy-speech clips ...', flush=True)
    backgrounds = [y for g, k, y in neg if k in ('environmental', 'music') and np.abs(y).max() > 1e-3]
    aug_idx = rng.choice(np.where(~held_mask)[0], N_AUGMENT, replace=False)
    Xa, ga = [], []
    for j, i in enumerate(aug_idx):
        e = embed(mix(load_audio(clips[i].path), backgrounds[rng.integers(len(backgrounds))], rng.uniform(0, 10)))
        if e is not None:
            Xa.append(e)
            ga.append(spk[i])
        if (j + 1) % 400 == 0:
            print(f'  {j + 1}/{N_AUGMENT}  ({time.time() - t0:.0f}s)', flush=True)
    Xa, ga = np.stack(Xa), np.array(ga)

    # ── 1. Grouped CV ────────────────────────────────────────────────────────
    tr = ~held_mask
    X = np.vstack([Xs[tr], Xa, Xn])
    y = np.r_[np.ones(tr.sum() + len(Xa)), np.zeros(len(Xn))]
    groups = np.r_[spk[tr], ga, gn]
    p = cross_val_predict(build_gate(), X, y, groups=groups, cv=GroupKFold(10), method='predict_proba', n_jobs=10)[:, 1]
    n_clean = tr.sum()
    p_speech, p_neg = p[:n_clean], p[n_clean + len(Xa):]
    hi = np.isin(emo[tr], ['angry', 'fearful'])
    cv = {
        'speech_accepted': float(np.mean(p_speech >= GATE_THRESHOLD)),
        'angry_fearful_speech_accepted': float(np.mean(p_speech[hi] >= GATE_THRESHOLD)),
        'noisy_speech_accepted': float(np.mean(p[n_clean:n_clean + len(Xa)] >= GATE_THRESHOLD)),
        **{f'{k}_rejected': float(np.mean(p_neg[kind == k] < GATE_THRESHOLD)) for k in sorted(set(kind))},
    }

    # ── 2. Robustness on held-out speakers with unseen noise files ─────────
    gate = build_gate().fit(X, y)
    unseen_noise = [y for c, y in esc50(4) if c not in HUMAN_VOCAL and np.abs(y).max() > 1e-3]
    test_idx = rng.choice(np.where(held_mask)[0], N_ROBUST, replace=False)
    conditions = {'clean': lambda a: a,
                  'noise_10dB': lambda a: mix(a, unseen_noise[rng.integers(len(unseen_noise))], 10),
                  'noise_0dB': lambda a: mix(a, unseen_noise[rng.integers(len(unseen_noise))], 0),
                  'phone_band': phone}
    robust = {}
    print('[gate] robustness test ...', flush=True)
    for name, fn in conditions.items():
        E = [embed(fn(load_audio(clips[i].path))) for i in test_idx]
        P = gate.predict_proba(np.stack([e for e in E if e is not None]))[:, 1]
        robust[name] = float(np.mean(P >= GATE_THRESHOLD))
    unseen_neg = gate.predict_proba(np.stack([e for e in (embed(y) for y in unseen_noise) if e is not None]))[:, 1]
    robust['unseen_environmental_rejected'] = float(np.mean(unseen_neg < GATE_THRESHOLD))

    # ── Final gate on everything (incl. held-out speakers) ──────────────────
    X_all = np.vstack([Xs, Xa, Xn])
    y_all = np.r_[np.ones(len(Xs) + len(Xa)), np.zeros(len(Xn))]
    import joblib
    final = build_gate().fit(X_all, y_all)
    joblib.dump({'pipeline': final, 'threshold': GATE_THRESHOLD, 'backbone': 'microsoft/wavlm-base-plus',
                 'layers': [4, 5, 6, 7]}, ROOT / 'models' / 'speech_gate.joblib', compress=3)

    metrics = {
        'method': 'logistic regression on the WavLM embedding used by the emotion model (layers 4-7, mean-pooled)',
        'threshold': GATE_THRESHOLD,
        'n': {'speech': int(len(Xs)), 'noisy_speech_augmented': int(len(Xa)), 'non_speech': int(len(Xn)),
              **{k: int((kind == k).sum()) for k in sorted(set(kind))}},
        'grouped_cv': cv,
        'robustness_held_out_speakers_unseen_noise': robust,
        'baseline_silero_vad': {
            'note': 'Silero VAD, frame threshold 0.5, >=0.5 s speech, 0.5 s padding (measured in the gate study)',
            'speech_accepted': 0.9886, 'angry_fearful_speech_accepted': 0.9754,
            'environmental_rejected': 0.994, 'music_rejected': 0.985, 'synthetic_rejected': 0.905,
        },
    }
    out = ROOT / 'results' / 'speech_gate'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({'grouped_cv': cv, 'robustness': robust}, indent=2))
    print(f'[gate] saved models/speech_gate.joblib ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
