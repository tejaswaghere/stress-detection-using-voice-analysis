"""
features.py — Audio loading, preprocessing and handcrafted feature extraction.

This module is the single source of truth for turning audio into model input.
Training (src/train.py), inference (src/predict.py) and the Gradio app all call
the functions here, so the features a model sees at serving time are exactly
the features it was trained on.

Preprocessing (applied to every clip, training and inference alike):
  1. Resample to 16 kHz mono
  2. Trim leading/trailing silence (RAVDESS clips have ~1 s of silence; mic
     recordings have arbitrary amounts — without trimming, clip duration and
     silence ratio leak into the features)
  3. Peak-normalise, so microphone gain does not masquerade as "loud = angry"

Handcrafted features (mean + std pooled over frames):
  MFCC (40) + Δ (40) + Δ² (40)  — vocal-tract shape and how it moves
  Log-mel spectrogram (40)       — perceptual spectral energy
  Spectral contrast (7)          — peak/valley ratio (breathy vs. pressed voice)
  Chroma (12)                    — pitch-class energy
  Centroid, bandwidth, roll-off, flatness, ZCR, RMS-dB (6) — brightness, noisiness, loudness
  + 5 pitch statistics (log-F0 mean/std/range, voiced ratio, jitter proxy)
  + speech duration
"""

from __future__ import annotations

import os
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16000
TRIM_TOP_DB = 30
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
FEATURE_VERSION = "handcrafted-v3"

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised',
}
EMOTIONS = list(EMOTION_MAP.values())
EMOTION_TO_INT = {e: i for i, e in enumerate(EMOTIONS)}
INT_TO_EMOTION = {i: e for e, i in EMOTION_TO_INT.items()}

_N_FFT = 512
_HOP = 160  # 10 ms at 16 kHz


# ─────────────────────────────────────────────────────────────────────────────
# RAVDESS metadata
# ─────────────────────────────────────────────────────────────────────────────

def parse_ravdess_filename(filename: str) -> dict | None:
    """
    RAVDESS filenames encode metadata: 03-01-05-01-02-02-12.wav
      modality-channel-EMOTION-INTENSITY-statement-repetition-ACTOR
    Returns {'emotion', 'intensity', 'actor'} or None if the name doesn't parse.
    """
    parts = Path(filename).stem.split('-')
    if len(parts) != 7 or parts[2] not in EMOTION_MAP:
        return None
    try:
        actor = int(parts[6])
    except ValueError:
        return None
    return {
        'emotion': EMOTION_MAP[parts[2]],
        'intensity': 'strong' if parts[3] == '02' else 'normal',
        'actor': actor,
    }


def get_emotion_from_filename(filename: str) -> str:
    """Return the emotion label encoded in a RAVDESS filename, or 'unknown'."""
    meta = parse_ravdess_filename(filename)
    return meta['emotion'] if meta else 'unknown'


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono, trim silence, peak-normalise. Raises ValueError on unusable audio."""
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        y = y / np.iinfo(y.dtype).max
    y = y.astype(np.float32)
    if y.ndim > 1:
        # Gradio numpy audio is (samples, channels); librosa expects (channels, samples)
        y = y.mean(axis=1) if y.shape[0] > y.shape[1] else y.mean(axis=0)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = y[: int(MAX_SECONDS * SAMPLE_RATE * 2)]  # bound work before trimming

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-4:
        raise ValueError("The recording is silent — check your microphone and try again.")

    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if len(y) < MIN_SECONDS * SAMPLE_RATE:
        raise ValueError(f"Too little speech detected (< {MIN_SECONDS} s). Please record at least 2 seconds.")
    y = y[: int(MAX_SECONDS * SAMPLE_RATE)]
    return y / np.max(np.abs(y))


def load_audio(path: str | os.PathLike) -> np.ndarray:
    """Load any audio file librosa can read and return preprocessed 16 kHz audio."""
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return preprocess(y, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Handcrafted features
# ─────────────────────────────────────────────────────────────────────────────

def _pool(m: np.ndarray) -> np.ndarray:
    return np.concatenate([m.mean(axis=1), m.std(axis=1)])


def pitch_track(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0 in Hz with NaN for unvoiced frames, voiced boolean mask)."""
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=600, sr=SAMPLE_RATE, frame_length=1024, hop_length=_HOP)
    return f0, voiced


def extract_features(y: np.ndarray) -> np.ndarray:
    """
    Compute the handcrafted feature vector for preprocessed 16 kHz audio
    (the output of load_audio / preprocess). Length == len(FEATURE_NAMES).
    """
    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    power = S ** 2
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=SAMPLE_RATE, n_mels=40))
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=40)
    width = min(9, mfcc.shape[1] - (1 - mfcc.shape[1] % 2))  # delta needs an odd width <= n_frames
    d1 = librosa.feature.delta(mfcc, width=max(width, 3), mode='nearest')
    d2 = librosa.feature.delta(mfcc, order=2, width=max(width, 3), mode='nearest')
    contrast = librosa.feature.spectral_contrast(S=S, sr=SAMPLE_RATE, n_bands=6, fmin=100)
    chroma = librosa.feature.chroma_stft(S=power, sr=SAMPLE_RATE)
    spectral = np.vstack([
        librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE),
        librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE),
        librosa.feature.spectral_rolloff(S=S, sr=SAMPLE_RATE),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(y, frame_length=_N_FFT, hop_length=_HOP)[:, : S.shape[1]],
        librosa.amplitude_to_db(librosa.feature.rms(S=S, frame_length=_N_FFT) + 1e-6),
    ])

    f0, voiced = pitch_track(y)
    lf0 = np.log(f0[voiced]) if np.any(voiced) else np.zeros(1)
    pitch = np.array([
        lf0.mean(),
        lf0.std(),
        np.percentile(lf0, 90) - np.percentile(lf0, 10),
        float(np.mean(voiced)),
        float(np.mean(np.abs(np.diff(lf0)))) if len(lf0) > 1 else 0.0,
    ])

    vec = np.concatenate([
        _pool(mfcc), _pool(d1), _pool(d2), _pool(logmel), _pool(contrast), _pool(chroma), _pool(spectral),
        pitch, [len(y) / SAMPLE_RATE],
    ]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _feature_names() -> list[str]:
    groups = [('mfcc', 40), ('d_mfcc', 40), ('d2_mfcc', 40), ('logmel', 40), ('contrast', 7), ('chroma', 12)]
    names = []
    for prefix, n in groups:
        names += [f'{prefix}_{i}_mean' for i in range(n)] + [f'{prefix}_{i}_std' for i in range(n)]
    spectral = ['centroid', 'bandwidth', 'rolloff', 'flatness', 'zcr', 'rms_db']
    names += [f'{s}_mean' for s in spectral] + [f'{s}_std' for s in spectral]
    names += ['logf0_mean', 'logf0_std', 'logf0_range', 'voiced_ratio', 'logf0_jitter', 'duration_s']
    return names


FEATURE_NAMES = _feature_names()


def features_from_file(path: str | os.PathLike) -> np.ndarray:
    """load_audio + extract_features in one call."""
    return extract_features(load_audio(path))


def describe_audio(y: np.ndarray) -> dict:
    """Human-readable acoustic summary for the UI (not used by the model)."""
    f0, voiced = pitch_track(y)
    rms = librosa.feature.rms(y=y, frame_length=_N_FFT, hop_length=_HOP)[0]
    return {
        'duration_s': len(y) / SAMPLE_RATE,
        'pitch_hz': float(np.nanmedian(f0[voiced])) if np.any(voiced) else float('nan'),
        'pitch_variability_semitones': float(12 * np.std(np.log2(f0[voiced]))) if np.sum(voiced) > 1 else 0.0,
        'voiced_ratio': float(np.mean(voiced)),
        'loudness_variability_db': float(np.std(librosa.amplitude_to_db(rms + 1e-6))),
        'spectral_centroid_hz': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE))),
        'f0': f0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading with caching
# ─────────────────────────────────────────────────────────────────────────────

def list_ravdess_files(dataset_path: str | os.PathLike) -> list[Path]:
    """All parseable RAVDESS .wav files under dataset_path, sorted for reproducibility."""
    return sorted(p for p in Path(dataset_path).rglob('*.wav') if parse_ravdess_filename(p.name))


def load_dataset(
    dataset_path: str | os.PathLike,
    cache_dir: str | os.PathLike = 'data',
    force_reload: bool = False,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract handcrafted features for every RAVDESS file (in parallel) and
    return (X, y, actors). Results are cached in cache_dir, keyed by
    FEATURE_VERSION so a stale cache from an older feature set is never reused.
    """
    from joblib import Parallel, delayed

    cache = Path(cache_dir) / f'features_{FEATURE_VERSION}.npz'
    if cache.exists() and not force_reload:
        print(f"[features] Loading cached features from {cache}")
        d = np.load(cache)
        return d['X'], d['y'], d['actors']

    files = list_ravdess_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No RAVDESS .wav files found under {dataset_path}")
    print(f"[features] Extracting features from {len(files)} files in {dataset_path} ...")

    X = np.stack(Parallel(n_jobs=n_jobs)(delayed(features_from_file)(p) for p in files))
    metas = [parse_ravdess_filename(p.name) for p in files]
    y = np.array([EMOTION_TO_INT[m['emotion']] for m in metas], dtype=np.int64)
    actors = np.array([m['actor'] for m in metas], dtype=np.int64)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, actors=actors)
    print(f"[features] Done: {X.shape[0]} samples x {X.shape[1]} features. Cached to {cache}")
    return X, y, actors
