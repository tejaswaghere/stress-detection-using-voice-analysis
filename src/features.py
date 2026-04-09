"""
features.py — Audio feature extraction for emotion/stress detection.

What this file does:
  - Loads .wav files using librosa
  - Extracts a rich set of audio features (MFCCs, deltas, chroma, spectral contrast, mel spectrogram, ZCR, RMS)
  - Parses emotion labels from RAVDESS filenames
  - Walks the dataset directory and builds a (features, labels) array pair
  - Caches the result to .npy files so re-runs are instant

Why each feature matters:
  MFCCs           — Represent the "shape" of the vocal tract; great for distinguishing phonemes and emotion tone
  Delta-MFCCs     — Rate of change of MFCCs over time; captures dynamics (e.g. rising anger vs flat neutral)
  Delta²-MFCCs    — Acceleration of change; adds temporal texture
  Chroma          — Pitch class energy; helps distinguish tonal quality across emotions
  Spectral Contrast — Difference between peaks/valleys in spectrum; textures that differentiate breathy vs sharp speech
  Mel Spectrogram — Perceptually-scaled frequency energy; good basis for CNN input
  ZCR             — Zero-crossing rate; higher in consonant-heavy or noisy/angry speech
  RMS Energy      — Loudness; stressed/angry speech is louder, sad speech softer
"""

import numpy as np
import librosa
import os
from pathlib import Path


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

EMOTION_TO_INT = {v: i for i, v in enumerate(EMOTION_MAP.values())}
INT_TO_EMOTION = {v: k for k, v in EMOTION_TO_INT.items()}


def get_emotion_from_filename(filename: str) -> str:
    """
    RAVDESS filenames follow the pattern: 03-01-05-01-02-02-12.wav
    The 3rd field (index 2) is the emotion code.
    Returns the string label like 'angry', or 'unknown' if parsing fails.
    """
    try:
        code = Path(filename).stem.split('-')[2]
        return EMOTION_MAP.get(code, 'unknown')
    except (IndexError, AttributeError):
        return 'unknown'


def extract_features(file_path: str, sr: int = 22050, duration: float = 3.0) -> np.ndarray:
    """
    Load an audio file and return a 1-D feature vector.

    Parameters
    ----------
    file_path : str or Path
        Path to the .wav file.
    sr : int
        Target sample rate (default 22050 Hz — standard for librosa).
    duration : float
        Max seconds to load. RAVDESS clips are ~3s; trimming keeps features consistent.

    Returns
    -------
    np.ndarray of shape (193,)
        Concatenation of: 40 MFCCs, 40 delta-MFCCs, 40 delta²-MFCCs,
        12 chroma, 7 spectral contrast, 40 mel spectrogram means,
        1 ZCR mean, 1 RMS mean, 1 RMS std  →  182 features (actual dim printed on first run)
    """
    y, sr = librosa.load(file_path, sr=sr, duration=duration)

    # --- MFCC + deltas ---
    # n_mfcc=40 gives finer spectral resolution than the original 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)           # shape (40,)

    delta_mfcc = librosa.feature.delta(mfcc)
    delta_mfcc_mean = np.mean(delta_mfcc, axis=1)  # shape (40,)

    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)  # shape (40,)

    # --- Chroma ---
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)        # shape (12,)

    # --- Spectral Contrast ---
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)    # shape (7,)

    # --- Mel Spectrogram ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_mean = np.mean(mel_db, axis=1)           # shape (40,)

    # --- ZCR (zero-crossing rate) ---
    zcr_mean = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])  # shape (1,)

    # --- RMS Energy ---
    rms = librosa.feature.rms(y=y)[0]
    rms_features = np.array([np.mean(rms), np.std(rms)])  # shape (2,)

    return np.hstack([
        mfcc_mean,
        delta_mfcc_mean,
        delta2_mfcc_mean,
        chroma_mean,
        contrast_mean,
        mel_mean,
        zcr_mean,
        rms_features,
    ])


def load_dataset(
    dataset_path: str,
    cache_dir: str = 'data',
    force_reload: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk dataset_path recursively, extract features from every .wav file,
    and return (X, y) numpy arrays.

    Caching: After the first run, features are saved to .npy files in cache_dir.
    Subsequent runs load from cache — this saves ~5–10 minutes per run.

    Parameters
    ----------
    dataset_path : str
        Root folder of the RAVDESS dataset (contains Actor_01, Actor_02, ...)
    cache_dir : str
        Where to save/load cached .npy files.
    force_reload : bool
        If True, ignore cache and re-extract.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)  — integer labels
    """
    cache_X = Path(cache_dir) / 'features_X.npy'
    cache_y = Path(cache_dir) / 'features_y.npy'

    if not force_reload and cache_X.exists() and cache_y.exists():
        print(f"[features] Loading cached features from {cache_dir}/")
        return np.load(cache_X), np.load(cache_y)

    print(f"[features] Extracting features from {dataset_path} ...")
    X, y = [], []
    errors = 0

    for root, _, files in os.walk(dataset_path):
        for fname in sorted(files):
            if not fname.endswith('.wav'):
                continue
            emotion = get_emotion_from_filename(fname)
            if emotion == 'unknown':
                continue
            try:
                feat = extract_features(os.path.join(root, fname))
                X.append(feat)
                y.append(EMOTION_TO_INT[emotion])
            except Exception as e:
                print(f"  [warn] Skipping {fname}: {e}")
                errors += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.save(cache_X, X)
    np.save(cache_y, y)
    print(f"[features] Done. {len(X)} samples, {X.shape[1]} features each. {errors} errors.")
    print(f"[features] Cached to {cache_dir}/")
    return X, y
