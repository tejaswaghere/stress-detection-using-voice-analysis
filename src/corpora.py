"""
corpora.py — Uniform access to emotional-speech corpora, plus cached feature extraction.

Supported corpora
  ravdess  RAVDESS speech: 24 actors, 8 emotions, 1440 clips, 2 fixed sentences
           https://zenodo.org/record/1188976 (CC BY-NC-SA 4.0)
  cremad   CREMA-D: 91 actors (ages 20–74, diverse ethnicity), 6 emotions, 7442 clips, 12 sentences
           https://github.com/CheyneyComputerScience/CREMA-D (ODbL)

Both corpora share 6 emotions. RAVDESS's extra 'calm' and 'surprised' have no CREMA-D counterpart,
so cross-corpus experiments use SHARED_EMOTIONS only.

Speaker IDs are namespaced ('ravdess:12', 'cremad:1001'), so the corpora can be pooled for
speaker-grouped cross-validation without collisions.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from features import extract_features, load_audio, parse_ravdess_filename

SHARED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust']

_CREMAD_EMOTIONS = {'NEU': 'neutral', 'HAP': 'happy', 'SAD': 'sad', 'ANG': 'angry', 'FEA': 'fearful', 'DIS': 'disgust'}


@dataclass(frozen=True)
class Clip:
    path: Path
    corpus: str
    emotion: str
    speaker: str  # namespaced, e.g. 'cremad:1001'
    sex: str      # 'male' | 'female'


def _ravdess(root: Path) -> list[Clip]:
    clips = []
    for p in sorted(root.rglob('*.wav')):
        m = parse_ravdess_filename(p.name)
        if m:
            clips.append(Clip(p, 'ravdess', m['emotion'], f"ravdess:{m['actor']}",
                              'male' if m['actor'] % 2 else 'female'))
    return clips


def _cremad(root: Path) -> list[Clip]:
    """Filenames: <actor>_<sentence>_<emotion>_<level>.wav, e.g. 1001_DFA_ANG_XX.wav."""
    demo = root / 'VideoDemographics.csv'
    sex = {}
    if demo.exists():
        with open(demo, newline='') as f:
            sex = {row['ActorID']: row['Sex'].lower() for row in csv.DictReader(f)}
    clips = []
    for p in sorted(root.rglob('*.wav')):
        parts = p.stem.split('_')
        if len(parts) != 4 or parts[2] not in _CREMAD_EMOTIONS or p.stat().st_size < 1000:
            continue  # a handful of CREMA-D files are empty/corrupt
        clips.append(Clip(p, 'cremad', _CREMAD_EMOTIONS[parts[2]], f'cremad:{parts[0]}', sex.get(parts[0], 'unknown')))
    return clips


CORPORA = {'ravdess': _ravdess, 'cremad': _cremad}
DEFAULT_ROOTS = {'ravdess': 'data/RAVDESS', 'cremad': 'data/CREMA-D'}


def list_clips(corpus: str, root: str | Path | None = None) -> list[Clip]:
    root = Path(root or DEFAULT_ROOTS[corpus])
    clips = CORPORA[corpus](root)
    if not clips:
        raise FileNotFoundError(f"No {corpus} clips found under {root}")
    return clips


def _safe(fn, p):
    """Run a feature function on one file; None if the audio is unusable (e.g. CREMA-D's silent 1076_MTI_SAD_XX)."""
    try:
        return fn(load_audio(p))
    except ValueError as e:
        print(f"  [skip] {p.name}: {e}")
        return None


def _embed_all(paths: list[Path], backbone: str, layers: tuple[int, ...], checkpoint: Path) -> list:
    """Embed sequentially, checkpointing every 500 clips so an interrupted run can resume."""
    from embeddings import SSLEmbedder

    done = list(np.load(checkpoint, allow_pickle=True)['out']) if checkpoint.exists() else []
    if done:
        print(f"  resuming from checkpoint at {len(done)}/{len(paths)}")
    embedder = SSLEmbedder(backbone, layers)
    t0 = time.time()
    for i, p in enumerate(paths[len(done):], len(done) + 1):
        done.append(_safe(embedder, p))
        if i % 500 == 0:
            arr = np.empty(len(done), dtype=object)
            for j, row in enumerate(done):  # 1-D object array of embeddings / None
                arr[j] = row
            np.savez(checkpoint, out=arr)
            print(f"  {i}/{len(paths)}  ({time.time() - t0:.0f}s)", flush=True)
    return done


def _handcrafted_all(paths: list[Path]) -> list:
    from joblib import Parallel, delayed

    return Parallel(n_jobs=-1)(delayed(_safe)(extract_features, p) for p in paths)


def load_features(corpus: str, kind: str = 'embedding', root: str | Path | None = None, cache_dir: str = 'data',
                  backbone: str = 'microsoft/wavlm-base-plus', layers: tuple[int, ...] = (4, 5, 6, 7),
                  force: bool = False) -> tuple[np.ndarray, list[Clip]]:
    """
    Feature matrix for every usable clip of a corpus, cached on disk; returns (X, clips) aligned row by row.
    Clips whose audio is unusable (silent / too short) are skipped and listed in the cache.
    kind: 'embedding' (WavLM, default) or 'handcrafted'.
    """
    from features import FEATURE_VERSION

    clips = list_clips(corpus, root)
    tag = f'{backbone.split("/")[-1]}_{"-".join(map(str, layers))}' if kind == 'embedding' else FEATURE_VERSION
    cache = Path(cache_dir) / f'{kind}_{corpus}_{tag}.npz'
    if cache.exists() and not force:
        d = np.load(cache)
        by_name = {c.path.name: c for c in clips}
        if set(d['names']) <= set(by_name) and len(d['names']) + len(d.get('skipped', [])) == len(clips):
            print(f"[corpora] Loaded cached {kind} features for {corpus} from {cache}")
            return d['X'], [by_name[n] for n in d['names']]
        print(f"[corpora] Cache {cache} is stale (file list changed); re-extracting")

    print(f"[corpora] Extracting {kind} features for {len(clips)} {corpus} clips ...")
    cache.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = cache.with_suffix('.partial.npz')
    paths = [c.path for c in clips]
    feats = _embed_all(paths, backbone, layers, checkpoint) if kind == 'embedding' else _handcrafted_all(paths)
    ok = [f is not None for f in feats]
    clips = [c for c, k in zip(clips, ok) if k]
    X = np.stack([f for f in feats if f is not None])
    skipped = np.array([p.name for p, k in zip(paths, ok) if not k])
    np.savez(cache, X=X, names=np.array([c.path.name for c in clips]), skipped=skipped)
    checkpoint.unlink(missing_ok=True)
    print(f"[corpora] {len(clips)} clips cached to {cache}; skipped {len(skipped)} unusable")
    return X, clips
