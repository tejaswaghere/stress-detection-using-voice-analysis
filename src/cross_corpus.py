"""
cross_corpus.py — Does the model generalise beyond the corpus it was trained on?

Within-corpus scores (even speaker-independent ones) only show that a model handles new
*speakers* recorded the same way: same studio, same sentences, same acting direction.
Real users differ in all of those. This script trains on one corpus and tests on another.

All conditions use the 6 emotions RAVDESS and CREMA-D share, and are speaker-independent:

  train \\ test  │ RAVDESS                        │ CREMA-D
  ───────────────┼────────────────────────────────┼────────────────────────────────
  RAVDESS        │ 6-fold GroupKFold by actor     │ fit on all RAVDESS
  CREMA-D        │ fit on all CREMA-D             │ 6-fold GroupKFold by actor
  both (pooled)  │ 6-fold GroupKFold over all 115 actors, scored separately per corpus

Plus one domain-adaptation variant: the cross-corpus cells re-run with each corpus standardised
by its *own* feature mean/std (unsupervised, as no target labels are used). This removes
recording-channel offsets such as microphone, room and loudness conventions.

Metrics: UAR (unweighted average recall = balanced accuracy, the standard metric in speech-emotion
research, robust to class imbalance), accuracy, macro-F1, and UAR by speaker sex.

Usage:
  python src/cross_corpus.py                        # WavLM embeddings (default)
  python src/cross_corpus.py --features handcrafted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))

from corpora import SHARED_EMOTIONS, load_features  # noqa: E402
from model import build_pipeline  # noqa: E402
import evaluate  # noqa: E402

N_FOLDS = 6
CORPORA = ('ravdess', 'cremad')
LABELS = {'ravdess': 'RAVDESS', 'cremad': 'CREMA-D', 'both': 'Both (pooled)'}


def load(corpus: str, kind: str, cache_dir: str):
    """Features, labels, speakers and sex for the clips of `corpus` that carry a shared emotion."""
    X, clips = load_features(corpus, kind, cache_dir=cache_dir)
    keep = np.array([c.emotion in SHARED_EMOTIONS for c in clips])
    clips = [c for c, k in zip(clips, keep) if k]
    y = np.array([SHARED_EMOTIONS.index(c.emotion) for c in clips])
    return X[keep], y, np.array([c.speaker for c in clips]), np.array([c.sex for c in clips])


def standardise(X: np.ndarray) -> np.ndarray:
    """Per-corpus z-scoring (unsupervised domain adaptation)."""
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def scores(y, pred, sex, proba=None) -> dict:
    out = {
        'uar': round(float(balanced_accuracy_score(y, pred)), 4),
        'accuracy': round(float(accuracy_score(y, pred)), 4),
        'macro_f1': round(float(f1_score(y, pred, average='macro')), 4),
        'n': int(len(y)),
    }
    if proba is not None:
        out['ece'] = round(evaluate.expected_calibration_error(y, proba), 4)
        out['mean_confidence'] = round(float(proba.max(1).mean()), 4)
    for s in ('male', 'female'):
        m = sex == s
        if m.any():
            out[f'uar_{s}'] = round(float(balanced_accuracy_score(y[m], pred[m])), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--features', choices=['embedding', 'handcrafted'], default='embedding')
    ap.add_argument('--cache-dir', default='data')
    ap.add_argument('--results', default='results/cross_corpus')
    args = ap.parse_args()
    model_type = 'logreg' if args.features == 'embedding' else 'svm'
    data = {c: load(c, args.features, args.cache_dir) for c in CORPORA}
    for c, (X, y, spk, _) in data.items():
        print(f"[cross] {c}: {len(y)} clips, {len(np.unique(spk))} speakers, {X.shape[1]} dims")

    results, preds = {}, {}

    # ── Within-corpus (speaker-independent CV) ───────────────────────────────
    for c in CORPORA:
        X, y, spk, sex = data[c]
        P = cross_val_predict(build_pipeline(model_type), X, y, groups=spk, cv=GroupKFold(N_FOLDS),
                              method='predict_proba', n_jobs=N_FOLDS)
        p = P.argmax(1)
        results[f'{c}->{c}'] = scores(y, p, sex, P)
        preds[f'{c}->{c}'] = (y, p)

    # ── Cross-corpus: fit on all of A, test on all of B ──────────────────────
    for norm in (False, True):
        for a, b in [('ravdess', 'cremad'), ('cremad', 'ravdess')]:
            Xa, ya, _, _ = data[a]
            Xb, yb, _, sexb = data[b]
            if norm:
                Xa, Xb = standardise(Xa), standardise(Xb)
            P = build_pipeline(model_type).fit(Xa, ya).predict_proba(Xb)
            p = P.argmax(1)
            key = f'{a}->{b}' + ('+corpus_norm' if norm else '')
            results[key] = scores(yb, p, sexb, P)
            preds[key] = (yb, p)

    # ── Pooled training, speaker-independent CV over all speakers ────────────
    for norm in (False, True):
        Xs = [standardise(data[c][0]) if norm else data[c][0] for c in CORPORA]
        X = np.vstack(Xs)
        y = np.concatenate([data[c][1] for c in CORPORA])
        spk = np.concatenate([data[c][2] for c in CORPORA])
        sex = np.concatenate([data[c][3] for c in CORPORA])
        src = np.concatenate([[c] * len(data[c][1]) for c in CORPORA])
        p = cross_val_predict(build_pipeline(model_type), X, y, groups=spk, cv=GroupKFold(N_FOLDS), n_jobs=N_FOLDS)
        for c in CORPORA:
            m = src == c
            key = f'both->{c}' + ('+corpus_norm' if norm else '')
            results[key] = scores(y[m], p[m], sex[m])
            preds[key] = (y[m], p[m])

    # ── Report ───────────────────────────────────────────────────────────────
    out = Path(args.results) / args.features
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        'features': args.features, 'model': model_type, 'emotions': SHARED_EMOTIONS,
        'chance_uar': round(1 / len(SHARED_EMOTIONS), 4),
        'protocol': 'within-corpus and pooled: 6-fold GroupKFold by speaker; cross-corpus: fit on all of A, test on all of B',
        'results': results,
    }
    (out / 'metrics.json').write_text(json.dumps(meta, indent=2))

    print(f"\n{'condition':32s} {'UAR':>6s} {'acc':>6s} {'F1':>6s} {'UAR♂':>6s} {'UAR♀':>6s} {'conf':>6s} {'ECE':>6s}")
    for k, r in results.items():
        print(f"{k:32s} {r['uar']:6.3f} {r['accuracy']:6.3f} {r['macro_f1']:6.3f} "
              f"{r.get('uar_male', float('nan')):6.3f} {r.get('uar_female', float('nan')):6.3f} "
              f"{r.get('mean_confidence', float('nan')):6.3f} {r.get('ece', float('nan')):6.3f}")

    rows = ['ravdess', 'cremad', 'both']
    for suffix, title in [('', 'no adaptation'), ('+corpus_norm', 'per-corpus normalisation')]:
        M = np.array([[results[f'{r}->{c}' + (suffix if r != c else '')]['uar'] for c in CORPORA] for r in rows])
        evaluate.plot_transfer_matrix(M, [LABELS[r] for r in rows], [LABELS[c] for c in CORPORA],
                                      f'Cross-corpus UAR, {args.features} ({title})',
                                      out / f'transfer_matrix{suffix.replace("+", "_")}.png', chance=1 / len(SHARED_EMOTIONS))
    for key in ('ravdess->cremad', 'both->cremad', 'both->ravdess'):
        y, p = preds[key]
        a, b = key.split('->')
        evaluate.plot_confusion_matrix(y, p, SHARED_EMOTIONS, out / f'confusion_{a}_to_{b}.png',
                                       title=f'Train {LABELS[a]} → test {LABELS[b]} (unseen speakers)')
    print(f"\n[cross] Wrote {out}/metrics.json and plots")


if __name__ == '__main__':
    main()
