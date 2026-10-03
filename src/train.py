"""
train.py — Train and evaluate an emotion classifier on RAVDESS.

Evaluation is speaker-independent: 6-fold GroupKFold over the 24 actors, so
every test clip comes from 4 actors the model never heard during training.
(A random clip-level split lets the same actor appear in train and test and
inflates accuracy by ~20 points — that number is also reported, labelled.)
The out-of-fold predictions cover all 1440 clips and drive every plot in
results/<features>_<model>/. The final model is then refit on every actor except two (23, 24) whose clips
serve as the app's examples, so those examples are genuinely unseen.

Usage:
  python src/train.py --dataset data/RAVDESS                          # WavLM embeddings + logreg (default)
  python src/train.py --dataset data/RAVDESS --features handcrafted --model svm   # no torch needed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))

from features import EMOTIONS, EMOTION_TO_INT, FEATURE_NAMES, list_ravdess_files, load_audio, load_dataset, parse_ravdess_filename  # noqa: E402
from model import CLASSIFIERS, build_pipeline, make_bundle, save_bundle  # noqa: E402
from stress import STRESS_WEIGHTS  # noqa: E402
import evaluate  # noqa: E402

N_FOLDS = 6  # 24 actors → 4 held-out actors (2 female, 2 male on average) per fold
DEFAULT_LAYERS = (4, 5, 6, 7)  # middle layers encode prosody best — see results/layer_probe.json


def load_embeddings(dataset: str, cache_dir: str, backbone: str, layers: tuple[int, ...], force: bool):
    from embeddings import SSLEmbedder

    tag = backbone.split('/')[-1]
    cache = Path(cache_dir) / f'embeddings_{tag}_{"-".join(map(str, layers))}.npz'
    if cache.exists() and not force:
        print(f"[train] Loading cached embeddings from {cache}")
        d = np.load(cache)
        return d['X'], d['y'], d['actors']

    files = list_ravdess_files(dataset)
    if not files:
        raise FileNotFoundError(f"No RAVDESS .wav files found under {dataset}")
    embedder = SSLEmbedder(backbone, layers)
    print(f"[train] Embedding {len(files)} clips with {backbone} layers {list(layers)} ...")
    t0, X = time.time(), []
    for i, f in enumerate(files, 1):
        X.append(embedder(load_audio(f)))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    metas = [parse_ravdess_filename(f.name) for f in files]
    X = np.stack(X)
    y = np.array([EMOTION_TO_INT[m['emotion']] for m in metas])
    actors = np.array([m['actor'] for m in metas])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, y=y, actors=actors)
    return X, y, actors


def parse_args():
    p = argparse.ArgumentParser(description='Train an emotion classifier on RAVDESS (speaker-independent evaluation).')
    p.add_argument('--dataset', required=True, help='RAVDESS root folder (contains Actor_01 ... Actor_24)')
    p.add_argument('--features', choices=['embedding', 'handcrafted'], default='embedding')
    p.add_argument('--model', choices=CLASSIFIERS, default=None, help='default: logreg for embeddings, svm for handcrafted')
    p.add_argument('--C', type=float, default=None, help='regularisation strength override')
    p.add_argument('--backbone', default='microsoft/wavlm-base-plus')
    p.add_argument('--layers', type=int, nargs='+', default=list(DEFAULT_LAYERS))
    p.add_argument('--cache-dir', default='data')
    p.add_argument('--out', default='models/emotion_model.joblib')
    p.add_argument('--results', default='results')
    p.add_argument('--demo-actors', type=int, nargs='*', default=[23, 24],
                   help='actors left out of the final fit so the app example clips are truly unseen')
    p.add_argument('--force-reload', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    model_type = args.model or ('logreg' if args.features == 'embedding' else 'svm')
    layers = tuple(args.layers)

    if args.features == 'embedding':
        X, y, actors = load_embeddings(args.dataset, args.cache_dir, args.backbone, layers, args.force_reload)
    else:
        X, y, actors = load_dataset(args.dataset, args.cache_dir, args.force_reload)
    print(f"[train] {X.shape[0]} clips, {X.shape[1]} features, {len(np.unique(actors))} actors")

    # ── Speaker-independent cross-validation ─────────────────────────────────
    pipe = build_pipeline(model_type, args.C)
    oof_proba = cross_val_predict(pipe, X, y, groups=actors, cv=GroupKFold(N_FOLDS), method='predict_proba', n_jobs=N_FOLDS)
    oof_pred = oof_proba.argmax(1)
    fold_acc = [accuracy_score(y[test], oof_pred[test]) for _, test in GroupKFold(N_FOLDS).split(X, y, actors)]

    # Same model, leaky random split — reported only to show the gap
    leaky = cross_val_predict(pipe, X, y, cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=42), n_jobs=N_FOLDS)

    report = classification_report(y, oof_pred, target_names=EMOTIONS, output_dict=True, zero_division=0)
    metrics = {
        'features': args.features,
        'model': model_type,
        'backbone': args.backbone if args.features == 'embedding' else None,
        'layers': list(layers) if args.features == 'embedding' else None,
        'n_clips': int(len(y)),
        'n_features': int(X.shape[1]),
        'evaluation': f'{N_FOLDS}-fold GroupKFold by actor (speaker-independent)',
        'accuracy': round(float(accuracy_score(y, oof_pred)), 4),
        'accuracy_fold_std': round(float(np.std(fold_acc)), 4),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, oof_pred)), 4),
        'macro_f1': round(float(f1_score(y, oof_pred, average='macro')), 4),
        'accuracy_random_split_leaky': round(float(accuracy_score(y, leaky)), 4),
        'per_class_f1': {e: round(report[e]['f1-score'], 4) for e in EMOTIONS},
        'final_fit_excludes_actors': args.demo_actors,
        'stress_weights': STRESS_WEIGHTS,
    }
    print(classification_report(y, oof_pred, target_names=EMOTIONS, zero_division=0))
    print(f"[train] Speaker-independent accuracy: {metrics['accuracy']:.3f} +/- {metrics['accuracy_fold_std']:.3f} "
          f"(macro-F1 {metrics['macro_f1']:.3f}) | leaky random split: {metrics['accuracy_random_split_leaky']:.3f}")

    # ── Plots from out-of-fold predictions ───────────────────────────────────
    res = Path(args.results) / f'{args.features}_{model_type}'
    res.mkdir(parents=True, exist_ok=True)
    evaluate.plot_confusion_matrix(y, oof_pred, EMOTIONS, res / 'confusion_matrix.png')
    evaluate.plot_roc_curves(y, oof_proba, EMOTIONS, res / 'roc_curves.png')
    evaluate.plot_per_actor_accuracy(actors, y, oof_pred, res / 'per_actor_accuracy.png')

    # ── Final model on all actors except the demo actors ─────────────────────
    keep = ~np.isin(actors, args.demo_actors)
    pipe.fit(X[keep], y[keep])
    if args.features == 'handcrafted':
        evaluate.plot_feature_importance(X, y, actors, FEATURE_NAMES, res / 'feature_importance.png')
    save_bundle(make_bundle(pipe, args.features, metrics, args.backbone if args.features == 'embedding' else None,
                            layers if args.features == 'embedding' else ()), args.out)

    metrics_path = res / 'metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] Metrics written to {metrics_path}")


if __name__ == '__main__':
    main()
