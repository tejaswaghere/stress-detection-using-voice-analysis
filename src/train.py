"""
train.py — Train and evaluate an emotion classifier on RAVDESS and/or CREMA-D.

Evaluation is speaker-independent: 6-fold GroupKFold over speakers, so every test clip comes
from speakers the model never heard during training. (A random clip-level split lets the same
speaker appear in train and test and inflates accuracy — that number is also reported, labelled.)
The out-of-fold predictions drive every plot in results/<run>/. The final model is then refit
on every speaker except RAVDESS actors 23 and 24, whose clips are the app's examples, so those
examples are genuinely unseen.

Label sets: RAVDESS alone uses its 8 emotions. Any run that includes CREMA-D uses the 6 emotions
both corpora share. CREMA-D has no 'calm' or 'surprised', and training those from RAVDESS alone
teaches "RAVDESS recording conditions → calm": a 7-class pooled model predicted 'calm' for 0.0% of
CREMA-D clips. So RAVDESS 'calm' is merged into 'neutral' (both mean "no vocal stress"), and
'surprised' is dropped. Without the merge, 61% of unseen calm clips scored as moderately stressed
(mostly read as 'sad'); with it, 18%.

Usage:
  python src/train.py                                      # WavLM + logreg on RAVDESS + CREMA-D (6 emotions)
  python src/train.py --corpora ravdess                    # RAVDESS only (8 emotions)
  python src/train.py --corpora ravdess --features handcrafted --model svm   # no torch needed

See src/cross_corpus.py for train-on-one, test-on-the-other experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))

from corpora import DEFAULT_ROOTS, SHARED_EMOTIONS, load_features  # noqa: E402
from features import EMOTIONS, FEATURE_NAMES  # noqa: E402
from model import CLASSIFIERS, build_pipeline, make_bundle, save_bundle  # noqa: E402
from stress import STRESS_WEIGHTS  # noqa: E402
import evaluate  # noqa: E402

N_FOLDS = 6
DEFAULT_LAYERS = (4, 5, 6, 7)  # middle layers encode prosody best — see results/layer_probe.json
DEMO_SPEAKERS = ['ravdess:23', 'ravdess:24']
POOLED_MERGE = {'calm': 'neutral'}  # see module docstring


def load(corpora: list[str], features: str, roots: dict, cache_dir: str, backbone: str, layers, force: bool):
    single = corpora == ['ravdess']
    labels = EMOTIONS if single else SHARED_EMOTIONS
    merge = {} if single else POOLED_MERGE
    Xs, clips = [], []
    for c in corpora:
        X, cl = load_features(c, features, roots[c], cache_dir, backbone, layers, force)
        keep = np.array([merge.get(x.emotion, x.emotion) in labels for x in cl])
        Xs.append(X[keep])
        clips += [x for x, k in zip(cl, keep) if k]
    X = np.vstack(Xs)
    y = np.array([labels.index(merge.get(c.emotion, c.emotion)) for c in clips])
    return X, y, clips, labels


def parse_args():
    p = argparse.ArgumentParser(description='Train an emotion classifier (speaker-independent evaluation).')
    p.add_argument('--corpora', nargs='+', choices=['ravdess', 'cremad'], default=['ravdess', 'cremad'])
    p.add_argument('--dataset', default=DEFAULT_ROOTS['ravdess'], help='RAVDESS root (Actor_01 ... Actor_24)')
    p.add_argument('--cremad', default=DEFAULT_ROOTS['cremad'], help='CREMA-D root (contains AudioWAV/)')
    p.add_argument('--features', choices=['embedding', 'handcrafted'], default='embedding')
    p.add_argument('--model', choices=CLASSIFIERS, default=None, help='default: logreg for embeddings, svm for handcrafted')
    p.add_argument('--C', type=float, default=None, help='regularisation strength override')
    p.add_argument('--backbone', default='microsoft/wavlm-base-plus')
    p.add_argument('--layers', type=int, nargs='+', default=list(DEFAULT_LAYERS))
    p.add_argument('--cache-dir', default='data')
    p.add_argument('--out', default='models/emotion_model.joblib')
    p.add_argument('--results', default='results')
    p.add_argument('--force-reload', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    corpora = sorted(set(args.corpora), key=['ravdess', 'cremad'].index)
    model_type = args.model or ('logreg' if args.features == 'embedding' else 'svm')
    layers = tuple(args.layers)
    roots = {'ravdess': args.dataset, 'cremad': args.cremad}

    X, y, clips, labels = load(corpora, args.features, roots, args.cache_dir, args.backbone, layers, args.force_reload)
    speakers = np.array([c.speaker for c in clips])
    sex = np.array([c.sex for c in clips])
    source = np.array([c.corpus for c in clips])
    print(f"[train] {'+'.join(corpora)}: {len(y)} clips, {X.shape[1]} features, "
          f"{len(np.unique(speakers))} speakers, {len(labels)} emotions")

    # ── Speaker-independent cross-validation ─────────────────────────────────
    pipe = build_pipeline(model_type, args.C)
    cv = GroupKFold(N_FOLDS)
    oof_proba = cross_val_predict(pipe, X, y, groups=speakers, cv=cv, method='predict_proba', n_jobs=N_FOLDS)
    oof_pred = oof_proba.argmax(1)
    fold_acc = [accuracy_score(y[test], oof_pred[test]) for _, test in cv.split(X, y, speakers)]

    # Same model, leaky random split — reported only to show the gap
    leaky = cross_val_predict(pipe, X, y, cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=42), n_jobs=N_FOLDS)

    report = classification_report(y, oof_pred, target_names=labels, output_dict=True, zero_division=0)
    metrics = {
        'corpora': corpora,
        'emotions': labels,
        'features': args.features,
        'model': model_type,
        'backbone': args.backbone if args.features == 'embedding' else None,
        'layers': list(layers) if args.features == 'embedding' else None,
        'n_clips': int(len(y)),
        'n_speakers': int(len(np.unique(speakers))),
        'n_features': int(X.shape[1]),
        'evaluation': f'{N_FOLDS}-fold GroupKFold by speaker (speaker-independent)',
        'accuracy': round(float(accuracy_score(y, oof_pred)), 4),
        'accuracy_fold_std': round(float(np.std(fold_acc)), 4),
        'uar': round(float(balanced_accuracy_score(y, oof_pred)), 4),
        'macro_f1': round(float(f1_score(y, oof_pred, average='macro')), 4),
        'accuracy_random_split_leaky': round(float(accuracy_score(y, leaky)), 4),
        'per_class_f1': {e: round(report[e]['f1-score'], 4) for e in labels},
        'per_corpus': {c: {'accuracy': round(float(accuracy_score(y[source == c], oof_pred[source == c])), 4),
                           'uar': round(float(balanced_accuracy_score(y[source == c], oof_pred[source == c])), 4)}
                       for c in corpora},
        'uar_by_sex': {s: round(float(balanced_accuracy_score(y[sex == s], oof_pred[sex == s])), 4)
                       for s in ('male', 'female') if (sex == s).any()},
        'label_merge': {} if corpora == ['ravdess'] else POOLED_MERGE,
        'final_fit_excludes_speakers': DEMO_SPEAKERS,
        'stress_weights': {e: w for e, w in STRESS_WEIGHTS.items() if e in labels},
    }
    print(classification_report(y, oof_pred, target_names=labels, zero_division=0))
    print(f"[train] Speaker-independent accuracy: {metrics['accuracy']:.3f} +/- {metrics['accuracy_fold_std']:.3f} "
          f"(UAR {metrics['uar']:.3f}, macro-F1 {metrics['macro_f1']:.3f}) | leaky random split: "
          f"{metrics['accuracy_random_split_leaky']:.3f} | per corpus: {metrics['per_corpus']}")

    # ── Plots from out-of-fold predictions ───────────────────────────────────
    run = f'{args.features}_{model_type}' + ('' if corpora == ['ravdess'] else '_' + '+'.join(corpora))
    res = Path(args.results) / run
    res.mkdir(parents=True, exist_ok=True)
    evaluate.plot_confusion_matrix(y, oof_pred, labels, res / 'confusion_matrix.png')
    evaluate.plot_roc_curves(y, oof_proba, labels, res / 'roc_curves.png')
    evaluate.plot_per_speaker_accuracy(speakers, sex, y, oof_pred, res / 'per_speaker_accuracy.png',
                                       chance=1 / len(labels))

    # ── Final model on all speakers except the demo speakers ────────────────
    keep = ~np.isin(speakers, DEMO_SPEAKERS)
    pipe.fit(X[keep], y[keep])
    if args.features == 'handcrafted':
        evaluate.plot_feature_importance(X, y, speakers, FEATURE_NAMES, res / 'feature_importance.png')
    bundle = make_bundle(pipe, args.features, metrics, args.backbone if args.features == 'embedding' else None,
                         layers if args.features == 'embedding' else ())
    bundle['classes'] = list(labels)
    save_bundle(bundle, args.out)

    (res / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"[train] Metrics written to {res / 'metrics.json'}")


if __name__ == '__main__':
    main()
