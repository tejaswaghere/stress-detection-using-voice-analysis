"""
evaluate.py — Evaluation plots, all computed from speaker-independent
out-of-fold predictions (see train.py).

  confusion_matrix.png    which emotions get confused with which
  roc_curves.png          one-vs-rest ROC per emotion
  per_speaker_accuracy.png how much accuracy varies between unseen speakers
  feature_importance.png  permutation importance (handcrafted features only)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Saved {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png',
                          title='Confusion matrix — speaker-independent (unseen actors)'):
    """Row-normalised confusion matrix: row = true emotion, column = predicted."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _save(fig, save_path)


def plot_transfer_matrix(M, row_labels, col_labels, title, save_path, chance=None):
    """Heatmap of a train-corpus × test-corpus score matrix."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.heatmap(M, annot=True, fmt='.2f', cmap='Purples', vmin=chance or 0, vmax=1, cbar_kws={'label': 'UAR'},
                xticklabels=col_labels, yticklabels=row_labels, linewidths=1, ax=ax)
    ax.set_xlabel('Test corpus (unseen speakers)')
    ax.set_ylabel('Train corpus')
    ax.set_title(title + (f'\nchance = {chance:.2f}' if chance else ''), fontweight='bold', fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0)
    _save(fig, save_path)


def plot_roc_curves(y_true, y_proba, class_names, save_path='results/roc_curves.png'):
    """One-vs-rest ROC curve per emotion."""
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f'{name} (AUC {auc(fpr, tpr):.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='chance')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC — one-vs-rest, unseen actors', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, save_path)


def plot_per_speaker_accuracy(speakers, sex, y_true, y_pred, save_path='results/per_speaker_accuracy.png', chance=None):
    """Accuracy for each speaker when that speaker was held out, sorted, coloured by sex."""
    ids = np.unique(speakers)
    acc = np.array([np.mean(y_pred[speakers == s] == y_true[speakers == s]) for s in ids])
    sx = np.array([sex[speakers == s][0] for s in ids])
    order = np.argsort(acc)
    ids, acc, sx = ids[order], acc[order], sx[order]
    palette = {'male': '#4C72B0', 'female': '#DD8452'}
    fig, ax = plt.subplots(figsize=(10 if len(ids) <= 30 else 12, 4))
    ax.bar(range(len(ids)), acc, color=[palette.get(s, '#999999') for s in sx], width=0.85)
    handles = [Patch(color=c, label=f'{s} (mean {acc[sx == s].mean():.2f})') for s, c in palette.items() if (sx == s).any()]
    handles.append(ax.axhline(acc.mean(), color='k', ls='--', lw=1, label=f'all speakers (mean {acc.mean():.2f})'))
    if chance:
        handles.append(ax.axhline(chance, color='grey', ls=':', lw=1, label=f'chance ({chance:.3f})'))
    if len(ids) <= 30:
        ax.set_xticks(range(len(ids)), [i.split(':')[-1] for i in ids])
        ax.set_xlabel('Speaker (sorted by accuracy)')
    else:
        ax.set_xticks([])
        ax.set_xlabel(f'{len(ids)} speakers, sorted by accuracy')
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per held-out speaker', fontweight='bold')
    ax.legend(handles=handles, loc='upper left', ncol=2, frameon=False, fontsize=9)
    _save(fig, save_path)


def plot_feature_importance(X, y, actors, feature_names, save_path='results/feature_importance.png', top_n=20):
    """
    Permutation importance of handcrafted features, measured with a random
    forest trained on 18 actors and scored on 6 held-out actors — i.e. how much
    accuracy on unseen speakers drops when a feature is shuffled.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupShuffleSplit

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, actors))
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
    imp = permutation_importance(rf, X_te, y_te, n_repeats=5, random_state=0, n_jobs=-1).importances_mean
    idx = np.argsort(imp)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], imp[idx], color='steelblue')
    ax.set_xlabel('Accuracy drop when shuffled')
    ax.set_title(f'Top {top_n} handcrafted features (permutation importance)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    _save(fig, save_path)
