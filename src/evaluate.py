"""
evaluate.py — Evaluation plots, all computed from speaker-independent
out-of-fold predictions (see train.py).

  confusion_matrix.png    which emotions get confused with which
  roc_curves.png          one-vs-rest ROC per emotion
  per_actor_accuracy.png  how much accuracy varies between unseen speakers
  feature_importance.png  permutation importance (handcrafted features only)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
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


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """Row-normalised confusion matrix: row = true emotion, column = predicted."""
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, linewidths=0.5, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix — speaker-independent (unseen actors)', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
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


def plot_per_actor_accuracy(actors, y_true, y_pred, save_path='results/per_actor_accuracy.png'):
    """Accuracy for each actor when that actor was held out. Odd IDs are male, even are female."""
    ids = np.unique(actors)
    acc = np.array([np.mean(y_pred[actors == a] == y_true[actors == a]) for a in ids])
    colors = ['#4C72B0' if a % 2 else '#DD8452' for a in ids]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(a) for a in ids], acc, color=colors)
    ax.axhline(acc.mean(), color='k', ls='--', lw=1, label=f'mean {acc.mean():.2f}')
    ax.axhline(1 / 8, color='grey', ls=':', lw=1, label='chance (0.125)')
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel('Actor (blue = male, orange = female)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per held-out actor', fontweight='bold')
    ax.legend(loc='upper center', ncol=2, frameon=False)
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
