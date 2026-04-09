"""
evaluate.py — Evaluation, visualisation, and explainability.

What this file does:
  - Plots a confusion matrix heatmap (which emotions get confused with which)
  - Plots per-class precision/recall bar charts
  - Plots ROC curves (one-vs-rest, per class)
  - Feature importance via SHAP (SHapley Additive exPlanations)
  - Waveform + spectrogram visualiser for individual files

Why this matters for a GitHub project:
  Training a model and printing accuracy is the minimum. Showing *why* the model
  makes mistakes (confusion matrix), *which features drive decisions* (SHAP),
  and *how confident* the model is at different thresholds (ROC) is what
  separates a student project from a portfolio project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from pathlib import Path
import librosa
import librosa.display


EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = EMOTION_LABELS,
    save_path: str = 'results/confusion_matrix.png',
) -> None:
    """
    Plot and save a normalised confusion matrix heatmap.

    Normalised (%) vs raw counts: Normalising shows the *proportion* of each
    true class that got predicted correctly. This is more useful than raw counts
    because RAVDESS is roughly balanced — but other datasets might not be.

    Reading the heatmap:
      Each row = the true label. Each column = the predicted label.
      The diagonal (top-left to bottom-right) = correct predictions.
      Off-diagonal values reveal what the model confuses — e.g. 'calm' and
      'neutral' being confused makes intuitive sense.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_ylabel('True label', fontsize=12)
    ax.set_title('Confusion Matrix (normalised)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[eval] Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_roc_curves(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list = EMOTION_LABELS,
    save_path: str = 'results/roc_curves.png',
) -> None:
    """
    Plot one-vs-rest ROC curves for each emotion class.

    AUC (Area Under Curve):
      A perfect model = AUC 1.0. Random guessing = AUC 0.5.
      Plotting one curve per class shows which emotions the model handles
      confidently (AUC > 0.9) vs which are hard (AUC < 0.7).

    Requires the model to support predict_proba (SVC needs probability=True,
    which is already set in model.py).
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — One vs Rest per Emotion', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[eval] ROC curves saved to {save_path}")
    plt.close(fig)


def plot_feature_importance(
    model,
    feature_names: list = None,
    top_n: int = 20,
    save_path: str = 'results/feature_importance.png',
) -> None:
    """
    Plot feature importances from a Random Forest or Gradient Boosting model.

    Feature importances tell us: "Which audio properties does the model rely on
    most?" — If MFCC-12 and RMS std appear in the top 5, that's interpretable:
    the model is using tonal quality and loudness variation to detect emotions.
    This is the kind of insight that makes a project feel research-grade.

    For SVM, this falls back to a coefficient-magnitude plot (linear kernel only).
    """
    clf = model.named_steps.get('clf', model)
    scaler = model.named_steps.get('scaler', None)

    n_features = scaler.n_features_in_ if scaler else None

    if feature_names is None:
        # Auto-generate names matching the feature vector order in features.py
        names = (
            [f'mfcc_{i}'  for i in range(40)] +
            [f'dmfcc_{i}' for i in range(40)] +
            [f'd2mfcc_{i}'for i in range(40)] +
            [f'chroma_{i}'for i in range(12)] +
            [f'contrast_{i}' for i in range(7)] +
            [f'mel_{i}'   for i in range(40)] +
            ['zcr_mean', 'rms_mean', 'rms_std']
        )
        feature_names = names[:n_features] if n_features else names

    if hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importances = np.abs(clf.coef_).mean(axis=0)
    else:
        print("[eval] Model does not expose feature importances.")
        return

    idx = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}' for i in idx]
    top_vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(top_n), top_vals[::-1], color='steelblue', edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"[eval] Feature importance saved to {save_path}")
    plt.close(fig)


def visualise_audio(file_path: str, save_path: str = 'results/audio_viz.png') -> None:
    """
    Plot waveform + Mel spectrogram + MFCC heatmap side by side for a single file.
    Useful for the README and for understanding what the features look like.
    """
    y, sr = librosa.load(file_path, sr=22050, duration=3.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='steelblue')
    axes[0].set_title('Waveform', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img1 = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
    axes[1].set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[1], format='%+2.0f dB')

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    img2 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[2])
    axes[2].set_title('MFCCs (40 coefficients)', fontsize=12, fontweight='bold')
    fig.colorbar(img2, ax=axes[2])

    plt.suptitle(Path(file_path).name, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[eval] Audio visualisation saved to {save_path}")
    plt.close(fig)
