"""
model.py — Classifier definitions and model-bundle persistence.

A saved model is a "bundle": a dict holding the fitted scikit-learn pipeline
plus everything needed to reproduce its inputs (feature type, backbone and
layers for embeddings, feature version, class order) and its evaluation
metrics. Loading a bundle whose feature version doesn't match the code fails
loudly instead of silently producing garbage predictions.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import EMOTIONS, FEATURE_VERSION

CLASSIFIERS = ('logreg', 'svm', 'rf')


def build_pipeline(model_type: str = 'logreg', C: float | None = None) -> Pipeline:
    """
    StandardScaler → classifier.

      logreg — L2 logistic regression. Best on high-dimensional embeddings;
               well-calibrated probabilities, tiny on disk.
      svm    — RBF SVM. Strong on the handcrafted features.
      rf     — Random forest. Weaker, but exposes feature importances.
    """
    if model_type == 'logreg':
        clf = LogisticRegression(C=C or 0.01, max_iter=3000, class_weight='balanced')
    elif model_type == 'svm':
        clf = SVC(kernel='rbf', C=C or 1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"model_type must be one of {CLASSIFIERS}")
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def make_bundle(pipeline: Pipeline, feature_type: str, metrics: dict, backbone: str | None = None,
                layers: tuple[int, ...] = ()) -> dict:
    return {
        'pipeline': pipeline,
        'feature_type': feature_type,  # 'handcrafted' | 'embedding'
        'feature_version': FEATURE_VERSION if feature_type == 'handcrafted' else f'{backbone}:{list(layers)}',
        'backbone': backbone,
        'layers': list(layers),
        'classes': list(EMOTIONS),
        'sklearn_version': sklearn.__version__,
        'metrics': metrics,
    }


def save_bundle(bundle: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    print(f"[model] Saved model bundle to {path}")


def load_bundle(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Train one with: python src/train.py --dataset data/RAVDESS")
    bundle = joblib.load(path)
    if bundle['feature_type'] == 'handcrafted' and bundle['feature_version'] != FEATURE_VERSION:
        raise RuntimeError(f"{path} was trained on features '{bundle['feature_version']}', "
                           f"but the code extracts '{FEATURE_VERSION}'. Retrain the model.")
    return bundle
