"""
train.py — End-to-end training script.

Run this file to:
  1. Load the RAVDESS dataset (or use cached features)
  2. Split into train/test sets (stratified — keeps class balance in both splits)
  3. Train a chosen sklearn model with cross-validation
  4. Evaluate and save all plots to results/
  5. Save the trained model to models/

Usage:
  python src/train.py --dataset path/to/RAVDESS --model svm
  python src/train.py --dataset path/to/RAVDESS --model rf --force-reload

Arguments:
  --dataset       Path to the RAVDESS root folder (contains Actor_01, Actor_02, ...)
  --model         Model type: svm | rf | gb  (default: svm)
  --test-size     Fraction of data for testing (default: 0.2)
  --force-reload  Ignore cache and re-extract features from audio
  --cache-dir     Where to save/load .npy feature cache (default: data/)
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from features import load_dataset, INT_TO_EMOTION
from model import train_sklearn
from evaluate import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train emotion/stress detection model on RAVDESS.')
    parser.add_argument('--dataset',      type=str, required=True,       help='Path to RAVDESS dataset root')
    parser.add_argument('--model',        type=str, default='svm',       choices=['svm', 'rf', 'gb'])
    parser.add_argument('--test-size',    type=float, default=0.2)
    parser.add_argument('--force-reload', action='store_true')
    parser.add_argument('--cache-dir',    type=str, default='data')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load features ──────────────────────────────────────────────────────
    X, y = load_dataset(
        dataset_path=args.dataset,
        cache_dir=args.cache_dir,
        force_reload=args.force_reload,
    )

    # ── 2. Train / test split (stratified) ───────────────────────────────────
    # stratify=y ensures both splits have the same proportion of each class.
    # Without this, you might accidentally put all 'fearful' clips in test only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )
    print(f"[train] Train: {X_train.shape}  Test: {X_test.shape}")

    # ── 3. Train model ────────────────────────────────────────────────────────
    results = train_sklearn(
        X_train, y_train,
        X_test,  y_test,
        model_type=args.model,
        save_path=f'models/{args.model}_model.pkl',
    )

    model = results['model']
    y_pred = model.predict(X_test)

    # ── 4. Evaluation plots ───────────────────────────────────────────────────
    class_names = [INT_TO_EMOTION[i] for i in range(len(INT_TO_EMOTION))]

    plot_confusion_matrix(y_test, y_pred, class_names=class_names)

    plot_roc_curves(model, X_test, y_test, class_names=class_names)

    if args.model in ('rf', 'gb'):
        plot_feature_importance(model)

    print("\n[train] All done! Check results/ for plots and models/ for the saved model.")


if __name__ == '__main__':
    main()
