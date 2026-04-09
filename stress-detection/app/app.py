"""
app.py — Gradio web demo for emotion/stress detection.

What this does:
  - Provides a browser UI where anyone can upload or record a .wav file
  - Runs the trained sklearn pipeline on the audio
  - Displays the predicted emotion + a bar chart of all class probabilities
  - Can be deployed to HuggingFace Spaces for a public live demo link

How to run:
  pip install gradio
  python app/app.py

How to deploy to HuggingFace Spaces (free):
  1. Create an account at huggingface.co
  2. New Space → Gradio → name it "stress-emotion-detection"
  3. Push this repo (or just app.py + models/ + requirements.txt) to the Space
  4. HuggingFace auto-runs it and gives you a public URL like:
     https://huggingface.co/spaces/yourname/stress-emotion-detection

Why Gradio?
  Gradio builds a full web UI in ~20 lines of Python. No HTML/JS needed.
  The 'microphone' input type lets users record directly in the browser.
  Adding a live demo link to your README is the single biggest signal to
  internship reviewers that you actually finished the project.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no display needed)
import librosa
import joblib
import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from features import extract_features, INT_TO_EMOTION

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'svm_model.pkl'

try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"[app] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"[app] WARNING: No model at {MODEL_PATH}. Run train.py first.")
    pipeline = None


EMOTION_LABELS = list(INT_TO_EMOTION.values())

# Colour palette for the bar chart (one per emotion)
EMOTION_COLORS = [
    '#4C9BE8', '#6EC6A0', '#F5A623', '#E8594A',
    '#9B59B6', '#2ECC71', '#E67E22', '#1ABC9C',
]


def predict_emotion(audio_path: str) -> tuple:
    """
    Given a path to an audio file, return (emotion label, confidence bar chart).

    Steps:
      1. Extract feature vector using the same function used during training
         (critical: same feature order, same hyperparameters)
      2. Run the sklearn pipeline (which applies StandardScaler internally)
      3. Get class probabilities with predict_proba
      4. Return the top prediction + a matplotlib bar chart

    Returns
    -------
    label : str      — e.g. "angry (87.3% confidence)"
    fig   : Figure   — matplotlib bar chart of all class probabilities
    """
    if pipeline is None:
        return "Model not loaded. Run train.py first.", None

    try:
        feat = extract_features(audio_path)
        feat = feat.reshape(1, -1)

        proba = pipeline.predict_proba(feat)[0]        # shape (8,)
        pred_idx = int(np.argmax(proba))
        pred_label = INT_TO_EMOTION[pred_idx]
        confidence = proba[pred_idx] * 100

        label_str = f"🎯  {pred_label.upper()}  ({confidence:.1f}% confidence)"

        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(
            EMOTION_LABELS,
            proba * 100,
            color=[EMOTION_COLORS[i] if i == pred_idx else '#D0D0D0' for i in range(len(EMOTION_LABELS))],
            edgecolor='white',
            height=0.6,
        )
        ax.set_xlabel('Probability (%)', fontsize=11)
        ax.set_title('Emotion Confidence Scores', fontsize=13, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.axvline(x=proba[pred_idx] * 100, color='#333', linestyle='--', linewidth=1, alpha=0.4)

        for bar, val in zip(bars, proba * 100):
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)

        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        return label_str, fig

    except Exception as e:
        return f"Error processing audio: {e}", None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(
        sources=['upload', 'microphone'],
        type='filepath',
        label='Upload a .wav file or record from microphone',
    ),
    outputs=[
        gr.Textbox(label='Prediction', lines=1),
        gr.Plot(label='Confidence Scores'),
    ],
    title='🎙️ Emotion & Stress Detection from Voice',
    description=(
        'Upload a short voice recording (3–5 seconds) or record directly.\n'
        'The model detects: Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised\n\n'
        'Trained on the **RAVDESS** dataset using MFCCs, delta-MFCCs, Chroma, Spectral Contrast, and Mel Spectrogram features.'
    ),
    examples=[],      # Add example .wav paths here once you have them
    theme=gr.themes.Soft(),
    allow_flagging='never',
)

if __name__ == '__main__':
    demo.launch(share=True)   # share=True generates a temporary public URL
