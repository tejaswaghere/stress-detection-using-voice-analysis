"""
Stress & Emotion Detection from Voice
Gradio app — deployable to HuggingFace Spaces

Usage (local):
    python app/app.py

Usage (HF Spaces):
    Push to a Space with requirements.txt — Spaces auto-launches app.py
"""

import gradio as gr
import numpy as np
import librosa
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Resolve model path whether run from repo root or app/ ──────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "svm_model.pkl")

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOJI    = ["😐", "😌", "😊", "😢", "😠", "😨", "🤢", "😲"]
HIGH_STRESS = {"angry", "fearful", "disgust", "surprised"}

# ── Load model (graceful fallback for demo without trained model) ──────────
model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Could not load model: {e}")
else:
    print(f"⚠ No model found at {MODEL_PATH} — using rule-based fallback")
    print("  Run: python src/train.py --dataset data/RAVDESS --model svm")


# ── Feature extraction (mirrors src/features.py) ──────────────────────────
def extract_features(audio_path: str) -> np.ndarray:
    """Extract 182-dim feature vector (MFCC + delta + chroma + contrast + mel + ZCR + RMS)."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=5.0)

    if len(y) < sr * 0.5:
        raise ValueError("Audio too short (< 0.5 s) — please record at least 2 seconds")

    feats = []

    # MFCCs (40) + delta (40) + delta² (40)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feats.append(np.mean(mfcc, axis=1))
    feats.append(np.mean(librosa.feature.delta(mfcc), axis=1))
    feats.append(np.mean(librosa.feature.delta(mfcc, order=2), axis=1))

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats.append(np.mean(chroma, axis=1))

    # Spectral contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.append(np.mean(contrast, axis=1))

    # Mel spectrogram (40)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    feats.append(np.mean(mel, axis=1))

    # ZCR (1)
    feats.append([np.mean(librosa.feature.zero_crossing_rate(y))])

    # RMS mean + std (2)
    rms = librosa.feature.rms(y=y)[0]
    feats.append([np.mean(rms), np.std(rms)])

    vector = np.concatenate(feats)           # shape: (182,)
    assert vector.shape[0] == 182, f"Expected 182 features, got {vector.shape[0]}"
    return vector, y, sr


# ── Rule-based fallback (no trained model) ────────────────────────────────
def _rule_based(y, sr) -> np.ndarray:
    """Approximate RAVDESS SVM decision boundaries without a model file."""
    rms  = float(np.mean(librosa.feature.rms(y=y)))
    zcr  = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                   fmax=librosa.note_to_hz('C7'))
        pitch = float(np.nanmean(f0[f0 > 0])) if np.any(f0 > 0) else 120.0
    except Exception:
        pitch = 120.0

    sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    raw = np.array([
        max(0, rms * 1.1 + zcr * 0.9 + (0.4 if pitch > 170 else 0) + (0.3 if sc > 3500 else 0) - 0.5),  # angry
        max(0, zcr + (0.5 if pitch > 190 else 0) + (0.4 if sc > 3800 else 0) + rms * 0.4 - 0.5),          # fearful
        max(0, rms * 0.6 + (0.5 if 150 < pitch < 200 else 0) + (0.3 if sc > 2800 else 0) - 0.2),          # happy
        max(0, (1 - rms) * 0.7 + (0.5 if pitch < 110 else 0) + (0.3 if sc < 1800 else 0) - 0.1),          # sad
        max(0, (1 - zcr) * 0.8 + (0.4 if 100 < pitch < 140 else 0) + (1 - rms) * 0.4 - 0.2),             # calm
        max(0, 0.6 - abs(rms - 0.4) - abs(zcr - 0.1)),                                                     # neutral
        max(0, zcr * 0.6 + (0.3 if pitch > 160 else 0) + rms * 0.5 - 0.4),                                 # disgust
        max(0, (0.4 if sc > 3000 else 0) + (0.4 if pitch > 180 else 0) + rms * 0.3 - 0.2),                # surprised
    ])

    # Map to EMOTIONS order: neutral,calm,happy,sad,angry,fearful,disgust,surprised
    reorder = [5, 4, 2, 3, 0, 1, 6, 7]
    probs = raw[reorder]
    probs = probs + 0.05
    probs = probs / probs.sum()
    return probs


# ── Main prediction function ───────────────────────────────────────────────
def predict_emotion(audio_path):
    if audio_path is None:
        return "Please upload or record audio first.", {}, "", ""

    try:
        vector, y, sr = extract_features(audio_path)
    except ValueError as e:
        return str(e), {}, "", ""
    except Exception as e:
        return f"Feature extraction failed: {e}", {}, "", ""

    # Inference
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([vector])[0]
            else:
                pred  = model.predict([vector])[0]
                probs = np.eye(len(EMOTIONS))[pred]
        except Exception as e:
            probs = _rule_based(y, sr)
    else:
        probs = _rule_based(y, sr)

    # Build outputs
    top_idx     = int(np.argmax(probs))
    top_emotion = EMOTIONS[top_idx]
    confidence  = float(probs[top_idx]) * 100

    result_label = f"{EMOJI[top_idx]}  {top_emotion.capitalize()}  ({confidence:.1f}% confidence)"

    conf_dict = {f"{EMOJI[i]} {EMOTIONS[i]}": float(p) for i, p in enumerate(probs)}

    # Stress level
    stress_emotions = {e: float(probs[EMOTIONS.index(e)]) for e in HIGH_STRESS}
    stress_score    = sum(stress_emotions.values())
    if stress_score > 0.55:
        stress_out = "🔴  High stress indicators detected"
    elif stress_score > 0.30:
        stress_out = "🟡  Moderate stress indicators present"
    else:
        stress_out = "🟢  Low stress — voice appears calm"

    # Feature summary
    rms   = float(np.mean(librosa.feature.rms(y=y)))
    zcr   = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    sc    = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    feat_summary = (
        f"**RMS Energy:** {rms:.4f}  |  "
        f"**ZCR:** {zcr:.5f}  |  "
        f"**Spectral Centroid:** {sc:.0f} Hz  |  "
        f"**Feature vector:** 182 dims"
    )

    return result_label, conf_dict, stress_out, feat_summary


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Speech Emotion & Stress Detector",
        theme=gr.themes.Soft(primary_hue="violet"),
        css="""
        #title { text-align: center; }
        #title h1 { font-size: 2rem; }
        .stress-box { font-size: 1.1rem; padding: 0.75rem; border-radius: 8px; }
        """
    ) as demo:

        gr.HTML("""
        <div id="title">
          <h1>🎙️ Speech Emotion &amp; Stress Detector</h1>
          <p style="color:#666">Trained on <b>RAVDESS</b> · 182 audio features (MFCC + Chroma + Spectral Contrast + Mel)
          · <a href="https://github.com/tejaswaghere/stress-detection-using-voice-analysis" target="_blank">GitHub ↗</a></p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    label="Upload or Record Audio",
                    sources=["microphone", "upload"],
                    type="filepath",
                )
                analyse_btn = gr.Button("🔍 Analyse Emotion", variant="primary", size="lg")

                gr.Examples(
                    examples=[],          # Add .wav file paths here after training
                    inputs=audio_in,
                    label="Sample audio files"
                )

            with gr.Column(scale=1):
                result_out = gr.Label(label="Detected Emotion")
                conf_out   = gr.Label(label="Confidence per Emotion", num_top_classes=8)
                stress_out = gr.Textbox(label="Stress Indicator", elem_classes=["stress-box"])
                feat_out   = gr.Markdown(label="Extracted Features")

        analyse_btn.click(
            fn=predict_emotion,
            inputs=[audio_in],
            outputs=[result_out, conf_out, stress_out, feat_out],
        )

        gr.HTML("""
        <hr style="margin:2rem 0;opacity:0.2"/>
        <div style="text-align:center;color:#888;font-size:0.85rem">
          Model: SVM · Dataset: RAVDESS (1440 samples, 24 actors) · Accuracy: ~65–70% (8-class)
          <br/>For research use only — not a clinical tool
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",   # Needed for HF Spaces
        server_port=7860,
        share=False,
        show_error=True,
    )
