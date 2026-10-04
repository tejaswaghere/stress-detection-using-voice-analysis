"""
Speech Emotion & Stress Detector — Gradio app (local or Hugging Face Spaces).

    python app/app.py

All inference goes through src/predict.py, the same code path used to evaluate
the model, so there is no train/serve feature skew.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / 'src').exists() else HERE  # repo layout or flat Space layout
sys.path.insert(0, str(ROOT / 'src'))

from features import SAMPLE_RATE, describe_audio  # noqa: E402
from predict import Predictor  # noqa: E402
from stress import STRESS_WEIGHTS  # noqa: E402

MODEL_PATH = Path(os.environ.get('MODEL_PATH', ROOT / 'models' / 'emotion_model.joblib'))
EXAMPLES_DIR = ROOT / 'app' / 'examples' if (ROOT / 'app' / 'examples').exists() else ROOT / 'examples'
REPO_URL = 'https://github.com/tejaswaghere/stress-detection-using-voice-analysis'

EMOJI = {'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢', 'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'}
COLORS = {'neutral': '#94a3b8', 'calm': '#38bdf8', 'happy': '#facc15', 'sad': '#6366f1', 'angry': '#ef4444',
          'fearful': '#a855f7', 'disgust': '#22c55e', 'surprised': '#fb923c'}
CORPUS_INFO = {'ravdess': ('RAVDESS', 24, 1440), 'cremad': ('CREMA-D', 91, 7441)}
LEVEL_COLORS = {'low': '#16a34a', 'moderate': '#d97706', 'high': '#dc2626'}

predictor = Predictor(MODEL_PATH)
M = predictor.bundle['metrics']
CLASSES = predictor.classes
CORPORA = M.get('corpora', ['ravdess'])
DATA_DESC = ' + '.join(CORPUS_INFO[c][0] for c in CORPORA)
N_SPEAKERS = M.get('n_speakers', sum(CORPUS_INFO[c][1] for c in CORPORA))
N_CLIPS = M.get('n_clips', sum(CORPUS_INFO[c][2] for c in CORPORA))  # clips actually used for training


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ─────────────────────────────────────────────────────────────────────────────

def result_card(pred) -> str:
    e = pred.emotion
    conf = pred.probs[e]
    color = LEVEL_COLORS[pred.stress_level]
    runner_up = sorted(pred.probs, key=pred.probs.get)[-2]
    hedge = '' if conf >= 0.5 else (f'<div class="hedge">Low confidence — could also be '
                                    f'<b>{runner_up}</b> ({pred.probs[runner_up]:.0%})</div>')
    return f"""
    <div class="card">
      <div class="emo">
        <div class="emo-icon">{EMOJI[e]}</div>
        <div><div class="emo-label">{e.capitalize()}</div>
             <div class="emo-conf">{conf:.0%} confidence</div></div>
      </div>
      {hedge}
      <div class="gauge-head"><span>Vocal stress index</span>
        <span style="color:{color};font-weight:700">{pred.stress:.0f}/100 · {pred.stress_level}</span></div>
      <div class="gauge"><div class="gauge-fill" style="width:{pred.stress:.0f}%;background:{color}"></div></div>
      <div class="gauge-desc">{pred.stress_description}</div>
    </div>"""


def analysis_plot(pred, info):
    y = pred.audio
    has_tl = pred.timeline is not None
    fig, axes = plt.subplots(3 if has_tl else 2, 1, figsize=(9, 6.8 if has_tl else 4.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.1, 1.3] if has_tl else [1, 1.1]})
    t = np.arange(len(y)) / SAMPLE_RATE
    axes[0].plot(t, y, lw=0.4, color='#7c3aed')
    axes[0].set_ylabel('waveform')
    axes[0].set_yticks([])

    f0 = info['f0']
    tf = np.arange(len(f0)) * 160 / SAMPLE_RATE
    axes[1].plot(tf, f0, '.', ms=2.5, color='#db2777')
    axes[1].set_ylabel('pitch (Hz)')
    if np.any(np.isfinite(f0)):
        lo, hi = np.nanpercentile(f0, [2, 98])
        axes[1].set_ylim(max(50, lo * 0.8), hi * 1.2)

    if has_tl:
        centres, probs = pred.timeline
        # Each window describes its centre; hold the first/last values out to the clip edges
        centres = np.concatenate([[0], centres, [t[-1]]])
        probs = [probs[0], *probs, probs[-1]]
        stack = np.array([[p[e] for e in CLASSES] for p in probs]).T
        axes[2].stackplot(centres, stack, colors=[COLORS[e] for e in CLASSES], labels=CLASSES, alpha=0.9)
        stress = [100 * sum(STRESS_WEIGHTS[e] * p[e] for e in CLASSES) for p in probs]
        ax2 = axes[2].twinx()
        ax2.plot(centres, stress, 'k-', lw=2, label='stress index')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('stress')
        axes[2].set_ylim(0, 1)
        axes[2].set_ylabel('emotion mix')
        axes[2].legend(loc='upper left', bbox_to_anchor=(1.08, 1), fontsize=8, frameon=False)
    axes[-1].set_xlabel('time (s, silence trimmed)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def acoustics_md(i: dict) -> str:
    pitch = f"{i['pitch_hz']:.0f} Hz" if np.isfinite(i['pitch_hz']) else 'n/a'
    return (
        "| Measure | Value | Why it matters |\n|---|---|---|\n"
        f"| Speech duration | {i['duration_s']:.1f} s | after trimming silence |\n"
        f"| Median pitch | {pitch} | raised pitch accompanies arousal (anger, fear, excitement) |\n"
        f"| Pitch variability | {i['pitch_variability_semitones']:.1f} semitones | flat = calm/sad, wide = expressive |\n"
        f"| Loudness variability | {i['loudness_variability_db']:.1f} dB | bursts of energy mark anger and surprise |\n"
        f"| Voiced ratio | {i['voiced_ratio']:.0%} | share of frames with vocal-fold vibration |\n"
        f"| Spectral centroid | {i['spectral_centroid_hz']:.0f} Hz | 'brightness' — tense voices sound brighter |\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def analyze(audio_path):
    if not audio_path:
        raise gr.Error('Record or upload some audio first.')
    try:
        pred = predictor.predict_file(audio_path)
    except ValueError as e:  # silent / too short
        raise gr.Error(str(e))
    api = {
        'emotion': pred.emotion,
        'confidence': round(pred.probs[pred.emotion], 4),
        'probabilities': {k: round(v, 4) for k, v in pred.probs.items()},
        'stress_index': round(pred.stress, 1),
        'stress_level': pred.stress_level,
    }
    labels = {f'{EMOJI[k]} {k}': v for k, v in pred.probs.items()}
    info = describe_audio(pred.audio)
    return result_card(pred), labels, analysis_plot(pred, info), acoustics_md(info), api


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
.card{border:1px solid var(--border-color-primary);border-radius:14px;padding:18px 20px;background:var(--background-fill-secondary)}
.emo{display:flex;align-items:center;gap:14px;margin-bottom:12px}
.emo-icon{font-size:48px;line-height:1}
.emo-label{font-size:26px;font-weight:700}
.emo-conf{color:var(--body-text-color-subdued)}
.hedge{font-size:13px;color:var(--body-text-color-subdued);margin:-4px 0 12px}
.gauge-head{display:flex;justify-content:space-between;font-size:14px;margin-bottom:6px}
.gauge{height:12px;border-radius:99px;background:var(--border-color-primary);overflow:hidden}
.gauge-fill{height:100%;border-radius:99px;transition:width .6s}
.gauge-desc{font-size:13px;color:var(--body-text-color-subdued);margin-top:6px}
.stats{display:flex;gap:10px;flex-wrap:wrap;justify-content:center;margin-top:6px}
.stat{border:1px solid var(--border-color-primary);border-radius:99px;padding:3px 12px;font-size:13px}
footer{display:none !important}
"""


def header_html() -> str:
    feats = 'WavLM embeddings' if predictor.bundle['feature_type'] == 'embedding' else 'handcrafted acoustic features'
    return f"""
    <div style="text-align:center">
      <h1 style="margin-bottom:4px">🎙️ Speech Emotion &amp; Stress Detector</h1>
      <p style="margin:0;color:var(--body-text-color-subdued)">Say a sentence the way you feel it. The model listens to
      <i>how</i> you speak (pitch, energy, voice quality), not the words.</p>
      <div class="stats">
        <span class="stat">🎯 {M['accuracy']:.0%} accuracy on unseen speakers ({len(CLASSES)} emotions, chance {1 / len(CLASSES):.0%})</span>
        <span class="stat">🧠 {feats} + {M['model']}</span>
        <span class="stat">📚 {DATA_DESC} · {N_SPEAKERS} actors · {N_CLIPS:,} clips</span>
        <span class="stat"><a href="{REPO_URL}" target="_blank">GitHub ↗</a></span>
      </div>
    </div>"""


ABOUT_MD = f"""
### How it works
1. **Preprocess** — resample to 16 kHz, trim silence, normalise volume (so mic gain doesn't look like anger).
2. **Embed** — a frozen, pretrained speech model turns the audio into a vector that captures prosody and voice quality.
3. **Classify** — a small linear classifier trained on {DATA_DESC} outputs probabilities for {len(CLASSES)} emotions
   ({', '.join(CLASSES)}).
4. **Stress index** — the probabilities are combined using arousal/valence weights
   ({', '.join(f'{k} {v:g}' for k, v in STRESS_WEIGHTS.items() if v and k in CLASSES)}).

### Honest limitations
- **Evaluated speaker-independently:** {M['accuracy']:.1%} accuracy / {M['macro_f1']:.2f} macro-F1 on actors never seen in training.
  The same model scores {M['accuracy_random_split_leaky']:.1%} on a random clip split, which leaks speakers — that's
  why the README reports the lower number.
- **Cross-corpus generalisation is the hard part:** a model trained on RAVDESS alone scores 38% (UAR) on CREMA-D, which
  is why this model is trained on both. Your microphone and room are another new "corpus", so expect lower accuracy than above.
- **Acted speech:** both datasets use actors performing fixed sentences in North-American English. Natural,
  subtle speech, other languages, background noise and phone mics are all harder.
- **"Stress" is derived, not measured** — neither dataset has stress labels. Treat the index as an indicator of tense,
  negative-arousal vocal delivery, not a diagnosis. **Not a medical or HR tool.**
- Audio is processed in memory to make the prediction and is not stored by this app.
"""


def build_ui() -> gr.Blocks:
    examples = sorted(EXAMPLES_DIR.glob('*.wav')) if EXAMPLES_DIR.exists() else []
    with gr.Blocks(title='Speech Emotion & Stress Detector') as demo:
        # Outputs are created up front (render=False) so the examples below can target them
        card = gr.HTML('<div class="card" style="color:var(--body-text-color-subdued)">'
                       'Results will appear here.</div>', render=False)
        probs = gr.Label(label='Emotion probabilities', num_top_classes=8, show_heading=False, render=False)
        plot = gr.Plot(label='Waveform · pitch contour · emotion over time (clips > 4 s)', render=False)
        acoustics = gr.Markdown(render=False)
        api_json = gr.JSON(render=False)
        outputs = [card, probs, plot, acoustics, api_json]

        gr.HTML(header_html())
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                audio_in = gr.Audio(sources=['microphone', 'upload'], type='filepath',
                                    label='Record (2–10 s works best) or upload audio')
                btn = gr.Button('Analyze', variant='primary', size='lg')
                # Registered before gr.Examples so this public endpoint gets the name /analyze
                btn.click(analyze, audio_in, outputs, api_name='analyze')
                if examples:
                    gr.Examples(
                        examples=[[str(p)] for p in examples],
                        inputs=audio_in,
                        outputs=outputs,
                        fn=analyze,
                        run_on_click=True,
                        cache_examples=False,
                        api_name='run_example', api_visibility='private',
                        label='Try a RAVDESS clip (actors 23 & 24, held out of training)',
                        example_labels=[p.stem.replace('_', ' ') for p in examples],
                    )
                gr.Markdown('💡 *Try saying "I can\'t believe this is happening" angrily, then calmly.*')
            with gr.Column(scale=6):
                card.render()
                probs.render()
        with gr.Row():
            with gr.Column(scale=7):
                plot.render()
            with gr.Column(scale=4):
                acoustics.render()
        with gr.Accordion('API response (JSON)', open=False):
            api_json.render()
        with gr.Accordion('How it works & limitations', open=False):
            gr.Markdown(ABOUT_MD)

        audio_in.stop_recording(analyze, audio_in, outputs, api_name='on_record', api_visibility='private')
        audio_in.upload(analyze, audio_in, outputs, api_name='on_upload', api_visibility='private')
    return demo


demo = build_ui()

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=2).launch(
        server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)),
        theme=gr.themes.Soft(primary_hue='violet'), css=CSS,
    )
