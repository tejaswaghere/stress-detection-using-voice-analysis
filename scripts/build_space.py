"""
build_space.py — Assemble a Hugging Face Space from this repo.

The Space is built from the same src/ and app/ code as the repo (never a
hand-edited copy), so the deployed demo can't drift from GitHub.

    python scripts/build_space.py            # writes ./space/
    python scripts/build_space.py --push tejaswaghere/stress-detection   # also uploads (needs `hf auth login`)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gradio

ROOT = Path(__file__).resolve().parent.parent

SPACE_README = """---
title: Speech Emotion & Stress Detector
emoji: 🎙️
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: {short_description}
models:
  - microsoft/wavlm-base-plus
tags:
  - audio-classification
  - speech-emotion-recognition
  - wavlm
---

# 🎙️ Speech Emotion & Stress Detector

Record or upload a short clip. A frozen **WavLM** speech model and a linear classifier
trained on **{data}** ({n_speakers} actors) predict one of {n_classes} emotions ({classes}),
plus a 0–100 vocal stress index.

- **{acc:.0%} accuracy on speakers never seen in training** ({n_classes} classes, 6-fold speaker-grouped CV; chance = {chance:.0%})
- Emotion timeline for recordings longer than 4 s
- API: `POST /analyze` (see "Use via API" at the bottom of the app)

Source, training code and evaluation: https://github.com/tejaswaghere/stress-detection-using-voice-analysis

Example clips are from RAVDESS actors 23 and 24, who were excluded from the final model's training.
RAVDESS — Livingstone & Russo (2018), CC BY-NC-SA 4.0. CREMA-D — Cao et al. (2014), ODbL.
Not a medical or HR tool.
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
transformers>=4.40
librosa==0.10.2.post1
numpy
scikit-learn=={sklearn_version}
joblib
soundfile
matplotlib
"""


def build(out: Path) -> Path:
    import joblib

    bundle = joblib.load(ROOT / 'models' / 'emotion_model.joblib')
    if out.exists():
        shutil.rmtree(out)
    (out / 'src').mkdir(parents=True)
    shutil.copy(ROOT / 'app' / 'app.py', out / 'app.py')
    for f in ['__init__.py', 'features.py', 'embeddings.py', 'gate.py', 'model.py', 'predict.py', 'stress.py']:
        shutil.copy(ROOT / 'src' / f, out / 'src' / f)
    shutil.copytree(ROOT / 'app' / 'examples', out / 'examples')
    (out / 'models').mkdir()
    shutil.copy(ROOT / 'models' / 'emotion_model.joblib', out / 'models' / 'emotion_model.joblib')
    if (ROOT / 'models' / 'speech_gate.joblib').exists():
        shutil.copy(ROOT / 'models' / 'speech_gate.joblib', out / 'models' / 'speech_gate.joblib')
    m = bundle['metrics']
    corpora = m.get('corpora', ['ravdess'])
    names = {'ravdess': ('RAVDESS', 24), 'cremad': ('CREMA-D', 91)}
    short_description = f"Emotion & vocal stress from speech, {m['accuracy']:.0%} on new voices"
    assert len(short_description) <= 60, 'Hugging Face rejects short_description over 60 characters'
    (out / 'README.md').write_text(SPACE_README.format(
        short_description=short_description,
        gradio_version=gradio.__version__, acc=m['accuracy'], n_classes=len(bundle['classes']),
        classes=', '.join(bundle['classes']), chance=1 / len(bundle['classes']),
        data=' + '.join(names[c][0] for c in corpora), n_speakers=sum(names[c][1] for c in corpora),
    ), encoding='utf-8')
    # Pin scikit-learn to the version that pickled the model
    (out / 'requirements.txt').write_text(SPACE_REQUIREMENTS.format(sklearn_version=bundle['sklearn_version']))
    (out / '.gitattributes').write_text('*.joblib filter=lfs diff=lfs merge=lfs -text\n*.wav filter=lfs diff=lfs merge=lfs -text\n')
    print(f"[space] Built {out} (gradio {gradio.__version__}, scikit-learn {bundle['sklearn_version']})")
    return out


def push(out: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(out), repo_id=repo_id, repo_type='space',
                          commit_message='Deploy WavLM model + redesigned app', delete_patterns=['*'])
    print(f"[space] Pushed to https://huggingface.co/spaces/{repo_id}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'space'))
    ap.add_argument('--push', metavar='OWNER/SPACE', help='upload the built folder to this Space')
    args = ap.parse_args()
    built = build(Path(args.out))
    if args.push:
        push(built, args.push)
