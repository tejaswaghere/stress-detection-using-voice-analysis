"""
predict.py — Inference shared by the CLI and the Gradio app.

Usage:
  python src/predict.py path/to/clip.wav [--model models/emotion_model.joblib]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from features import extract_features, load_audio, preprocess  # noqa: E402
from gate import DEFAULT_GATE, SpeechGate  # noqa: E402
from model import load_bundle  # noqa: E402
from stress import stress_index, stress_level  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / 'models' / 'emotion_model.joblib'


@dataclass
class Prediction:
    probs: dict[str, float]
    stress: float
    stress_level: str
    stress_description: str
    audio: np.ndarray
    timeline: tuple[np.ndarray, list[dict[str, float]]] | None = field(default=None)
    speech_probability: float | None = None

    @property
    def emotion(self) -> str:
        return max(self.probs, key=self.probs.get)


class Predictor:
    """
    Emotion + stress prediction. With an embedding model and models/speech_gate.joblib present,
    clips without speech raise NoSpeechError (a ValueError) instead of getting a made-up emotion.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, gate_path: str | Path | None = DEFAULT_GATE):
        self.bundle = load_bundle(model_path)
        self.pipeline = self.bundle['pipeline']
        self.classes = self.bundle['classes']
        self.embedder = None
        self.gate = None
        if self.bundle['feature_type'] == 'embedding':
            from embeddings import SSLEmbedder
            self.embedder = SSLEmbedder(self.bundle['backbone'], tuple(self.bundle['layers']))
            if gate_path and Path(gate_path).exists():
                self.gate = SpeechGate(gate_path)
                if (self.gate.backbone, list(self.gate.layers)) != (self.bundle['backbone'], list(self.bundle['layers'])):
                    raise RuntimeError('Speech gate and emotion model were trained on different embeddings.')

    def _probs(self, X: np.ndarray) -> list[dict[str, float]]:
        P = self.pipeline.predict_proba(X)
        return [{self.classes[c]: float(p) for c, p in zip(self.pipeline.classes_, row)} for row in P]

    def predict_audio(self, y: np.ndarray, timeline: bool = True) -> Prediction:
        """Predict from *preprocessed* 16 kHz audio."""
        tl, speech_p = None, None
        if self.embedder is not None:
            x = self.embedder(y)[None]
            if self.gate is not None:
                speech_p = self.gate.check(x[0])  # raises NoSpeechError for music / noise / non-speech
            if timeline and len(y) > 16000 * 4:
                centres, W = self.embedder.windows(y)
                tl = (centres, self._probs(W))
        else:
            x = extract_features(y)[None]
        probs = self._probs(x)[0]
        s = stress_index(probs)
        level, desc = stress_level(s)
        return Prediction(probs, s, level, desc, y, tl, speech_p)

    def predict_file(self, path: str | Path, **kw) -> Prediction:
        return self.predict_audio(load_audio(path), **kw)

    def predict_array(self, y: np.ndarray, sr: int, **kw) -> Prediction:
        return self.predict_audio(preprocess(y, sr), **kw)


def main():
    ap = argparse.ArgumentParser(description='Predict emotion and vocal stress for audio files.')
    ap.add_argument('files', nargs='+')
    ap.add_argument('--model', default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')  # bar characters on Windows consoles
    predictor = Predictor(args.model)
    for f in args.files:
        try:
            p = predictor.predict_file(f, timeline=False)
        except ValueError as e:  # no speech / silent / too short
            print(f"\n{f}\n  {e}")
            continue
        print(f"\n{f}\n  emotion: {p.emotion} ({p.probs[p.emotion]:.0%})   stress: {p.stress:.0f}/100 ({p.stress_level})")
        for e, v in sorted(p.probs.items(), key=lambda kv: -kv[1]):
            print(f"    {e:10s} {'█' * int(v * 40):<40s} {v:6.1%}")


if __name__ == '__main__':
    main()
