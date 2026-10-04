"""
gate.py — Speech gate: refuse to predict an emotion when a clip contains no speech.

The emotion classifier always outputs one of its emotions, even for silence, music or a door
slam, and it can sound sure of itself (a pure tone came back "sad, 90%"). The gate is a small
logistic regression on the same WavLM embedding the emotion model uses, so it costs ~0 ms extra.
It was trained on speech, speech mixed with background noise, and non-speech sounds; see
scripts/train_speech_gate.py and results/speech_gate/metrics.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GATE_THRESHOLD = 0.5
DEFAULT_GATE = Path(__file__).resolve().parent.parent / 'models' / 'speech_gate.joblib'


class NoSpeechError(ValueError):
    """Raised when a clip doesn't contain speech the model can analyse."""


def build_gate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight='balanced'))


class SpeechGate:
    def __init__(self, path: str | Path = DEFAULT_GATE):
        import joblib

        bundle = joblib.load(path)
        self.pipeline = bundle['pipeline']
        self.threshold = bundle['threshold']
        self.backbone, self.layers = bundle['backbone'], bundle['layers']

    def speech_probability(self, embedding: np.ndarray) -> float:
        return float(self.pipeline.predict_proba(np.asarray(embedding).reshape(1, -1))[0, 1])

    def check(self, embedding: np.ndarray) -> float:
        p = self.speech_probability(embedding)
        if p < self.threshold:
            raise NoSpeechError(
                "No clear speech detected. This app analyses a spoken voice: please say a full sentence "
                "close to the microphone. Music, background noise or silence can't be analysed.")
        return p
