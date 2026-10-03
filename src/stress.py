"""
stress.py — Map emotion probabilities to a 0–100 vocal stress index.

RAVDESS has no stress labels, so stress is derived from the emotion classifier
using the circumplex (arousal × valence) model of affect: stress corresponds to
high arousal combined with negative valence. Each emotion gets a weight for how
strongly it reflects that quadrant, and the index is the probability-weighted
sum. This is a transparent heuristic, not a clinically validated measure.
"""

from __future__ import annotations

STRESS_WEIGHTS = {
    'angry':     1.00,  # high arousal, negative valence
    'fearful':   1.00,  # high arousal, negative valence
    'disgust':   0.70,  # moderate arousal, negative valence
    'sad':       0.45,  # low arousal, negative valence
    'surprised': 0.35,  # high arousal, ambiguous valence
    'happy':     0.10,  # high arousal, positive valence
    'neutral':   0.00,
    'calm':      0.00,
}

LEVELS = [  # (upper bound, level, description)
    (30, 'low', 'Voice sounds relaxed'),
    (55, 'moderate', 'Some vocal stress markers'),
    (101, 'high', 'Strong vocal stress markers'),
]


def stress_index(probs: dict[str, float]) -> float:
    """Probability-weighted stress index in [0, 100]."""
    total = sum(probs.values()) or 1.0
    return 100.0 * sum(STRESS_WEIGHTS.get(e, 0.0) * p for e, p in probs.items()) / total


def stress_level(index: float) -> tuple[str, str]:
    """Return (level, description) for a stress index."""
    for upper, level, desc in LEVELS:
        if index < upper:
            return level, desc
    return LEVELS[-1][1], LEVELS[-1][2]
