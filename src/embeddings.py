"""
embeddings.py — Utterance embeddings from a pretrained self-supervised speech model.

Why: handcrafted features are compact and interpretable, but they generalise
poorly to speakers the model has never heard. Self-supervised models such as
WavLM were pretrained on ~94k hours of speech and their intermediate layers
encode prosody and voice quality in a largely speaker-robust way. A small
linear classifier on top of frozen WavLM features roughly doubles
speaker-independent accuracy compared with the original 182-feature SVM
(see results/metrics.json).

The backbone stays frozen — nothing is fine-tuned — so training is just
"extract embeddings once, fit a scikit-learn classifier", and the only file
we ship is the small classifier head (the backbone downloads from the Hub).

Requires: torch, transformers
"""

from __future__ import annotations

import numpy as np

from features import SAMPLE_RATE

DEFAULT_BACKBONE = 'microsoft/wavlm-base-plus'


class SSLEmbedder:
    """Frozen SSL backbone → time-averaged hidden states, averaged over the selected layers."""

    def __init__(self, backbone: str = DEFAULT_BACKBONE, layers: tuple[int, ...] = (), device: str = 'cpu'):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self._torch = torch
        self.backbone = backbone
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone).eval().to(device)
        n_layers = self.model.config.num_hidden_layers + 1  # + CNN feature projection output
        self.layers = tuple(layers) if layers else tuple(range(n_layers))

    def hidden_states(self, y: np.ndarray):
        """(n_layers, T, D) hidden states for preprocessed 16 kHz audio."""
        torch = self._torch
        inputs = self.extractor(y, sampling_rate=SAMPLE_RATE, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hs = self.model(**inputs, output_hidden_states=True).hidden_states
        return torch.stack([hs[i] for i in self.layers])[:, 0]

    @staticmethod
    def pool(h) -> np.ndarray:
        """(L, T, D) → (D,): mean over time, then over layers. (Adding std pooling or
        concatenating layers instead of averaging gave no significant gain — see results/layer_probe.json.)"""
        return h.mean(dim=(0, 1)).cpu().numpy()

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.pool(self.hidden_states(y)).astype(np.float32)

    def windows(self, y: np.ndarray, win_s: float = 3.0, hop_s: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """
        Embed sliding windows from a single forward pass (frames are 20 ms).
        Returns (window_centres_s, embeddings[n_windows, dim]).
        """
        h = self.hidden_states(y)
        fps = 50
        T = h.shape[1]
        win, hop = int(win_s * fps), int(hop_s * fps)
        if T <= win:
            return np.array([T / fps / 2]), self.pool(h)[None]
        starts = list(range(0, T - win + 1, hop))
        embs = np.stack([self.pool(h[:, s:s + win]) for s in starts])
        return np.array([(s + win / 2) / fps for s in starts]), embs.astype(np.float32)
