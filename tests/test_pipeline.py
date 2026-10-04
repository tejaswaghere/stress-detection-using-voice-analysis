import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from features import (EMOTIONS, FEATURE_NAMES, SAMPLE_RATE, extract_features,  # noqa: E402
                      get_emotion_from_filename, parse_ravdess_filename, preprocess)
from model import build_pipeline, load_bundle, make_bundle, save_bundle  # noqa: E402
from predict import Predictor  # noqa: E402
from stress import STRESS_WEIGHTS, stress_index, stress_level  # noqa: E402


def voice_like(seconds=2.0, f0=150.0, sr=SAMPLE_RATE, seed=0):
    """Harmonic tone with a pitch glide and syllable-rate amplitude envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t))) / sr
    y = sum(np.sin(k * phase) / k for k in range(1, 6))
    y *= 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
    return (0.3 * y + 0.005 * rng.standard_normal(t.size)).astype(np.float32)


def test_parse_ravdess_filename():
    assert parse_ravdess_filename('03-01-05-02-01-01-12.wav') == {'emotion': 'angry', 'intensity': 'strong', 'actor': 12}
    assert parse_ravdess_filename('Actor_01/03-01-01-01-01-01-01.wav')['emotion'] == 'neutral'
    assert parse_ravdess_filename('recording.wav') is None
    assert parse_ravdess_filename('03-01-09-01-01-01-01.wav') is None
    assert get_emotion_from_filename('foo.wav') == 'unknown'


def test_feature_vector_matches_names():
    vec = extract_features(preprocess(voice_like(), SAMPLE_RATE))
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_features_handle_minimum_length_clip():
    vec = extract_features(preprocess(voice_like(0.6), SAMPLE_RATE))
    assert vec.shape == (len(FEATURE_NAMES),)


def test_preprocess_is_gain_invariant():
    """Peak normalisation: the same voice recorded quietly or loudly gives the same features."""
    y = voice_like()
    a = extract_features(preprocess(y, SAMPLE_RATE))
    b = extract_features(preprocess(y * 0.1, SAMPLE_RATE))
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


def test_preprocess_resamples_and_mixes_down_int16_stereo():
    y = voice_like(sr=44100)
    stereo = (np.stack([y, y], axis=1) * 32767).astype(np.int16)  # Gradio layout: (samples, channels)
    out = preprocess(stereo, 44100)
    assert out.dtype == np.float32
    assert abs(len(out) / SAMPLE_RATE - 2.0) < 0.2
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_preprocess_rejects_silence_and_short_audio():
    with pytest.raises(ValueError, match='silent'):
        preprocess(np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError, match='Too little speech'):
        preprocess(voice_like(0.2), SAMPLE_RATE)


def test_stress_index():
    assert set(STRESS_WEIGHTS) == set(EMOTIONS)
    assert stress_index({'calm': 1.0}) == 0
    assert stress_index({'angry': 1.0}) == 100
    assert stress_index({'angry': 0.5, 'calm': 0.5}) == pytest.approx(50)
    assert stress_level(10)[0] == 'low'
    assert stress_level(40)[0] == 'moderate'
    assert stress_level(90)[0] == 'high'


def test_predictor_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, len(FEATURE_NAMES)))
    y = np.arange(80) % len(EMOTIONS)
    pipe = build_pipeline('logreg').fit(X, y)
    path = tmp_path / 'm.joblib'
    save_bundle(make_bundle(pipe, 'handcrafted', {'accuracy': 0.0}), path)

    p = Predictor(path).predict_array(voice_like(3.0), SAMPLE_RATE)
    assert set(p.probs) == set(EMOTIONS)
    assert sum(p.probs.values()) == pytest.approx(1.0)
    assert 0 <= p.stress <= 100
    assert p.emotion in EMOTIONS


def test_stale_feature_version_is_rejected(tmp_path):
    bundle = make_bundle(build_pipeline('logreg'), 'handcrafted', {})
    bundle['feature_version'] = 'handcrafted-v1'
    path = tmp_path / 'old.joblib'
    save_bundle(bundle, path)
    with pytest.raises(RuntimeError, match='Retrain'):
        load_bundle(path)


@pytest.mark.skipif(not (ROOT / 'models' / 'emotion_model.joblib').exists(), reason='no trained model')
def test_shipped_model_loads():
    bundle = load_bundle(ROOT / 'models' / 'emotion_model.joblib')
    assert set(bundle['classes']) <= set(EMOTIONS)
    assert len(bundle['pipeline'].classes_) == len(bundle['classes'])
    assert bundle['metrics']['accuracy'] > 1 / len(bundle['classes'])


def test_corpora_parse_and_namespace_speakers(tmp_path):
    import soundfile as sf
    from corpora import SHARED_EMOTIONS, list_clips

    rav = tmp_path / 'rav' / 'Actor_01'
    rav.mkdir(parents=True)
    cre = tmp_path / 'cre' / 'AudioWAV'
    cre.mkdir(parents=True)
    y = voice_like(1.0)
    for name in ['03-01-05-01-01-01-01.wav', '03-01-02-01-01-01-01.wav']:  # angry, calm
        sf.write(rav / name, y, SAMPLE_RATE)
    for name in ['1001_DFA_ANG_XX.wav', '1002_IEO_SAD_HI.wav', 'notes.wav']:
        sf.write(cre / name, y, SAMPLE_RATE)
    (cre.parent / 'VideoDemographics.csv').write_text('"ActorID","Age","Sex","Race","Ethnicity"\n1001,51,"Male","X","Y"\n')

    r = list_clips('ravdess', tmp_path / 'rav')
    c = list_clips('cremad', tmp_path / 'cre')
    assert {(x.emotion, x.speaker, x.sex) for x in r} == {('angry', 'ravdess:1', 'male'), ('calm', 'ravdess:1', 'male')}
    assert [(x.emotion, x.speaker, x.sex) for x in c] == [('angry', 'cremad:1001', 'male'), ('sad', 'cremad:1002', 'unknown')]
    assert {x.emotion for x in c} <= set(SHARED_EMOTIONS)


def test_speech_gate_rejects_and_accepts(tmp_path):
    import joblib
    from gate import NoSpeechError, SpeechGate, build_gate

    rng = np.random.default_rng(0)
    speech, noise = rng.normal(1, 1, (60, 8)), rng.normal(-1, 1, (60, 8))
    pipe = build_gate().fit(np.vstack([speech, noise]), np.r_[np.ones(60), np.zeros(60)])
    path = tmp_path / 'gate.joblib'
    joblib.dump({'pipeline': pipe, 'threshold': 0.5, 'backbone': 'b', 'layers': [1]}, path)

    gate = SpeechGate(path)
    assert gate.check(np.full(8, 2.0)) > 0.5
    with pytest.raises(NoSpeechError, match='No clear speech'):
        gate.check(np.full(8, -2.0))
    assert issubclass(NoSpeechError, ValueError)  # callers that catch ValueError still handle it


@pytest.mark.skipif(not (ROOT / 'models' / 'speech_gate.joblib').exists(), reason='no trained gate')
def test_shipped_gate_matches_emotion_model():
    import joblib

    gate = joblib.load(ROOT / 'models' / 'speech_gate.joblib')
    bundle = load_bundle(ROOT / 'models' / 'emotion_model.joblib')
    assert (gate['backbone'], list(gate['layers'])) == (bundle['backbone'], list(bundle['layers']))
    assert gate['pipeline'].n_features_in_ == bundle['pipeline'].n_features_in_
