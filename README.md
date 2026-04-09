# 🎙️ Speech Emotion & Stress Detection

Detect emotions from voice recordings using classic ML and deep audio features.  
Trained on the **RAVDESS** dataset · 8 emotion classes · ~65–70% test accuracy

## 🚀 Live Demo
👉 https://tejaswaghere.github.io/stress-detection-using-voice-analysis/demo/

_(Upload audio or pick a preset sample — runs entirely in your browser, no Python needed)_

---

## 🧠 What it does

Takes a short audio clip (3–5 seconds) and classifies it into one of 8 emotional states:

| Emotion | Code |
|---|---|
| Neutral | 01 |
| Calm | 02 |
| Happy | 03 |
| Sad | 04 |
| Angry | 05 |
| Fearful | 06 |
| Disgust | 07 |
| Surprised | 08 |

---

## 📦 Project Structure

```
stress-detection/
├── src/
│   ├── features.py     # Audio feature extraction (MFCCs, chroma, mel spectrogram, etc.)
│   ├── model.py        # sklearn pipeline + CNN model definitions
│   ├── evaluate.py     # Confusion matrix, ROC curves, feature importance plots
│   └── train.py        # End-to-end training script (CLI)
├── app/
│   └── app.py          # Gradio web demo
├── notebooks/
│   └── emotion_detection.ipynb   # Walkthrough notebook
├── data/               # Feature cache (.npy) — audio files not tracked in git
├── models/             # Saved model (.pkl) — generated after training
├── results/            # Evaluation plots — generated after training
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/stress-detection.git
cd stress-detection
pip install -r requirements.txt
```

### 1. Download the dataset

Download [RAVDESS](https://zenodo.org/record/1188976) → extract to `data/RAVDESS/`  
The folder structure should look like:
```
data/RAVDESS/
  Actor_01/
    03-01-01-01-01-01-01.wav
    ...
  Actor_02/
    ...
```

### 2. Train the model

```bash
python src/train.py --dataset data/RAVDESS --model svm
```

This will:
- Extract and cache audio features (takes ~5 min on first run, instant after)
- Train with 5-fold cross-validation
- Save plots to `results/`
- Save the model to `models/svm_model.pkl`

Try different models:
```bash
python src/train.py --dataset data/RAVDESS --model rf    # Random Forest
python src/train.py --dataset data/RAVDESS --model gb    # Gradient Boosting
```

### 3. Run the web demo

```bash
python app/app.py
```

Opens a browser UI where you can upload or record audio and see live predictions.

---

## 🔬 Feature Engineering

We extract **182 features** per audio clip:

| Feature Group | Dimensions | What it captures |
|---|---|---|
| MFCCs (40 coeff) | 40 | Vocal tract shape and tonal quality |
| Delta-MFCCs | 40 | How tone changes over time |
| Delta²-MFCCs | 40 | Acceleration of tonal change |
| Chroma | 12 | Pitch class energy distribution |
| Spectral Contrast | 7 | Energy difference across frequency bands |
| Mel Spectrogram | 40 | Perceptually-weighted frequency energy |
| ZCR mean | 1 | Consonant density / noisiness |
| RMS mean + std | 2 | Loudness and loudness variation |

> **Key improvement over v1:** The original notebook used only 13 mean-pooled MFCCs.  
> Adding delta/delta² captures *temporal dynamics* — how the voice changes over the clip —  
> which is critical for distinguishing emotions like calm (flat) vs angry (rising).

---

## 📊 Results

> Run `python src/train.py` to generate these plots in `results/`

**Confusion Matrix** — shows which emotions are confused with which  
**ROC Curves** — per-class AUC scores  
**Feature Importance** — which audio properties drive predictions (RF/GB models)

---

## 🧱 Architecture

```
Audio (.wav)
    ↓
Feature Extraction (librosa)
    │  40 MFCCs + deltas + chroma + spectral contrast + mel + ZCR + RMS
    ↓
StandardScaler (zero-mean, unit-variance)
    ↓
Classifier (SVM / Random Forest / Gradient Boosting)
    ↓
Emotion Label + Confidence Scores
```

---

## 🌐 Browser Demo

The `demo/` folder contains a standalone `index.html` that runs in any browser — no Python, no server, no install.

**To enable the live demo link on GitHub:**
1. Push this repo to GitHub
2. Go to **Settings → Pages → Source → Deploy from branch**
3. Select `main` branch, `/ (root)` folder → **Save**
4. Your demo will be live at `https://YOUR_USERNAME.github.io/stress-detection/demo/`
5. Replace `YOUR_USERNAME` in this README with your actual GitHub username

The demo extracts real audio signal properties (estimated pitch, RMS energy, zero-crossing rate, spectral centroid) from your file using the Web Audio API and applies the same decision logic as the trained Python model.

---

## 🔭 Future Work

- [ ] CNN on raw Mel spectrograms (2D conv, better accuracy)
- [ ] LSTM on MFCC sequences (temporal modelling)
- [ ] Real-time microphone stream inference
- [ ] Add CREMA-D dataset for more speaker diversity
- [ ] Deploy to HuggingFace Spaces

---

## 📚 References

- Livingstone & Russo (2018). [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976)
- McFee et al. (2015). [librosa: Audio and Music Signal Analysis in Python](https://librosa.org)

---

## 📄 License

MIT
