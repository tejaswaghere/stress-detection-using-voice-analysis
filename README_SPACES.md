# Deploying to HuggingFace Spaces

Free deployment — your Gradio app becomes live at `https://huggingface.co/spaces/YOUR_USERNAME/stress-detection`

## Step-by-step (10 minutes)

### 1. Create a Space

- Go to [huggingface.co/new-space](https://huggingface.co/new-space)
- Name: `stress-detection`
- SDK: **Gradio**
- Hardware: CPU Basic (free)
- Visibility: Public

### 2. Upload your files

The Space only needs these files (not the full repo):

```
app.py              ← copy from app/app.py in this repo
requirements.txt    ← use the one below
models/
  svm_model.pkl     ← copy your trained model here
```

**requirements.txt for Spaces:**
```
gradio>=4.0.0
librosa>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
soundfile>=0.12.0
```

### 3. Train and export your model first

If you haven't trained yet:
```bash
python src/train.py --dataset data/RAVDESS --model svm
```
This saves `models/svm_model.pkl`.

### 4. Push via Git (recommended)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Clone your empty space
git clone https://huggingface.co/spaces/YOUR_USERNAME/stress-detection
cd stress-detection

# Copy files
cp /path/to/repo/app/app.py .
cp /path/to/repo/requirements.txt .
mkdir models && cp /path/to/repo/models/svm_model.pkl models/

# Push
git add . && git commit -m "Initial deploy" && git push
```

### 5. Link in README

Once live, add this badge to the top of your README.md:

```markdown
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/stress-detection)
```

## Notes

- Spaces cold-starts take ~30s on first visit (free tier)
- Model file must be < 100MB (SVM pickle is usually < 5MB — fine)
- If you upgrade to CNN/LSTM later, use Git LFS for large model files
