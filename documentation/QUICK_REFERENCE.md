# Forensics Adapter - Quick Reference Guide

This is a quick reference for working with the Forensics Adapter codebase on your Mac.

---

## 📁 Project Structure

```
ForensicsAdapter/
├── config/
│   ├── test.yaml          # Testing configuration
│   └── train.yaml         # Training configuration
├── dataset/
│   └── abstract_dataset.py # Dataset loader
├── dataset_jsons/
│   ├── Celeb-DF-v1.json   # Full dataset metadata
│   └── Celeb-DF-v1-mini.json # Mini test set (4 videos)
├── model/
│   ├── ds.py              # Main model (DS = Detector System)
│   ├── adapters/
│   │   └── adapter.py     # Adapter network
│   ├── attn.py            # Attention mechanisms
│   ├── layer.py           # Post-processing layers
│   └── clip/              # CLIP model
├── weights/
│   └── ckpt_best.pth      # Pretrained model weights
├── test.py                # Main testing script
└── trainer/
    └── metrics/
        └── utils.py       # Metric computation
```

---

## 🚀 Quick Commands

### Run Test
```bash
# Basic test with default config
/opt/miniconda3/envs/FA2/bin/python test.py

# Test with specific config
/opt/miniconda3/envs/FA2/bin/python test.py --detector_path config/test.yaml

# Test with specific weights
/opt/miniconda3/envs/FA2/bin/python test.py --weights_path weights/ckpt_best.pth

# Test on specific dataset
/opt/miniconda3/envs/FA2/bin/python test.py --test_dataset Celeb-DF-v1-mini
```

### Analyze Results
```bash
# Run detailed analysis (after testing)
python3 verify_results.py
```

### Create Mini Dataset
```bash
# Edit create_mini_dataset.py to set number of videos
python3 create_mini_dataset.py
```

---

## 📊 Understanding Metrics

### Frame-Level Metrics

| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| **acc** | 0-1 | >0.55 | Accuracy: % of frames correctly classified |
| **auc** | 0-1 | >0.75 | Area Under ROC Curve: separation ability |
| **eer** | 0-1 | <0.30 | Equal Error Rate: false pos = false neg |
| **ap** | 0-1 | >0.70 | Average Precision: precision across thresholds |

### Video-Level Metrics

| Metric | Range | Good Value | Description |
|--------|-------|------------|-------------|
| **video_auc** | 0-1 | >0.95 | Video-level AUC: aggregated performance |

**Important**: Video-level is more reliable than frame-level!

---

## ⚙️ Configuration Quick Reference

### Key Config Parameters (test.yaml)

```yaml
# Model
clip_model_name: "ViT-L/14"     # CLIP backbone
vit_name: 'vit_tiny_patch16_224' # Adapter network
num_quires: 128                  # Adapter queries
fusion_map: {0: 0, 1: 1, 2: 8, 3: 15} # CLIP-Adapter interaction

# Testing
test_dataset: [Celeb-DF-v1-mini] # Datasets to test
test_batchSize: 2                # Batch size
frame_num: {'test': 32}          # Frames per video
workers: 0                       # Data loading workers

# Hardware
device: 'cpu'                    # 'cpu' or 'cuda'
cuda: false                      # Enable CUDA
```

### To Modify:

**Test different dataset:**
```yaml
test_dataset: [Celeb-DF-v1-mini]  # Change this line
```

**Change batch size (if memory issues):**
```yaml
test_batchSize: 1  # Reduce to 1 if needed
```

**Use GPU (if available):**
```yaml
device: 'cuda'
cuda: true
```

---

## 🔍 Key Code Files

### 1. test.py
Main testing script. Key functions:

```python
test_epoch()           # Runs full test epoch
test_one_dataset()     # Tests on single dataset
inference()            # Single batch inference
```

**Flow**: Load config → Create dataset → Load model → Run test → Compute metrics

### 2. model/ds.py
Main model class (`DS`):

```python
__init__()             # Initialize CLIP + Adapter
forward()              # Forward pass (returns predictions)
get_losses()           # Compute training losses
get_test_metrics()     # Compute test metrics
```

**Key Components**:
- `self.clip_model`: Frozen CLIP ViT-L/14
- `self.adapter`: Trainable adapter network
- `self.rec_attn_clip`: Attention interaction
- `self.clip_post_process`: Classification head

### 3. dataset/abstract_dataset.py
Dataset loader (`DeepfakeAbstractBaseDataset`):

```python
__init__()             # Load dataset from JSON
__getitem__()          # Get single sample
collate_fn()           # Batch collation
```

**Returns**: Dict with image, label, mask, landmark, xray, patch_labels

### 4. trainer/metrics/utils.py
Metric computation:

```python
get_test_metrics()     # Compute all metrics
get_video_metrics()    # Video-level aggregation
```

---

## 🎯 Common Tasks

### 1. Test on Different Dataset

**Step 1**: Create dataset JSON
```python
# In generate_dataset_json.py
generate_celeb_df_json(
    dataset_root="datasets/YOUR_DATASET",
    output_json="dataset_jsons/YOUR_DATASET.json"
)
```

**Step 2**: Update config
```yaml
# In config/test.yaml
test_dataset: [YOUR_DATASET]
```

**Step 3**: Run test
```bash
/opt/miniconda3/envs/FA2/bin/python test.py
```

### 2. Create Larger Test Set

**Edit create_mini_dataset.py**:
```python
create_mini_dataset(
    input_json="dataset_jsons/Celeb-DF-v1.json",
    output_json="dataset_jsons/Celeb-DF-v1-medium.json",
    num_real=10,   # Increase
    num_fake=10    # Increase
)
```

Then run and update config.

### 3. Analyze Model Predictions

**After running test.py**, copy predictions and labels, then:

**Edit verify_results.py**:
```python
# Replace these arrays with your output
predictions = np.array([...])  # From test.py output
labels = np.array([...])       # From test.py output

# Run analysis
python3 verify_results.py
```

### 4. Debug Model Loading

```python
# Check model loading
import torch
ckpt = torch.load('weights/ckpt_best.pth', map_location='cpu')
print(ckpt.keys())  # See what's in checkpoint
```

### 5. Visualize Forgery Boundaries

```python
# In test.py, modify test_one_dataset() to save xray predictions
import matplotlib.pyplot as plt
import cv2

# After model forward pass
xray_pred = predictions['xray_pred']  # Get boundary prediction
xray_np = xray_pred.cpu().numpy().squeeze()

# Visualize
plt.imshow(xray_np, cmap='hot')
plt.savefig('xray_visualization.png')
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size
```yaml
test_batchSize: 1
```

### Issue: "Dataset not found"
**Solution**: Check JSON file exists and paths are correct
```bash
ls -l dataset_jsons/
cat dataset_jsons/YOUR_DATASET.json | head -20
```

### Issue: "Model weights don't load"
**Solution**: Check file exists and is correct
```bash
ls -lh weights/ckpt_best.pth
```

### Issue: "Frames not found"
**Solution**: Verify image paths in JSON match actual files

```python
import json

with open('../dataset_jsons/Celeb-DF-v1-mini.json') as f:
    data = json.load(f)
# Check first frame path
first_frame = data['Celeb-DF-v1-mini']['0-real']['test']['video_0000']['frames'][0]
print(f"Checking: {first_frame}")
# Verify it exists
import os

print(f"Exists: {os.path.exists(first_frame)}")
```

### Issue: "Low accuracy"
**Check**:
1. Are you looking at frame-level or video-level?
2. Video-level should be high (>0.95)
3. Frame-level naturally lower (~0.6-0.8)

---

## 📈 Expected Performance

### On Mini Dataset (4 videos, 128 frames):
- **video_auc**: ~1.0 (perfect or near-perfect)
- **frame_auc**: 0.75-0.85
- **frame_acc**: 0.55-0.70

### On Full Celeb-DF-v1 (paper results):
- **AUC**: ~98-99%

### Cross-Dataset Generalization:
Model trained on FaceForensics++ generalizes to:
- Celeb-DF: ~98-99%
- DFDC: ~85-90%
- Other datasets: 75-95%

---

## 🔬 Understanding the Model

### What Does Each Component Do?

**CLIP (Frozen)**:
- Extracts general visual features
- Pre-trained on image-text pairs
- Provides semantic understanding

**Adapter (Trainable)**:
- Learns forgery-specific features
- Focuses on blending boundaries
- Small (5.7M params) → prevents overfitting

**RecAttnClip**:
- Enables interaction between CLIP and Adapter
- Uses attention mechanisms
- Combines general + specific features

**PostProcess**:
- Final classification layer
- Maps features to real/fake probability

### Why This Architecture Works:

1. **CLIP provides generalization** (trained on diverse images)
2. **Adapter adds specialization** (learns forgery traces)
3. **Small adapter prevents overfitting** (retains CLIP's versatility)
4. **Result**: Strong cross-dataset performance!

---

## 💡 Tips for Research

### 1. Start Small
- Use mini datasets (4-10 videos)
- Iterate quickly
- Scale up when confident

### 2. Trust Video-Level Metrics
- More stable than frame-level
- Better reflects real-world performance
- Use for final evaluation

### 3. Visualize Everything
- Prediction distributions
- ROC curves
- Confusion matrices
- Forgery boundaries (xray)

### 4. Compare Across Datasets
- Test on multiple datasets
- Check cross-dataset generalization
- Understand failure modes

### 5. Document Your Findings
- Keep notes on experiments
- Track which configs work best
- Note unexpected behaviors

---

## 📚 Useful Python Snippets

### Load and Inspect Config
```python
import yaml
with open('config/test.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config.keys())
```

### Check Dataset Size
```python
import json
with open('dataset_jsons/Celeb-DF-v1-mini.json') as f:
    data = json.load(f)
    
dataset_name = 'Celeb-DF-v1-mini'
real_videos = len(data[dataset_name]['0-real']['test'])
fake_videos = len(data[dataset_name]['1-fake']['test'])
print(f"Real: {real_videos}, Fake: {fake_videos}")
```

### Compute Custom Metrics
```python
import numpy as np
from sklearn.metrics import roc_curve, auc

# Your predictions and labels
pred = np.array([...])
label = np.array([...])

# Custom threshold
threshold = 0.7
acc = np.mean((pred > threshold) == label)
print(f"Accuracy @ {threshold}: {acc:.4f}")
```

---

## 🎓 Learning Resources

### To Understand CLIP:
- Paper: "Learning Transferable Visual Models From Natural Language Supervision"
- OpenAI CLIP: https://github.com/openai/CLIP

### To Understand Adapters:
- Paper: "Parameter-Efficient Transfer Learning for NLP"
- Concept: Add small trainable modules to frozen models

### To Understand Deepfake Detection:
- Survey: "Media Forensics and DeepFakes: an Overview"
- Datasets: FaceForensics++, Celeb-DF, DFDC

---

## 📞 Quick Help

**Model not loading?**
→ Check `weights/ckpt_best.pth` exists

**Low performance?**
→ Check you're using the right dataset and config

**Out of memory?**
→ Reduce `test_batchSize` to 1

**Confused about metrics?**
→ Focus on `video_auc` (most important)

**Want to test your own images?**
→ You'll need to preprocess (face detection + cropping) first

---

**Remember**: Video-level AUC = 1.0 means your model is working! 🎉

---

*Quick Reference Guide for Forensics Adapter*  
*Last Updated: November 18, 2025*

