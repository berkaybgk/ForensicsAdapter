# Forensics Adapter - Model Architecture Explained

A visual guide to understanding the Forensics Adapter architecture and data flow.

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Face Image                       │
│                      (256×256 RGB)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ├──────────────────┐
                      │                  │
                      ▼                  ▼
        ┌─────────────────────┐  ┌──────────────┐
        │   CLIP ViT-L/14     │  │   Adapter    │
        │   (FROZEN 304M)     │  │ (TRAIN 5.7M) │
        │                     │  │              │
        │ Extract features    │  │ Learn forgery│
        │ from layers:        │  │ boundaries   │
        │  • Layer 0          │  │              │
        │  • Layer 1          │  │ ViT-tiny     │
        │  • Layer 8          │  │ 128 queries  │
        │  • Layer 15         │  │              │
        └──────────┬──────────┘  └──────┬───────┘
                   │                    │
                   │    Interaction     │
                   │    (Attention)     │
                   │                    │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │   RecAttnClip      │
                   │  (Interaction)     │
                   │                    │
                   │ Combine CLIP +     │
                   │ Adapter features   │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │  PostProcess       │
                   │  (Classification)  │
                   │                    │
                   │  MLP: 768→2        │
                   └─────────┬──────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌──────────────────┐         ┌──────────────────┐
    │  Classification  │         │  Forgery Maps    │
    │  [Real, Fake]    │         │  (Xray/Boundary) │
    │                  │         │                  │
    │  Softmax→Prob    │         │  256×256         │
    └──────────────────┘         └──────────────────┘
```

---

## 🔄 Data Flow

### 1. Input Stage
```
Original Image → Face Detection → Crop & Align → Resize to 256×256
                                                         │
                                                         ├→ To CLIP (resize to 224)
                                                         └→ To Adapter
```

### 2. Feature Extraction Stage
```
Image (224×224)
    │
    ▼
┌─────────────────────────────────────────────┐
│         CLIP Visual Encoder                 │
│                                             │
│  Input: 224×224×3                           │
│                                             │
│  → Patch Embedding (16×16 patches)          │
│  → 24 Transformer Layers                    │
│                                             │
│  Extract at specific layers:                │
│    Layer 0:  Initial features               │
│    Layer 1:  Low-level patterns             │
│    Layer 8:  Mid-level structures           │
│    Layer 15: High-level semantics           │
│                                             │
│  Output: 4 feature maps                     │
└─────────────────────────────────────────────┘
```

### 3. Adapter Processing Stage
```
Image (256×256) + CLIP Features
    │
    ▼
┌─────────────────────────────────────────────┐
│            Adapter Network                  │
│                                             │
│  Architecture: ViT-tiny                     │
│  Learnable Queries: 128                     │
│                                             │
│  Process:                                   │
│  1. Extract visual features                 │
│  2. Apply cross-attention with queries      │
│  3. Learn forgery-specific patterns         │
│  4. Generate attention biases               │
│  5. Predict forgery boundaries (xray)       │
│                                             │
│  Focus: Blending boundaries where           │
│         fake faces are composited           │
└─────────────────────────────────────────────┘
```

### 4. Fusion Stage
```
CLIP Features + Adapter Attention Biases
    │
    ▼
┌─────────────────────────────────────────────┐
│           RecAttnClip                       │
│                                             │
│  Mechanism: Attention-based interaction     │
│                                             │
│  Process:                                   │
│  1. Take CLIP's output features             │
│  2. Apply adapter's attention biases        │
│  3. Re-weight CLIP features                 │
│  4. Highlight forgery-relevant regions      │
│                                             │
│  Result: Enhanced feature representation    │
│          (general knowledge + forgery       │
│           specific knowledge)               │
└─────────────────────────────────────────────┘
```

### 5. Classification Stage
```
Enhanced Features
    │
    ▼
┌─────────────────────────────────────────────┐
│         PostClipProcess                     │
│                                             │
│  Architecture: MLP                          │
│  Input: 768-dim features                    │
│  Hidden: 256-dim                            │
│  Output: 2 classes [Real, Fake]             │
│                                             │
│  Process:                                   │
│  1. Linear projection                       │
│  2. ReLU activation                         │
│  3. Final classification layer              │
│  4. Softmax → probabilities                 │
└─────────────────────────────────────────────┘
```

---

## 🎯 Training vs Inference

### Training Mode
```
Input Batch (B × 3 × 256 × 256)
    │
    ├──→ CLIP (frozen) → Features
    ├──→ Adapter (train) → Attention + Xray
    └──→ Interaction (train) → Classification
           │
           ├──→ Classification Loss (CE)
           ├──→ Xray Loss (MSE with GT mask)
           ├──→ Intra-adapter Loss (consistency)
           └──→ CLIP Loss (feature alignment)
                  │
                  └──→ Total Loss = 10×L_cls + 200×L_xray 
                                  + 20×L_intra + 10×L_clip
```

### Inference Mode
```
Input Image
    │
    ├──→ CLIP (frozen) → Features
    ├──→ Adapter (frozen) → Attention + Xray
    └──→ Interaction (frozen) → Classification
           │
           └──→ Probability: P(Fake|Image)
                  │
                  ├──→ Single frame score
                  └──→ Video-level: Average of frame scores
```

---

## 📊 Parameter Distribution

```
Total Parameters: ~310M
├─ CLIP ViT-L/14:     ~304M (FROZEN ❄️)
│  └─ Not updated during training
│
└─ Trainable:         ~5.7M (TRAINED 🔥)
   ├─ Adapter:        ~5.0M
   │  ├─ ViT-tiny backbone
   │  ├─ 128 learnable queries
   │  └─ Cross-attention layers
   │
   ├─ RecAttnClip:    ~0.5M
   │  └─ Attention interaction
   │
   └─ PostProcess:    ~0.2M
      └─ Classification head
```

**Key Insight**: Only 1.8% of parameters are trainable!
- Prevents overfitting
- Retains CLIP's generalization
- Adds task-specific knowledge

---

## 🧠 What Each Component Learns

### CLIP (Frozen)
**Provides**: General visual understanding
- Object recognition
- Scene understanding
- Semantic features
- Pretrained on 400M image-text pairs

**Why Frozen?**
- Already has strong generalization
- Training on limited deepfake data would overfit
- Serves as stable feature extractor

### Adapter (Trainable)
**Learns**: Forgery-specific patterns
- Blending boundaries
- Inconsistent lighting
- Unnatural textures
- Face-background transitions

**Why Small?**
- Prevents overfitting on training data
- Forces learning of generalizable patterns
- Doesn't override CLIP's knowledge

### RecAttnClip (Trainable)
**Learns**: How to combine knowledge
- Which CLIP features are relevant
- How to weight adapter's findings
- Where to focus attention
- Feature fusion strategy

### PostProcess (Trainable)
**Learns**: Final decision mapping
- How to interpret combined features
- Classification boundary
- Confidence calibration

---

## 🔍 Why This Architecture Works

### 1. Leverages Pre-trained Knowledge
```
CLIP (trained on 400M images)
    └──→ General visual understanding
         └──→ Transfers to face images
              └──→ Provides strong baseline
```

### 2. Adds Task-Specific Learning
```
Adapter (trained on deepfakes)
    └──→ Learns forgery patterns
         └──→ Complements CLIP
              └──→ Improves detection
```

### 3. Efficient Interaction
```
Attention Mechanism
    └──→ Combines general + specific
         └──→ Highlights relevant features
              └──→ Robust classification
```

### 4. Strong Generalization
```
Small trainable part (5.7M)
    └──→ Avoids overfitting
         └──→ Generalizes to new datasets
              └──→ Cross-dataset performance
```

---

## 🎨 Forgery Detection Process

### Step-by-Step:

**1. Input Image**
```
Face image with potential forgery
```

**2. CLIP Processing**
```
Extract multi-level features:
- Low-level: edges, textures
- Mid-level: face parts, structures
- High-level: identity, expression
```

**3. Adapter Analysis**
```
Focus on forgery traces:
- Find blending boundaries
- Detect inconsistencies
- Generate attention maps
```

**4. Feature Fusion**
```
Combine CLIP + Adapter:
- Enhance forgery-relevant features
- Suppress irrelevant information
- Create discriminative representation
```

**5. Classification**
```
Map features to probability:
- P(Real) = low → likely real
- P(Fake) = high → likely fake
- Output confidence score
```

**6. Video Aggregation** (if applicable)
```
For video:
- Collect frame predictions
- Average scores
- More stable than single frame
```

---

## 📈 Information Flow Visualization

```
┌──────────────┐
│ Input Image  │  Raw facial image
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ CLIP Branch  │  General features (what is it?)
└──────┬───────┘
       │
       ├─────────────┐
       │             │
       ▼             ▼
┌──────────────┐ ┌──────────────┐
│ Adapter      │ │ CLIP Output  │
│ Branch       │ │              │
└──────┬───────┘ └──────┬───────┘
       │                │
       │ Forgery maps   │ Visual features
       │ Attention bias │
       │                │
       └────────┬───────┘
                │
                ▼
       ┌────────────────┐
       │ Feature Fusion │  Enhanced representation
       └────────┬───────┘
                │
                ▼
       ┌────────────────┐
       │ Classification │  Real or Fake?
       └────────┬───────┘
                │
                ▼
       ┌────────────────┐
       │ Probability    │  Confidence score
       └────────────────┘
```

---

## 🔬 Key Innovations

### 1. Adapter Design
- **Small**: Only 5.7M parameters
- **Focused**: Learns specific forgery patterns
- **Efficient**: Doesn't require retraining CLIP

### 2. Fusion Strategy
- **Attention-based**: Dynamic feature weighting
- **Layer-specific**: Fuses at multiple CLIP layers
- **Bidirectional**: Information flows both ways

### 3. Multi-Task Learning
- **Classification**: Real vs Fake
- **Boundary Detection**: Xray prediction
- **Feature Consistency**: Intra-adapter loss
- **CLIP Alignment**: Maintains feature quality

### 4. Generalization Mechanism
- **CLIP's versatility**: Trained on diverse data
- **Adapter's specificity**: Task-focused learning
- **Small parameter count**: Prevents overfitting
- **Multi-dataset training**: Improves robustness

---

## 🎯 Comparison with Other Approaches

### Traditional CNN-based:
```
Input → CNN → Classification
Problems: Overfits to training data, poor generalization
```

### Fine-tuned CLIP:
```
Input → CLIP (fine-tuned) → Classification
Problems: Loses general knowledge, requires many parameters
```

### Forensics Adapter (This Work):
```
Input → CLIP (frozen) + Adapter (small) → Classification
Benefits: Retains generalization, efficient, task-specific
```

---

## 💡 Why Your Results Make Sense

### Frame-Level Accuracy: 59%
```
Individual frames → Noisy predictions
Some frames lack clear forgery signs
Model must classify every frame
→ Moderate accuracy expected
```

### Frame-Level AUC: 78%
```
Across all thresholds → Good separation
Model distinguishes real/fake patterns
Reasonable discrimination ability
→ Good performance
```

### Video-Level AUC: 100%
```
Averaged per video → Smooth predictions
Noise cancels out
Clear video-level patterns
→ Perfect separation
```

**Conclusion**: Architecture designed for video-level performance!

---

## 🎓 Understanding the "Adapter" Concept

### What is an Adapter?
```
Pretrained Model (Frozen)
        │
        ├──→ Small trainable module (Adapter)
        │         │
        │         └──→ Task-specific knowledge
        │
        └──→ Combined output
```

### Why Adapters Work:
1. **Preserve** pre-trained knowledge
2. **Add** task-specific capabilities
3. **Efficient** (few parameters)
4. **Flexible** (easy to swap)

### Applied to Deepfake Detection:
- **Preserve**: CLIP's visual understanding
- **Add**: Forgery detection capability
- **Result**: Best of both worlds!

---

## 🚀 Practical Implications

### For Research:
- Efficient fine-tuning approach
- Strong baseline for future work
- Interpretable (attention maps)
- Extensible architecture

### For Deployment:
- Reliable video-level detection
- Generalizes across datasets
- Reasonable computational cost
- Clear decision boundaries

### For Understanding Deepfakes:
- Shows importance of blending boundaries
- Validates multi-level feature approach
- Demonstrates value of pre-training
- Highlights generalization challenges

---

**Summary**: The Forensics Adapter cleverly combines CLIP's general visual knowledge with a small, specialized adapter network to achieve strong, generalizable deepfake detection. Your results confirm this design works as intended!

---

*Model Architecture Guide for Forensics Adapter*  
*Last Updated: November 18, 2025*

