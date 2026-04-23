# 🔬 Embryo Quality Classification using CNN + LSTM

> **Course:** Deep Learning for Medical Images  
> **Name:** A Smaran Reddy  
> **Roll No.:** CS23B1011  

---

## 📋 Overview

This project builds an end-to-end deep learning pipeline to automatically classify embryo quality from time-lapse microscopy sequences. The model combines a **CNN backbone** for spatial feature extraction with an **LSTM** for temporal pattern modeling — mimicking how an expert embryologist assesses embryo development across multiple time points.

---

## 🗂️ Dataset

**Source:** [`abhishekbuddiga06/embryo-dataset`](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset) on Kaggle

| Property | Value |
|---|---|
| Total embryo sequences | 704 |
| Total images | 342,363 |
| Raw grades | 12 (Grade 1 – Grade 12) |
| Annotation format | Per-embryo CSV (`AA83-7_phases.csv` → Grade 7) |
| Image type | Time-lapse JPEG frames |

### Directory Structure
```
/kaggle/input/datasets/abhishekbuddiga06/embryo-dataset/
    embryo_dataset/
        embryo_dataset/
            AA83-7/          ← one folder per embryo
                frame_001.jpg
                frame_002.jpg
                ...
    embryo_dataset_annotations/
        embryo_dataset_annotations/
            AA83-7_phases.csv   ← morphokinetic timing data + grade in filename
```

### Grade Grouping

Raw grades are mapped to 3 coarse quality classes to handle class imbalance:

| Raw Grades | Quality Class | Embryos |
|---|---|---|
| 1 – 4 | Good | 285 |
| 5 – 8 | Average | 307 |
| 9 – 12 | Poor | 107 |

---

## 🏗️ Model Architecture

```
Input [Batch, T=10, 3, 224, 224]
        │
        ▼  reshape → [Batch × T, 3, 224, 224]
   ┌─────────────────────┐
   │   ResNet18 (CNN)     │  ← pretrained ImageNet weights
   │   Shared across T    │  ← same weights applied to every frame
   └─────────────────────┘
        │  [Batch × T, 512]
        ▼  reshape → [Batch, T, 512]
   ┌─────────────────────┐
   │   2-layer LSTM       │  hidden=256, dropout=0.3
   └─────────────────────┘
        │  [Batch, T, 256]
        ▼
   ┌─────────────────────┐
   │  Temporal Attention  │  soft-weights informative frames
   └─────────────────────┘
        │  [Batch, 256]
        ▼
   Dropout(0.5) → FC(128) → GELU → Dropout(0.25) → FC(3)
        │
   Output: [Batch, 3]  logits
```

**Total trainable parameters:** 12,525,891

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| CNN Backbone | ResNet18 (pretrained) |
| Sequence Length | 10 frames (uniformly sampled) |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=4) |
| Loss Function | CrossEntropyLoss (weighted + label smoothing 0.05) |
| Gradient Clipping | 1.0 |
| CNN Freeze Epochs | 5 (warm-up phase) |
| Early Stopping | Patience=8, Delta=1e-4 |
| Max Epochs | 40 |
| Train / Val / Test Split | 70% / 15% / 15% |

### Key Training Techniques
- **CNN warm-up:** Backbone frozen for first 5 epochs so LSTM stabilises before fine-tuning
- **WeightedRandomSampler:** Oversamples minority classes each epoch to handle imbalance
- **Class-weighted loss:** Further compensates for the Poor class being underrepresented
- **Temporal attention:** Learns to weight informative frames (e.g. cell division moments) more heavily

---

## 📊 Results

### Test Set Performance

| Metric | Value |
|---|---|
| Accuracy | **92.38%** |
| Precision (weighted) | 0.9230 |
| Recall (weighted) | 0.9230 |
| F1-Score (weighted) | **0.9230** |
| Best epoch | 21 / 40 |

### Per-Class Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Good | ~0.95 | **96.36%** | ~0.95 |
| Average | ~0.92 | **92.11%** | ~0.92 |
| Poor | ~0.75 | **75.00%** | ~0.75 |

### Confusion Matrix Highlights
- ✅ **No Good → Poor or Poor → Good misclassifications** (0.00%) — the clinically dangerous errors are avoided
- ⚠️ 3 Poor embryos misclassified as Average — a conservative error at the grade 8/9 boundary
- ✅ Good class identified with 96.36% recall

---

## 📁 Notebook Structure

| Section | Description |
|---|---|
| 01 · Imports & Seed | All libraries, reproducibility setup |
| 02 · Paths & Config | Hardcoded Kaggle paths, all hyperparameters |
| 03 · Manifest Builder | Parses grade from annotation filenames, builds image manifest |
| 04 · Grade Grouping | Maps 12 raw grades → 3 quality classes |
| 05 · Dataset Visualisation | Class distribution plots, sample frame grids |
| 06 · Dataset & DataLoaders | EmbryoDataset class, frame sampling, weighted sampler |
| 07 · CNN + LSTM Model | Full model definition with temporal attention |
| 08 · Training Loop | Loss, optimizer, early stopping, train/eval functions |
| 09 · Training Curves | Loss, accuracy, LR plots |
| 10 · Test Evaluation | Accuracy, precision, recall, F1, classification report |
| 11 · Confusion Matrix | Raw counts + recall-normalised heatmaps |
| 12 · Prediction Examples | Visual predictions with probability bars |
| 13 · Export & Inference | Full checkpoint save, single-embryo inference function |

---

## 🔧 How to Run on Kaggle

1. Open a new Kaggle Notebook
2. Click **Add Data** → search `abhishekbuddiga06/embryo-dataset` → Add
3. Set **Accelerator** to GPU (Settings → Accelerator → GPU T4 x2)
4. Upload `embryo_lstm_kaggle.ipynb`
5. Click **Run All**

All outputs are saved to `/kaggle/working/outputs/` and `/kaggle/working/checkpoints/`.

---

## 📦 Output Files

| File | Description |
|---|---|
| `checkpoints/best_model.pt` | Best model weights (by val loss) |
| `checkpoints/full_checkpoint.pt` | Full state: weights + optimizer + history + config |
| `outputs/training_curves.png` | Loss, accuracy, LR plots |
| `outputs/confusion_matrix.png` | Test set confusion matrices |
| `outputs/predictions.png` | Sample prediction examples |
| `outputs/class_distribution.png` | Dataset class balance |
| `outputs/sample_frames.png` | Sample embryo frames per class |
| `outputs/test_metrics.json` | Final test metrics (JSON) |

---

## 🔍 Inference on a New Embryo

```python
result = predict_embryo(
    image_paths = sorted(Path('/path/to/embryo_folder').glob('*.jpg')),
    model       = model,
    transform   = eval_tfm,
    seq_len     = cfg.SEQ_LEN,
    class_names = CLASS_NAMES,
    device      = cfg.DEVICE
)
print(result)
# {'predicted_class': 'Good', 'confidence': 0.964, 'probabilities': {...}}
```

---

## 🚀 Possible Improvements

| Area | Idea |
|---|---|
| Architecture | Replace LSTM with `nn.TransformerEncoder` for better long-range temporal modeling |
| Backbone | Swap ResNet18 → EfficientNet-B2 or ConvNeXt-Tiny via `timm` |
| Data | Add optical flow between frames as a 4th input channel |
| Training | OneCycleLR scheduler for faster convergence |
| Annotation | Use morphokinetic timing values from CSVs as auxiliary tabular features |
| Deployment | Export to ONNX for edge inference |

---

## 📚 References

- He et al. (2016) — *Deep Residual Learning for Image Recognition*
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*
- Khosravi et al. (2019) — *Deep learning enables robust assessment and selection of human blastocysts after IVF*
- Dataset: [abhishekbuddiga06/embryo-dataset](https://www.kaggle.com/datasets/abhishekbuddiga06/embryo-dataset)
