# OffRoad Vision — Terrain Segmentation

> Semantic segmentation of off-road terrain scenes using **Mask2Former** (Swin-Base backbone) with optional **SAM3 refinement** for boundary sharpening. Built for the **Duality AI Track — Hack Energy 2.0** hackathon. Deployed live as a web application on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-orange?style=flat-square)](https://huggingface.co/spaces/harsha3777/offroad-segmentation)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?style=flat-square)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40+-yellow?style=flat-square)](https://huggingface.co/docs/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-green?style=flat-square)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Table of Contents

- [Demo](#demo)
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Team](#team)

---

## Demo

Upload any off-road scene image and the model instantly segments every terrain type pixel by pixel.

| Original | Segmentation Mask | Overlay |
|----------|-------------------|---------|
| Raw input image | Color-coded terrain classes | Mask blended with original |

**Live app:** https://huggingface.co/spaces/harsha3777/offroad-segmentation

---

## Overview

Off-road environments are highly unstructured — a mix of trees, rocks, dry grass, bushes, logs, and sky all blended together. Semantic segmentation assigns a terrain class to every single pixel in the image, which is critical for:

- Autonomous off-road vehicle navigation
- Agricultural and mining robots
- Drone landing zone detection
- Environmental monitoring and land cover mapping
- Search and rescue robotics

This project fine-tunes **Mask2Former** (a state-of-the-art transformer-based segmentation model) on the Duality AI off-road dataset, with an optional **SAM3** (Segment Anything Model 3) refinement stage to sharpen predictions on uncertain regions. The final model is deployed as a real-time web application where users can upload images and instantly visualize terrain segmentation results.

---

## Dataset

Provided by **Duality AI** as part of Hack Energy 2.0.

| Split | Images |
|-------|--------|
| Train | 2,857 |
| Validation | 317 |
| Test | 1,002 |

### Terrain Classes (11 total)

| ID | Class | Color |
|----|-------|-------|
| 0 | Background | Black |
| 1 | Trees | Forest Green |
| 2 | Lush Bushes | Bright Green |
| 3 | Dry Grass | Tan |
| 4 | Dry Bushes | Brown |
| 5 | Ground Clutter | Grey |
| 6 | Flowers | Pink |
| 7 | Logs | Dark Brown |
| 8 | Rocks | Silver Grey |
| 9 | Landscape | Wheat |
| 10 | Sky | Sky Blue |

> **Bug fix:** The original Duality-provided script was missing class 600 (Flowers). We identified and corrected this, bringing the total from 10 to 11 classes.

---

## Model Architecture

The model is a two-stage pipeline:

```
Input Image (512×512)
       │
       ▼
┌──────────────────────────┐
│   Mask2Former             │  ← Fine-tuned
│   (Swin-Base backbone)    │
│   Pretrained on ADE20K    │
│   Encoder: Frozen         │
│   Decoder: Trained        │
└──────────┬───────────────┘
           │  per-pixel class logits
           ▼
    Semantic Segmentation Map
           │
           ▼  (optional)
┌──────────────────────────┐
│   SAM3 Refinement         │  ← Low-confidence regions only
│   (Segment Anything 3)    │
│   Bounding-box prompts    │
│   per uncertain class     │
└──────────┬───────────────┘
           │
           ▼
   Refined Segmentation Map
```

### Stage 1 — Mask2Former (Training + Inference)

- **Backbone:** Swin-Base Transformer, pretrained on ImageNet-21K
- **Base model:** `facebook/mask2former-swin-base-IN21k-ade-semantic`
- **Encoder:** Frozen during training (pixel-level module)
- **Decoder:** Fine-tuned on offroad terrain data
- **Output:** Per-pixel semantic segmentation map with 11 classes

### Stage 2 — SAM3 Refinement (Inference Only)

- **Model:** `facebook/sam3` (Segment Anything Model 3)
- **Purpose:** Refine low-confidence Mask2Former predictions
- **Method:** For each class region where Mask2Former confidence < 0.7, a bounding-box prompt is sent to SAM3, which produces sharper masks
- **Behavior:** Only overwrites uncertain pixels — high-confidence Mask2Former predictions are preserved

---

## Training

### Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| Weight decay | 1e-4 |
| Scheduler | Cosine Annealing |
| Loss | Mask2Former Hungarian matching loss |
| Hardware | Google Colab Tesla T4 GPU |
| Image size | 512 × 512 |
| Gradient clipping | max_norm = 1.0 |

### Data Augmentation

```python
augmentations = [
    HorizontalFlip(p=0.5),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
    GaussianBlur(p=0.2),
]
```

---

## Results

| Metric | Score |
|--------|-------|
| Validation mIoU | **~60%** |
| Validation Pixel Accuracy | **~87.9%** |
| Training Epochs | 20 |
| Backbone | Swin-Base (IN21K pretrained) |

---

## Project Structure

```
Offroad-Terrain-Semantic-Segmentation/
├── train_mask2former.py              # Full training + SAM3 evaluation pipeline
├── test_mask2former.py               # Single-image inference & visualization
├── upload_model.py                   # Push trained model to HuggingFace Hub
├── deploy_space.py                   # Deploy Gradio app to HuggingFace Spaces
├── hf_space/
│   ├── app.py                        # Gradio web demo (deployed to HF Spaces)
│   ├── requirements.txt              # Space-specific dependencies
│   └── README.md                     # HF Space metadata
├── Offroad_Segmentation_Scripts/
│   ├── train_segmentation.py         # Alternative training script
│   ├── test_segmentation.py          # Alternative testing script
│   ├── visualize.py                  # Visualization utilities
│   └── ENV_SETUP/                    # Environment setup scripts
├── requirements.txt                  # Project dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- HuggingFace account & write token (for model upload / space deployment)

### Clone the repository

```bash
git clone https://github.com/harsha3v777/Offroad-Terrain-Semantic-Segmentation-.git
cd Offroad-Terrain-Semantic-Segmentation-
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download dataset

Download and extract the Duality AI offroad segmentation dataset into the project root:

```
Offroad_Segmentation_Training_Dataset/
└── Offroad_Segmentation_Training_Dataset/
    ├── train/
    │   ├── Color_Images/
    │   └── Segmentation/
    └── val/
        ├── Color_Images/
        └── Segmentation/
```

---

## Running Locally

### Train the model

```bash
python train_mask2former.py \
  --epochs 20 \
  --batch-size 2 \
  --hf-token $HF_TOKEN
```

Key flags:
- `--train-dir` / `--val-dir` — custom dataset paths
- `--runs-dir` — output directory for checkpoints & plots (default: `./runs_mask2former`)
- `--eval-only` — skip training, run evaluation only

### Test on a single image

```bash
python test_mask2former.py \
  --image path/to/image.jpg \
  --model-dir ./runs_mask2former/mask2former_best \
  --output result.png
```

Upload any off-road image to get instant terrain segmentation with:
- Color-coded segmentation mask
- Overlay of mask on original image
- Per-class IoU metrics

---

## Deployment

### Upload model to HuggingFace Hub

```bash
python upload_model.py \
  --username your-hf-username \
  --token hf_...
```

### Deploy Gradio app to HuggingFace Spaces

```bash
python deploy_space.py \
  --username your-hf-username \
  --token hf_...
```

The app runs automatically on push to the Hugging Face Space repository. Gradio handles all dependencies and serves the web interface.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Segmentation model | Mask2Former (Facebook Research) |
| Backbone | Swin-Base Transformer (IN21K) |
| Refinement | SAM3 — Segment Anything Model 3 |
| Framework | PyTorch + HuggingFace Transformers |
| Training | Google Colab T4 GPU |
| Web demo | Gradio |
| Model hosting | HuggingFace Hub |
| App hosting | HuggingFace Spaces |
| Augmentation | Albumentations |
| Version control | GitHub |

---

## Future Work

- **Larger backbone** — Upgrade from Swin-Base to Swin-Large for richer feature representations
- **Full SAM3 integration** — Run SAM3 refinement on all test images and benchmark the mIoU improvement
- **Advanced augmentation** — Add CutMix, MixUp, and random scale/crop for more robust training
- **Backbone fine-tuning** — Unfreeze Swin encoder with a very small learning rate for domain adaptation
- **Full test set evaluation** — Run inference on all 1,002 test images and submit to the hackathon leaderboard
- **Real-time video** — Extend the app to process live video streams from drone or vehicle cameras
- **Edge deployment** — Quantize and export to ONNX/TensorRT for mobile and embedded inference
- **Class-balanced sampling** — Implement oversampling for rare classes (flowers, logs) to boost their IoU

---

## Team

**Team Name:** OffRoad Vision  
**Hackathon:** Hack Energy 2.0 — Duality AI Track

| Name | Role |
|------|------|
| Harsha Vennapusa | Model training, deployment, full-stack development |

---

## Acknowledgements

- [Duality AI](https://duality.ai) for the off-road segmentation dataset and hackathon
- [Facebook Research](https://github.com/facebookresearch/Mask2Former) for the Mask2Former architecture
- [Facebook Research](https://github.com/facebookresearch/segment-anything) for the Segment Anything Model
- [Hugging Face](https://huggingface.co) for free model and app hosting

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
