# 🏜️ COHO — Offroad Terrain Semantic Segmentation

A deep learning pipeline for **pixel-level semantic segmentation** of offroad terrain, built on **Mask2Former** (Swin-Base backbone) with optional **SAM3 refinement** for boundary sharpening.

> Fine-tuned on 2,857 offroad images across **11 terrain classes** — achieving **~60% mIoU** and **~87.9% pixel accuracy**.

---

## 🗂️ Project Structure

```
COHO/
├── train_mask2former.py          # Full training + SAM3 evaluation pipeline
├── test_mask2former.py           # Single-image inference & visualization
├── upload_model.py               # Push trained model to HuggingFace Hub
├── deploy_space.py               # Deploy Gradio app to HuggingFace Spaces
├── hf_space/
│   ├── app.py                    # Gradio web demo
│   ├── requirements.txt          # Space-specific dependencies
│   └── README.md                 # HF Space metadata
├── Offroad_Segmentation_Scripts/
│   ├── train_segmentation.py     # Alternative training script
│   ├── test_segmentation.py      # Alternative testing script
│   └── visualize.py              # Visualization utilities
├── requirements.txt              # Project dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🎯 Segmentation Classes

| ID | Class           | Color                                                        |
|----|-----------------|--------------------------------------------------------------|
| 0  | Background      | ![#000000](https://placehold.co/12x12/000000/000000) Black  |
| 1  | Trees           | ![#228B22](https://placehold.co/12x12/228B22/228B22) Green  |
| 2  | Lush Bushes     | ![#00FF00](https://placehold.co/12x12/00FF00/00FF00) Lime   |
| 3  | Dry Grass       | ![#D2B48C](https://placehold.co/12x12/D2B48C/D2B48C) Tan   |
| 4  | Dry Bushes      | ![#8B5A2B](https://placehold.co/12x12/8B5A2B/8B5A2B) Brown |
| 5  | Ground Clutter  | ![#808080](https://placehold.co/12x12/808080/808080) Gray   |
| 6  | Flowers         | ![#FF69B4](https://placehold.co/12x12/FF69B4/FF69B4) Pink   |
| 7  | Logs            | ![#654321](https://placehold.co/12x12/654321/654321) Umber  |
| 8  | Rocks           | ![#A9A9A9](https://placehold.co/12x12/A9A9A9/A9A9A9) Silver |
| 9  | Landscape       | ![#DEB887](https://placehold.co/12x12/DEB887/DEB887) Wheat  |
| 10 | Sky             | ![#87CEEB](https://placehold.co/12x12/87CEEB/87CEEB) Blue   |

---

## ⚙️ Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- HuggingFace account & write token (for model upload / space deployment)

### Installation

```bash
git clone https://github.com/<your-username>/COHO.git
cd COHO
pip install -r requirements.txt
```

### Dataset

Download and extract the offroad segmentation dataset into the project root:

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

## 🚀 Usage

### 1. Train

```bash
python train_mask2former.py \
  --epochs 20 \
  --batch-size 2 \
  --hf-token $HF_TOKEN
```

Key flags:
- `--train-dir` / `--val-dir` — custom dataset paths
- `--runs-dir` — output directory for checkpoints & plots (default: `./runs_mask2former`)
- `--eval-only` — skip training and run evaluation only

### 2. Test on a Single Image

```bash
python test_mask2former.py \
  --image path/to/image.jpg \
  --model-dir ./runs_mask2former/mask2former_best \
  --output result.png
```

### 3. Upload Model to HuggingFace

```bash
python upload_model.py \
  --username your-hf-username \
  --token hf_...
```

### 4. Deploy Gradio Demo to HuggingFace Spaces

```bash
python deploy_space.py \
  --username your-hf-username \
  --token hf_...
```

---

## 🏗️ Architecture

The pipeline consists of two stages:

1. **Mask2Former** — Transformer-based segmentation model with Swin-Base backbone, fine-tuned with frozen encoder and cosine annealing schedule.
2. **SAM3 Refinement** *(optional)* — Segment Anything Model 3 refines low-confidence Mask2Former predictions using bounding-box prompts per class, improving boundary precision.

---

## 📊 Results

| Metric          | Value   |
|-----------------|---------|
| Val mIoU        | ~60%    |
| Val Pixel Acc   | ~87.9%  |
| Training Epochs | 20      |
| Backbone        | Swin-Base (IN21K pretrained) |

---

## 🌐 Live Demo

Try the model on HuggingFace Spaces:  
🔗 [Offroad Segmentation Demo](https://huggingface.co/spaces/kartheek1531/offroad-segmentation)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
