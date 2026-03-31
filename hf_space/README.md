---
title: Offroad Semantic Segmentation
emoji: 🏜️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Mask2Former fine-tuned for off-road terrain segmentation
---

# 🏜️ Offroad Semantic Segmentation

A **Mask2Former** (Swin-Base backbone) model fine-tuned on a custom offroad driving dataset across **11 terrain classes**.

## Classes
| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Background | 6 | Flowers |
| 1 | Trees | 7 | Logs |
| 2 | Lush Bushes | 8 | Rocks |
| 3 | Dry Grass | 9 | Landscape |
| 4 | Dry Bushes | 10 | Sky |
| 5 | Ground Clutter | | |

## Performance (20 epochs)
- **Val mIoU**: ~60%
- **Val Pixel Accuracy**: ~87.9%
- **Training samples**: 2,857 | **Val samples**: 317
