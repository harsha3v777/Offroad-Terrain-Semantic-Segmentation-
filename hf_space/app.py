"""
Offroad Semantic Segmentation — Hugging Face Spaces Demo
Mask2Former fine-tuned on 11-class offroad terrain dataset.

Set the MODEL_REPO environment variable in your Space settings to point to
your uploaded model, e.g.: "your-username/offroad-mask2former"
"""

import os
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from transformers import (
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)
import gradio as gr

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REPO = os.environ.get("MODEL_REPO", "kartheek1531/offroad-mask2former")

ID2LABEL = {
    0: "background",    1: "trees",          2: "lush bushes",
    3: "dry grass",     4: "dry bushes",     5: "ground clutter",
    6: "flowers",       7: "logs",           8: "rocks",
    9: "landscape",    10: "sky",
}

CLASS_COLORS = {
    0:  (0,   0,   0),    1:  (34,  139, 34),  2:  (0,   200,  0),
    3:  (210, 180, 140),  4:  (139,  90, 43),  5:  (128, 128, 128),
    6:  (255, 105, 180),  7:  (101,  67, 33),  8:  (169, 169, 169),
    9:  (222, 184, 135), 10:  (135, 206, 235),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (cached at startup)
# ──────────────────────────────────────────────────────────────────────────────
print(f"[INFO] Loading model from: {MODEL_REPO}")
try:
    processor = Mask2FormerImageProcessor.from_pretrained(MODEL_REPO)
    model     = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_REPO)
    model     = model.to(DEVICE).eval()
    print("[INFO] Model loaded successfully ✅")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    processor, model = None, None


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(image_pil: Image.Image) -> np.ndarray:
    """Return a semantic class-index map (H×W numpy array)."""
    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_pil.size[::-1]]
    )[0]
    return result.cpu().numpy().astype(np.uint8)


def mask_to_color(mask_2d: np.ndarray) -> np.ndarray:
    """Convert class-index mask → RGB image."""
    h, w   = mask_2d.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in CLASS_COLORS.items():
        canvas[mask_2d == cls_id] = rgb
    return canvas


def class_iou(pred: np.ndarray, gt_dummy: np.ndarray | None = None) -> dict:
    """Return pixel-coverage (%) for each predicted class."""
    total  = pred.size
    result = {}
    for cls_id, name in ID2LABEL.items():
        pct = float((pred == cls_id).sum()) / total * 100
        if pct > 0.5:
            result[name] = round(pct, 1)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# BUILD FIGURE (matches reference screenshot layout)
# ──────────────────────────────────────────────────────────────────────────────
def build_figure(
    orig_pil: Image.Image,
    semantic_map: np.ndarray,
    alpha: float,
) -> Image.Image:
    """Produce the 3-panel figure with colour legend."""
    orig_np   = np.array(orig_pil)
    color_np  = mask_to_color(semantic_map)
    color_pil = Image.fromarray(color_np)
    blend_pil = Image.blend(orig_pil.convert("RGB"), color_pil, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    panels = [
        (orig_np,          "Original Image"),
        (color_np,         "Mask2Former Prediction"),
        (np.array(blend_pil), f"Blended Overlay  (α = {alpha:.1f})"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color="white", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Colour legend
    present_classes = sorted({int(c) for c in np.unique(semantic_map)})
    patches = [
        mpatches.Patch(
            color=np.array(CLASS_COLORS[c]) / 255,
            label=f"{c}: {ID2LABEL[c]}",
        )
        for c in present_classes if c in CLASS_COLORS
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=min(6, len(patches)),
        fontsize=10,
        frameon=True,
        framealpha=0.15,
        facecolor="#ffffff",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO HANDLER
# ──────────────────────────────────────────────────────────────────────────────
def segment(image: Image.Image, alpha: float) -> tuple[Image.Image, str]:
    if image is None:
        return None, "⚠️ Please upload an image first."
    if model is None:
        return None, "❌ Model failed to load. Check the MODEL_REPO env variable."

    try:
        image_rgb    = image.convert("RGB")
        semantic_map = run_inference(image_rgb)
        figure       = build_figure(image_rgb, semantic_map, alpha)
        coverage     = class_iou(semantic_map)

        stats_lines = ["**Detected classes (pixel coverage):**\n"]
        for name, pct in sorted(coverage.items(), key=lambda x: -x[1]):
            bar   = "█" * int(pct / 5)
            stats_lines.append(f"`{name:<18}` {bar} **{pct}%**")

        return figure, "\n".join(stats_lines)

    except Exception as e:
        return None, f"❌ Inference error: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# GRADIO UI
# ──────────────────────────────────────────────────────────────────────────────
EXAMPLES = []          # add example image paths here if you bundle some
DESCRIPTION = """
<div style="text-align:center; margin-bottom: 12px;">
  <h1 style="font-size:2rem; font-weight:800; margin-bottom:4px;">
    🏜️ Offroad Semantic Segmentation
  </h1>
  <p style="color:#888; font-size:1rem;">
    Mask2Former · Swin-Base · Fine-tuned on 2,857 offroad images · 11 terrain classes
  </p>
  <p style="color:#888; font-size:0.9rem;">
    Upload any offroad / desert / trail image and get a per-pixel class prediction.
  </p>
</div>
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f0f1a",
        block_background_fill="#1a1a2e",
        block_border_color="#2a2a4a",
        block_title_text_color="#e0e0ff",
        label_background_fill="#1a1a2e",
        input_background_fill="#12122a",
        button_primary_background_fill="#e8740c",
        button_primary_background_fill_hover="#f59320",
        button_primary_text_color="white",
    ),
    title="Offroad Segmentation",
) as demo:

    gr.HTML(DESCRIPTION)

    with gr.Row():
        # ── LEFT: inputs ──────────────────────────────────────────────────
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="📷 Upload Offroad Image",
                height=320,
            )
            alpha_slider = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                label="🎨 Overlay Blend  (0 = original, 1 = full mask)",
            )
            run_btn = gr.Button("🚀 Run Segmentation", variant="primary", size="lg")

        # ── RIGHT: outputs ────────────────────────────────────────────────
        with gr.Column(scale=2):
            fig_output   = gr.Image(
                label="Segmentation Result",
                type="pil",
                height=400,
                show_download_button=True,
            )
            stats_output = gr.Markdown(label="Class Coverage")

    run_btn.click(
        fn=segment,
        inputs=[img_input, alpha_slider],
        outputs=[fig_output, stats_output],
    )
    img_input.change(
        fn=segment,
        inputs=[img_input, alpha_slider],
        outputs=[fig_output, stats_output],
    )

    gr.Markdown("""
---
**Model**: `Mask2Former-Swin-Base` fine-tuned from `facebook/mask2former-swin-base-IN21k-ade-semantic`  
**Classes**: background · trees · lush bushes · dry grass · dry bushes · ground clutter · flowers · logs · rocks · landscape · sky  
**Val mIoU**: ~60% &nbsp;|&nbsp; **Val Pixel Acc**: ~87.9%
""")

if __name__ == "__main__":
    demo.launch(share=False)
