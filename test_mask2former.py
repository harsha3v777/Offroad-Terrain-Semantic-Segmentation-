import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

# =====================================================================
# CONFIGURATION
# =====================================================================
ID2LABEL = {
    0: "background", 1: "trees", 2: "lush bushes", 3: "dry grass",
    4: "dry bushes", 5: "ground clutter", 6: "flowers", 7: "logs",
    8: "rocks", 9: "landscape", 10: "sky",
}

CLASS_COLORS = {
    0: (0, 0, 0),       1: (34, 139, 34),   2: (0, 255, 0),
    3: (210, 180, 140), 4: (139, 90, 43),   5: (128, 128, 128),
    6: (255, 105, 180), 7: (101, 67, 33),   8: (169, 169, 169),
    9: (222, 184, 135), 10: (135, 206, 235),
}

def mask_to_color(mask_2d):
    """Convert class index mask to an RGB image."""
    h, w = mask_2d.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS.items():
        color[mask_2d == c] = rgb
    return color

def main():
    parser = argparse.ArgumentParser(description="Test Mask2Former Model")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--model-dir", type=str, default="./runs_mask2former/mask2former_best", help="Path to the trained model directory")
    parser.add_argument("--output", type=str, default="test_output.png", help="Path to save the result image")
    parser.add_argument("--alpha", type=float, default=0.5, help="Opacity of the overlaid mask (0.0 to 1.0)")
    args = parser.parse_args()

    # 1. Device and Model Check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Executing inference on: {device}")
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model directory '{args.model_dir}' not found.")
        print("Run the training script first to generate your best model.")
        return
        
    if not os.path.exists(args.image):
        print(f"❌ Error: Image '{args.image}' not found.")
        return

    # 2. Load Model & Processor
    print(f"⏳ Loading trained Mask2Former from '{args.model_dir}'...")
    try:
        processor = Mask2FormerImageProcessor.from_pretrained(args.model_dir)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model_dir).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    print("✅ Model loaded successfully!")

    # 3. Load Image
    img_pil = Image.open(args.image).convert("RGB")
    orig_w, orig_h = img_pil.size
    print(f"🖼️  Loaded image '{os.path.basename(args.image)}' (Size: {orig_w}x{orig_h})")

    # 4. Run Inference
    print("⏳ Running inference...")
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to original size
    results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(orig_h, orig_w)])[0]
    semantic_map = results.cpu().numpy().astype(np.uint8)
    print("✅ Inference complete!")

    # 5. Visualization Preparation
    color_mask = mask_to_color(semantic_map)
    color_mask_pil = Image.fromarray(color_mask)
    
    # Create an overlaid image (Original Image + Mask)
    blended_img = Image.blend(img_pil, color_mask_pil, alpha=args.alpha)

    # 6. Plot and Save Complete Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(color_mask_pil)
    axes[1].set_title("Mask2Former Prediction", fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(blended_img)
    axes[2].set_title(f"Blended Overlay (Alpha={args.alpha})", fontweight='bold')
    axes[2].axis('off')

    # Add Legend
    patches = [mpatches.Patch(color=np.array(col)/255, label=f'{i}: {ID2LABEL[i]}') 
               for i, col in CLASS_COLORS.items()]
    fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.05))

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.savefig(args.output, dpi=150)
    print(f"\n🎉 Success! Visualization saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
