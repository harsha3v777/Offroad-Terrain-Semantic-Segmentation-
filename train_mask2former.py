import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import gc

from huggingface_hub import login
from transformers import (
    Mask2FormerImageProcessor, 
    Mask2FormerForUniversalSegmentation,
    Sam3Processor, 
    Sam3Model
)

# =====================================================================
# CONFIGURATION
# =====================================================================
ID2LABEL = {
    0 : "background", 1 : "trees", 2 : "lush bushes", 3 : "dry grass",
    4 : "dry bushes", 5 : "ground clutter", 6 : "flowers", 7 : "logs",
    8 : "rocks", 9 : "landscape", 10: "sky",
}
LABEL2ID  = {v: k for k, v in ID2LABEL.items()}
N_CLASSES = len(ID2LABEL)

VALUE_MAP = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 
    550: 5, 600: 6, 700: 7, 800: 8, 7100: 9, 10000: 10,
}

CLASS_COLORS = {
    0 : (0,   0,   0),   1 : (34,  139, 34), 2 : (0,   255, 0),
    3 : (210, 180, 140), 4 : (139, 90,  43), 5 : (128, 128, 128),
    6 : (255, 105, 180), 7 : (101, 67,  33), 8 : (169, 169, 169),
    9 : (222, 184, 135), 10: (135, 206, 235),
}


# =====================================================================
# DATASET AND UTILS
# =====================================================================
def convert_mask(mask_pil):
    """Convert raw pixel IDs → class indices"""
    arr     = np.array(mask_pil)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, cls in VALUE_MAP.items():
        new_arr[arr == raw] = cls
    return new_arr

def mask_to_color(mask_2d):
    """Class index mask → RGB"""
    h, w  = mask_2d.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in CLASS_COLORS.items():
        color[mask_2d == c] = rgb
    return color

class DesertSegDataset(Dataset):
    def __init__(self, data_dir, processor, augment=False):
        self.img_dir   = os.path.join(data_dir, 'Color_Images')
        self.mask_dir  = os.path.join(data_dir, 'Segmentation')
        self.processor = processor
        self.augment   = augment
        self.fnames    = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        image    = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        mask_raw = Image.open(os.path.join(self.mask_dir, fname))
        mask     = convert_mask(mask_raw)

        if self.augment:
            import albumentations as A
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
                A.GaussianBlur(p=0.2),
            ])
            augmented = transform(image=np.array(image), mask=mask)
            image = Image.fromarray(augmented['image'])
            mask  = augmented['mask']

        inputs = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )
        return inputs

def collate_fn(batch):
    """Custom collate for Mask2Former variable-length masks"""
    pixel_values = torch.stack([x["pixel_values"].squeeze() for x in batch])
    pixel_mask   = torch.stack([x["pixel_mask"].squeeze()   for x in batch])
    class_labels = [x["class_labels"][0] for x in batch]
    mask_labels  = [x["mask_labels"][0] for x in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask"  : pixel_mask,
        "class_labels": class_labels,
        "mask_labels" : mask_labels,
    }

def compute_miou_batch(pred_logits, gt_masks_list, processor, device, target_sizes, num_classes=N_CLASSES):
    sem_maps = processor.post_process_semantic_segmentation(pred_logits, target_sizes=target_sizes)
    batch_ious = []
    for pred_map, gt in zip(sem_maps, gt_masks_list):
        pred = pred_map.cpu().numpy().astype(np.int32)
        gt   = gt.astype(np.int32)
        class_ious = []
        for c in range(num_classes):
            inter = np.logical_and(pred == c, gt == c).sum()
            union = np.logical_or( pred == c, gt == c).sum()
            if union == 0: continue
            class_ious.append(inter / union)
        if class_ious:
            batch_ious.append(np.mean(class_ious))
    return float(np.mean(batch_ious)) if batch_ious else 0.0

def compute_iou_numpy(pred, gt, num_classes=N_CLASSES):
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (gt == c)).sum()
        union = ((pred == c) | (gt == c)).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(inter / union)
    return np.nanmean(ious)


# =====================================================================
# SAM3 INFERENCE AND REFINEMENT
# =====================================================================
def run_mask2former_inference(image_pil, model, processor, device, conf_threshold=0.7):
    model = model.to(device)
    model.eval()

    inputs = processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image_pil.size[::-1]]
    )[0]

    semantic_map = result.cpu().numpy()
    logits = outputs.masks_queries_logits[0]
    probs  = torch.sigmoid(logits).max(dim=0).values
    probs  = F.interpolate(
        probs.unsqueeze(0).unsqueeze(0),
        size=image_pil.size[::-1], mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()

    low_conf_mask = probs < conf_threshold
    return semantic_map, probs, low_conf_mask

def run_sam3_refinement(image_pil, semantic_map, low_conf_mask, sam3_model, sam3_processor, device, conf_threshold=0.7):
    """
    SAM3 refinement: Only write to pixels where Mask2Former had LOW confidence.
    We use SAM3's bounding-box prompt per uncertain class region to sharpen edges.
    If SAM3 returns no mask, we leave the original Mask2Former prediction unchanged.
    """
    orig_w, orig_h = image_pil.size
    refined_map    = semantic_map.copy()

    for class_id, class_name in ID2LABEL.items():
        if class_name == "background": continue
        class_region    = (semantic_map == class_id)
        if not class_region.any(): continue

        uncertain_class = class_region & low_conf_mask
        if not uncertain_class.any(): continue

        rows = np.where(uncertain_class.any(axis=1))[0]
        cols = np.where(uncertain_class.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0: continue

        x1, y1 = int(cols.min()), int(rows.min())
        x2, y2 = int(cols.max()), int(rows.max())
        if (x2 - x1) < 4 or (y2 - y1) < 4: continue  # skip tiny boxes

        try:
            box_inputs = sam3_processor(
                images=image_pil, input_boxes=[[[x1, y1, x2, y2]]],
                input_boxes_labels=[[1]], return_tensors="pt"
            ).to(device)
            box_inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in box_inputs.items()}

            with torch.no_grad():
                box_outputs = sam3_model(**box_inputs)

            box_results = sam3_processor.post_process_instance_segmentation(
                box_outputs, threshold=0.5, mask_threshold=0.5, target_sizes=[[orig_h, orig_w]]
            )[0]

            if len(box_results['masks']) > 0:
                best_mask    = box_results['masks'][0].cpu().numpy()
                # Only overwrite the specific uncertain pixels that SAM3 confirms
                write_region = best_mask & uncertain_class
                refined_map[write_region] = class_id
        except Exception:
            continue  # if SAM3 fails on this class, leave Mask2Former prediction intact

    return refined_map


# =====================================================================
# MAIN ROUTINE
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="ForgeX Pipeline: Mask2Former Training + SAM3 Refinement")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--hf-token", type=str, default="", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--train-dir", type=str, default="./Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/train")
    parser.add_argument("--val-dir", type=str, default="./Offroad_Segmentation_Training_Dataset/Offroad_Segmentation_Training_Dataset/val")
    parser.add_argument("--runs-dir", type=str, default="./runs_mask2former")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation with SAM3")
    args = parser.parse_args()

    # Hardware check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Executing on: {device}")
    
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ HuggingFace login successful!")
    else:
        print("⚠️ No HF_TOKEN provided. If models are gated (like facebook/sam3), the script will fail.")

    os.makedirs(args.runs_dir, exist_ok=True)
    
    m2f_processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-IN21k-ade-semantic",
        ignore_index=255, reduce_labels=False, do_resize=True, size={"height": 512, "width": 512},
    )

    if not args.eval_only:
        print("\n--- STAGE 1: TRAINING MASK2FORMER ---")
        train_ds = DesertSegDataset(args.train_dir, m2f_processor, augment=True)
        val_ds   = DesertSegDataset(args.val_dir, m2f_processor, augment=False)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
        print(f"✅ Loaded {len(train_ds)} train | {len(val_ds)} val samples")

        model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-IN21k-ade-semantic",
            id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True,
        ).to(device)

        # Freeze backbone
        for name, param in model_m2f.named_parameters():
            if "model.pixel_level_module.encoder" in name:
                param.requires_grad = False

        optimizer = AdamW(filter(lambda p: p.requires_grad, model_m2f.parameters()), lr=5e-5, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_loss = float('inf')
        train_losses, val_losses, val_mious, val_accs = [], [], [], []

        for epoch in range(args.epochs):
            model_m2f.train()
            t_losses = []
            for batch in tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs} [Train]"):
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask   = batch["pixel_mask"].to(device)
                class_labels = [c.to(device) for c in batch["class_labels"]]
                mask_labels  = [m.to(device) for m in batch["mask_labels"]]
                
                # Check empty masks
                if any(m.ndim != 3 or m.shape[0] == 0 for m in mask_labels): continue

                try:
                    outputs = model_m2f(pixel_values=pixel_values, pixel_mask=pixel_mask, class_labels=class_labels, mask_labels=mask_labels)
                    if torch.isnan(outputs.loss) or torch.isinf(outputs.loss): continue
                    
                    optimizer.zero_grad()
                    outputs.loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_m2f.parameters(), max_norm=1.0)
                    optimizer.step()
                    t_losses.append(outputs.loss.item())
                except RuntimeError as e:
                    optimizer.zero_grad(); continue

            model_m2f.eval()
            v_losses, v_mious, v_correct, v_total = [], [], 0, 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{args.epochs} [Val]"):
                    pixel_values = batch["pixel_values"].to(device)
                    pixel_mask   = batch["pixel_mask"].to(device)
                    class_labels = [c.to(device) for c in batch["class_labels"]]
                    mask_labels  = [m.to(device) for m in batch["mask_labels"]]

                    if any(m.ndim != 3 or m.shape[0] == 0 for m in mask_labels): continue
                    
                    try:
                        outputs = model_m2f(pixel_values=pixel_values, pixel_mask=pixel_mask, class_labels=class_labels, mask_labels=mask_labels)
                        v_losses.append(outputs.loss.item())

                        H, W = pixel_values.shape[-2:]
                        gt_semantic_list = []
                        for ml, cl in zip(batch["mask_labels"], batch["class_labels"]):
                            gt_map = np.zeros((H, W), dtype=np.int32)
                            for mask_i, class_id in zip(ml.cpu().numpy(), cl.cpu().numpy()):
                                mask_t = mask_i if mask_i.shape == (H, W) else F.interpolate(torch.tensor(mask_i).float().unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze().numpy()
                                gt_map[mask_t > 0.5] = int(class_id)
                            gt_semantic_list.append(gt_map)
                        
                        target_sizes = [(H, W)] * len(gt_semantic_list)
                        v_mious.append(compute_miou_batch(outputs, gt_semantic_list, m2f_processor, device, target_sizes))

                        sem_maps = m2f_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
                        for pred_map, gt in zip(sem_maps, gt_semantic_list):
                            pred = pred_map.cpu().numpy()
                            v_correct += (pred == gt).sum()
                            v_total   += gt.size
                    except RuntimeError:
                        continue

            tl, vl = np.mean(t_losses) if t_losses else float('nan'), np.mean(v_losses) if v_losses else float('nan')
            miou = np.mean(v_mious) if v_mious else float('nan')
            acc = v_correct / v_total if v_total > 0 else 0.0

            train_losses.append(tl); val_losses.append(vl); val_mious.append(miou); val_accs.append(acc)
            scheduler.step()

            print(f"Epoch {epoch+1:02d} | Train: {tl:.4f} | Val: {vl:.4f} | Val mIoU: {miou*100:.2f}% | Val Acc: {acc*100:.2f}%")
            if vl < best_loss:
                best_loss = vl
                model_m2f.save_pretrained(f'{args.runs_dir}/mask2former_best')
                m2f_processor.save_pretrained(f'{args.runs_dir}/mask2former_best')
                print(f"💾 Best Model Saved -> Val Loss = {vl:.4f}")

        # Plot training stats
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(train_losses, label='Train Loss', color='red'); axes[0].plot(val_losses, label='Val Loss', color='orange')
        axes[0].set_title('Training Loss'); axes[0].legend()
        axes[1].plot([m*100 for m in val_mious], label='Val mIoU (%)', color='blue', marker='o')
        axes[1].set_title('Validation mIoU'); axes[1].legend()
        plt.tight_layout()
        plt.savefig(f'{args.runs_dir}/training_curve.png')
        print(f"📈 Training curves saved to {args.runs_dir}/training_curve.png")

        print("📤 Offloading training model from GPU...")
        model_m2f = model_m2f.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- STAGE 2: FINAL EVALUATION ---")
    if not os.path.exists(f'{args.runs_dir}/mask2former_best'):
        print(f"❌ Could not find a trained model at {args.runs_dir}/mask2former_best to evaluate. Exiting.")
        return

    model_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(f'{args.runs_dir}/mask2former_best').to(device)
    m2f_proc  = Mask2FormerImageProcessor.from_pretrained(f'{args.runs_dir}/mask2former_best')
    model_m2f.eval()

    val_img_dir  = os.path.join(args.val_dir, 'Color_Images')
    val_mask_dir = os.path.join(args.val_dir, 'Segmentation')

    if os.path.exists(val_img_dir):
        all_files  = sorted(os.listdir(val_img_dir))
        test_files = random.sample(all_files, min(6, len(all_files)))

        fig, axes = plt.subplots(len(test_files), 3, figsize=(18, 5 * len(test_files)))
        if len(test_files) == 1: axes = [axes]
        col_titles = ['Original Image', 'Ground Truth Mask', 'Mask2Former Prediction']
        for ax, t in zip(axes[0], col_titles):
            ax.set_title(t, fontsize=13, fontweight='bold')

        ious = []
        for row, fname in enumerate(test_files):
            print(f"\nProcessing: {fname}")
            img_pil = Image.open(f'{val_img_dir}/{fname}').convert("RGB")
            gt_mask = convert_mask(Image.open(f'{val_mask_dir}/{fname}'))

            semantic_map, _, _ = run_mask2former_inference(img_pil, model_m2f, m2f_proc, device)
            iou = compute_iou_numpy(semantic_map, gt_mask)
            ious.append(iou)
            print(f"  IoU: {iou:.4f} ({iou*100:.2f}%)")

            axes[row][0].imshow(img_pil.resize((semantic_map.shape[1], semantic_map.shape[0]))); axes[row][0].axis('off')
            axes[row][1].imshow(mask_to_color(np.array(Image.fromarray(gt_mask).resize((semantic_map.shape[1], semantic_map.shape[0]), Image.NEAREST)))); axes[row][1].axis('off')
            axes[row][2].imshow(mask_to_color(semantic_map)); axes[row][2].axis('off')
            axes[row][0].set_ylabel(fname[:16], fontsize=9, rotation=0, labelpad=80)

        patches = [mpatches.Patch(color=np.array(col)/255, label=f'{i}: {ID2LABEL[i]}') for i, col in CLASS_COLORS.items()]
        fig.legend(handles=patches, loc='lower center', ncol=6, fontsize=10, bbox_to_anchor=(0.5, 0.02))
        plt.suptitle(f'Mask2Former Offroad Segmentation — Mean IoU: {np.mean(ious)*100:.2f}%', fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{args.runs_dir}/evaluation_results.png', dpi=120, bbox_inches='tight')
        print(f"\n🖼️  Visualizations saved → {args.runs_dir}/evaluation_results.png")
        print(f"📊 Mean IoU across {len(ious)} samples: {np.mean(ious)*100:.2f}%")

if __name__ == "__main__":
    main()
