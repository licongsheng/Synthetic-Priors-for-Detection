
import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from generate_scratch_defects import ScratchSynthesizer

DAGM_ROOT = "/home/AI_CAC/OtherDataset4PAMI/DAGM_KaggleUpload"
OUTPUT_DIR = "/home/AI_CAC/results/dagm_fusion_validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_CLASSES = [2, 4, 6, 8, 10]

def load_normal_sample(class_id):
    """Load a random normal (defect-free) image from Train set"""
    class_path = os.path.join(DAGM_ROOT, f"Class{class_id}", "Train")
    all_imgs = glob.glob(os.path.join(class_path, "*.PNG"))
    all_imgs = [p for p in all_imgs if "_label" not in p]
    
    # Filter out label files themselves just in case
    # Identify defective images
    label_path = os.path.join(class_path, "Label")
    defective_basenames = []
    if os.path.exists(label_path):
        labels = glob.glob(os.path.join(label_path, "*_label.PNG"))
        for l in labels:
            base = os.path.basename(l).replace("_label.PNG", ".PNG")
            defective_basenames.append(base)
            
    normal_imgs = [p for p in all_imgs if os.path.basename(p) not in defective_basenames]
    
    if not normal_imgs:
        # Fallback if filtering fails or dataset structure differs slightly
        return random.choice(all_imgs)
        
    return random.choice(normal_imgs)

def blend_scratch_refined(img, mask, mode='dark', opacity=1.0, offset_mask=None):
    """
    Refined blending.
    img: 0-255 uint8
    mask: 0.0-1.0 float
    mode: 'dark', 'bright', 'ridge'
    opacity: 0.0-1.0 strength scalar
    """
    img_float = img.astype(np.float32)
    max_intensity_shift = 200.0 * opacity # Tune this. 200 is strong.
    
    if mode == 'ridge':
        # Dark line + Bright line
        # Use mask as Dark, offset_mask as Bright
        if offset_mask is None:
            # Fallback: create offset by shifting mask
            M = np.float32([[1, 0, 3], [0, 1, 3]]) # Shift 3,3 pixels
            h, w = mask.shape
            offset_mask = cv2.warpAffine(mask, M, (w, h))
            
        # Apply Subtract (Dark) then Add (Bright)
        # Class 8 description: "Black line accompanied by white long line"
        blended = img_float - mask * max_intensity_shift
        blended = blended + offset_mask * (max_intensity_shift * 0.8) # White line slightly weaker?
        
    elif mode == 'dark':
        blended = img_float - mask * max_intensity_shift
    elif mode == 'bright':
        blended = img_float + mask * max_intensity_shift
    else:
        blended = img_float
        
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended

def process_and_visualize():
    syn = ScratchSynthesizer(canvas_size=(512, 512))
    
    fig, axes = plt.subplots(len(TARGET_CLASSES), 3, figsize=(12, 4 * len(TARGET_CLASSES)))
    fig.suptitle("Refined Synthesis (Classes 2,4,6,8,10)", fontsize=16)
    
    ax_idx = 0
    for cls_id in TARGET_CLASSES:
        img_path = load_normal_sample(cls_id)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        # --- Class Specific Parameters ---
        # Defaults
        width = 3.0
        waviness = 2.0
        opacity = 0.8
        mode = 'dark'
        ridge_offset = None
        
        if cls_id == 2:
            # Lines complex but smoother arc
            width = random.uniform(2.0, 4.0)
            waviness = random.uniform(0.0, 1.0) # Greatly reduced
            opacity = 0.9
            mode = 'dark'
            
        elif cls_id == 4:
            # Grey-black (less dark), smoother
            width = random.uniform(2.0, 3.5)
            waviness = random.uniform(0.0, 0.5) # Greatly reduced
            opacity = 0.4 
            mode = 'dark'
            
        elif cls_id == 6:
            # Black, Wide, simple
            width = random.uniform(5.0, 9.0)
            waviness = 0.0 # Straight/Simple arc
            opacity = 1.0 
            mode = 'dark'
            
        elif cls_id == 8:
            # Black line + White line (Ridge), smooth
            width = random.uniform(2.0, 3.5)
            waviness = random.uniform(0.0, 0.5)
            opacity = 0.8
            mode = 'ridge'
            
        elif cls_id == 10:
            # Dark black (not full), smoother arc
            width = random.uniform(2.5, 4.5)
            waviness = random.uniform(0.0, 1.0)
            opacity = 0.65 # Increased from 0.5
            mode = 'dark'

        # Generate Mask - Force internal placement for demo
        h, w = 512, 512
        # Pick a start point somewhat central
        sx = random.randint(100, w-100)
        sy = random.randint(100, h-100)
        start_pos = (sx, sy)
        
        # Pick an end point within bounds
        angle = random.uniform(0, 2*np.pi)
        length = random.uniform(50, 200) # Shorter length to ensure it fits
        ex = sx + np.cos(angle) * length
        ey = sy + np.sin(angle) * length
        # Clamp to be safe, though cosmetic
        ex = np.clip(ex, 50, w-50)
        ey = np.clip(ey, 50, h-50)
        end_pos = (ex, ey)

        mask = syn.generate_scratch(start_pos=start_pos, end_pos=end_pos, width=width, roughness=0.6, waviness=waviness, intensity=1.0)
        
        # Special handling for Ridge mode (Class 8)
        offset_mask = None
        if mode == 'ridge':
            # Create a "partner" line that is shifted
            # Shift perpendicular to average direction or just fixed shift
            shift = int(width * 1.5) 
            # Simple shift
            M = np.float32([[1, 0, shift], [0, 1, shift]])
            h, w = mask.shape
            offset_mask = cv2.warpAffine(mask, M, (w, h))

        # Blend
        blended = blend_scratch_refined(img, mask, mode=mode, opacity=opacity, offset_mask=offset_mask)
        
        # Plot
        ax_row = axes[ax_idx]
        
        # Original
        ax_row[0].imshow(img, cmap='gray')
        ax_row[0].set_title(f"Class {cls_id} Normal")
        ax_row[0].axis('off')
        
        # Mask (Show combined for Ridge)
        if mode == 'ridge':
            combined_vis = mask - offset_mask
            ax_row[1].imshow(combined_vis, cmap='bwr') # Blue=Dark, Red=Bright
            ax_row[1].set_title(f"Ridge Mask\n(w={width:.1f}, wav={waviness:.1f})")
        else:
            ax_row[1].imshow(mask, cmap='jet')
            ax_row[1].set_title(f"Mask (Op={opacity})\n(w={width:.1f}, wav={waviness:.1f})")
        ax_row[1].axis('off')
        
        # Result
        ax_row[2].imshow(blended, cmap='gray')
        ax_row[2].set_title(f"Refined Fusion ({mode})")
        ax_row[2].axis('off')

        ax_idx += 1

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fusion_refined_comparison.png")
    plt.savefig(save_path)
    print(f"Saved refined visualization to {save_path}")

if __name__ == "__main__":
    process_and_visualize()
