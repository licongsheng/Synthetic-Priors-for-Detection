
import os
import cv2
import random
import numpy as np
import sys

# Ensure we can import the synthesizer
sys.path.append("/home/AI_CAC/src/generation")
from generate_scratch_defects import ScratchSynthesizer

DAGM_ROOT = "/home/AI_CAC/OtherDataset4PAMI/DAGM_KaggleUpload"
OUTPUT_DIR = "/home/AI_CAC/docs/assets"

TARGET_CLASSES = [2, 4, 6, 8, 10]

def load_normal_sample(class_id):
    path = os.path.join(DAGM_ROOT, f"Class{class_id}", "Train")
    # Quick globs
    import glob
    files = glob.glob(os.path.join(path, "*.PNG"))
    # Filter out labels just in case
    files = [f for f in files if "_label" not in f]
    if not files: return None
    return random.choice(files)

def blend_ridge(img, mask, opacity):
    img_float = img.astype(np.float32)
    shift_amount = 200.0 * opacity
    
    # Create offset for highlight
    M = np.float32([[1, 0, 3], [0, 1, 3]]) # Shift 3,3
    h, w = mask.shape
    offset_mask = cv2.warpAffine(mask, M, (w, h))
    
    # Darken then lighten
    blended = img_float - mask * shift_amount
    blended = blended + offset_mask * (shift_amount * 0.8)
    return np.clip(blended, 0, 255).astype(np.uint8)

def blend_dark(img, mask, opacity):
    img_float = img.astype(np.float32)
    shift_amount = 200.0 * opacity
    blended = img_float - mask * shift_amount
    return np.clip(blended, 0, 255).astype(np.uint8)

def generate_demos():
    syn = ScratchSynthesizer(canvas_size=(512, 512))
    
    for cls_id in TARGET_CLASSES:
        print(f"Generating demo for Class {cls_id}...")
        img_path = load_normal_sample(cls_id)
        if img_path is None:
            print(f"Skipping Class {cls_id}, no images found.")
            continue
            
        bg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        bg = cv2.resize(bg, (512, 512)) # Ensure fit
        
        # Params from the HTML text
        if cls_id == 2:
            width = 3.0; waviness = 0.5; opacity = 0.9; mode = 'dark'
        elif cls_id == 4:
            width = 3.0; waviness = 0.2; opacity = 0.4; mode = 'dark'
        elif cls_id == 6:
            width = 7.0; waviness = 0.0; opacity = 1.0; mode = 'dark'
        elif cls_id == 8:
            width = 3.0; waviness = 0.3; opacity = 0.8; mode = 'ridge'
        elif cls_id == 10:
            width = 3.5; waviness = 0.5; opacity = 0.65; mode = 'dark'

        # Generate a nice visible scratch
        attempts = 0
        while attempts < 5:
            # Force central
            sx = random.randint(100, 400)
            sy = random.randint(100, 400)
            angle = random.uniform(0, 6.28)
            length = 200
            ex = sx + np.cos(angle)*length
            ey = sy + np.sin(angle)*length
            
            mask = syn.generate_scratch(start_pos=(sx,sy), end_pos=(ex,ey), 
                                      width=width, waviness=waviness, 
                                      intensity=1.0, roughness=0.5)
            
            if np.max(mask) > 0.1:
                break
            attempts += 1
            
        # Blend
        if mode == 'ridge':
            res = blend_ridge(bg, mask, opacity)
        else:
            res = blend_dark(bg, mask, opacity)
            
        # Add a colored bounding box for visualization? No, keep it raw but maybe crop?
        # Let's save the full 512x512
        out_path = os.path.join(OUTPUT_DIR, f"class_{cls_id}_demo.png")
        cv2.imwrite(out_path, res)
        
        # Also create a small crop to zoom in on feature? 
        # Maybe just use the full image, but scale it down in HTML.

if __name__ == "__main__":
    generate_demos()
