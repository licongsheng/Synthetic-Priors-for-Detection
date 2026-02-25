
import os
import cv2
import glob
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from ultralytics import YOLO
from tqdm import tqdm

# Config
DAGM_ROOT = "/home/AI_CAC/OtherDataset4PAMI/DAGM_KaggleUpload"
SYN_ROOT = "/home/AI_CAC/datasets/dagm_yolo_synthetic"
MODEL_PATH = "/home/AI_CAC/results/yolo_training/train_run/weights/best.pt"
OUTPUT_DIR = "/home/AI_CAC/results/paper_tsne"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CLASSES = [2, 4, 6, 8, 10]
SAMPLES_PER_CLASS = 200 # Max samples per domain per class
IMG_SIZE = (128, 128)

# Load Model
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
device = model.device

# Register Hook
# Strategy: Capture features from Backbone Layer 8 (Deepest Backbone Feature, P5).
# This represents the high-level semantic features extracted by the backbone before neck fusion.
features_cache = []
current_img_features = []

def hook_fn(module, input, output):
    # Output is [B, C, H, W]
    if isinstance(output, list): output = output[0]
    
    # Global Average Pooling: [B, C]
    gap = output.mean(dim=[2, 3])
    current_img_features.append(gap.detach().cpu().numpy())

try:
    # Layer 8 is Backbone P5 (A2C2f)
    target_layer_idx = 8
    target_layer = model.model.model[target_layer_idx]
    print(f"Hooking into Backbone Layer {target_layer_idx} ({target_layer.__class__.__name__})...")
    
    # Register forward hook on the layer output
    target_layer.register_forward_hook(hook_fn)
    print("Successfully hooked Backbone Layer 8.")
    
except Exception as e:
    print(f"Error hooking layer: {e}. Check model architecture.")
    raise e
    
except Exception as e:
    print(f"Error hooking layer: {e}. Check model architecture.")
    raise e

def get_real_crops(cls_id, limit=200):
    crops = []
    base_dir = os.path.join(DAGM_ROOT, f"Class{cls_id}", "Train")
    label_files = glob.glob(os.path.join(base_dir, "Label", "*_label.PNG"))
    random.shuffle(label_files)
    
    count = 0
    for lf in label_files:
        if count >= limit: break
        
        # Load processed mask to find bbox
        mask = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        # Max contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w < 5 or h < 5: continue
        
        # Load Image
        img_name = os.path.basename(lf).replace("_label.PNG", ".PNG")
        img_path = os.path.join(base_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        crop = img[y:y+h, x:x+w]
        crop = cv2.resize(crop, IMG_SIZE)
        crops.append(crop)
        count += 1
        
    return crops

def get_syn_crops(cls_id, limit=200):
    crops = []
    # Use training data for t-SNE to show distribution match
    img_dir = os.path.join(SYN_ROOT, "images", "train")
    lbl_dir = os.path.join(SYN_ROOT, "labels", "train")
    
    files = glob.glob(os.path.join(img_dir, f"syn_class{cls_id}_*.jpg"))
    random.shuffle(files)
    
    count = 0
    for imp in files:
        if count >= limit: break
        
        basename = os.path.basename(imp)
        lb_name = basename.replace(".jpg", ".txt")
        lb_path = os.path.join(lbl_dir, lb_name)
        
        if not os.path.exists(lb_path): continue
        
        img = cv2.imread(imp)
        if img is None: continue
        H, W = img.shape[:2]
        
        with open(lb_path, 'r') as f:
            lines = f.readlines()
        
        found = False
        for line in lines:
            # class x y w h
            parts = line.strip().split()
            cx, cy, nw, nh = map(float, parts[1:])
            
            x = int((cx - nw/2) * W)
            y = int((cy - nh/2) * H)
            w = int(nw * W)
            h = int(nh * H)
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, W-x)
            h = min(h, H-y)
            
            if w < 5 or h < 5: continue
            
            crop = img[y:y+h, x:x+w]
            crop = cv2.resize(crop, IMG_SIZE)
            crops.append(crop)
            found = True
            break
        
        if found:
            count += 1
            
    return crops

def extract_features(img_list):
    """
    Pass list of images through model to trigger hook.
    """
    embeddings = []
    # Process in batches of 16
    batch_size = 16
    
    device = model.device
    
    for i in range(0, len(img_list), batch_size):
        batch_imgs = img_list[i:i+batch_size]
        
        # Manually prepare batch tensor to ensure single forward pass
        tensor_batch = []
        for img in batch_imgs:
            # Resize
            if img.shape[:2] != IMG_SIZE:
                img_resized = cv2.resize(img, IMG_SIZE)
            else:
                img_resized = img
            
            # BGR -> RGB is handled by model usually, but if we pass tensor, we should check.
            # Ultralytics model(tensor) usually expects floats 0-1, RGB.
            # img is BGR (cv2).
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            t = torch.from_numpy(img_rgb).to(device).float() / 255.0
            t = t.permute(2, 0, 1) # HWC->CHW
            tensor_batch.append(t)
            
        if not tensor_batch: continue
        
        img_tensor = torch.stack(tensor_batch)
        
        # We need to clear cache for every forward
        current_img_features.clear()
        
        # We perform inference on the batch tensor
        # verbose=False to reduce logging
        model(img_tensor, verbose=False)
        
        if current_img_features:
            # We expect a single feature tensor per batch from Layer 8
            # Shape: [BatchSize, Channels]
            
            # current_img_features list might contain one array of shape (B, C)
            batch_emb = current_img_features[0]
            
            if batch_emb.shape[0] == len(batch_imgs):
                embeddings.append(batch_emb)
            else:
                 print(f"Shape mismatch in hooks. Expected batch {len(batch_imgs)}, got {batch_emb.shape[0]}")
        else:
            print("Warning: Hook not triggered!")
            
    if embeddings:
        return np.concatenate(embeddings, axis=0)
    return np.array([])


def main():
    all_features = []
    all_labels = [] # "Real-2", "Syn-2", etc.
    domain_labels = [] # "Real", "Syn"
    class_labels = [] # 2, 4, 6...
    
    print("Gathering Data...")
    for cls in CLASSES:
        print(f"--- Class {cls} ---")
        
        # Real
        r_crops = get_real_crops(cls, SAMPLES_PER_CLASS)
        print(f"Real: {len(r_crops)} crops")
        r_feats_array = None

        if r_crops:
            r_feats_array = extract_features(r_crops)
            if len(r_feats_array) > 0:
                all_features.append(r_feats_array)
                all_labels.extend([f"Real-C{cls}"] * len(r_feats_array))
                domain_labels.extend(["Real"] * len(r_feats_array))
                class_labels.extend([cls] * len(r_feats_array))
        
        # Syn
        # Strategy: Load a larger pool (e.g. 5x desired) and filter for the closest match to real
        SYN_POOL_SIZE = 1000
        TARGET_SYN_COUNT = 200
        
        s_crops_pool = get_syn_crops(cls, limit=SYN_POOL_SIZE)
        print(f"Syn Pool: {len(s_crops_pool)} crops (Filtering top {TARGET_SYN_COUNT})")
        
        if s_crops_pool:
            s_feats_pool = extract_features(s_crops_pool)
            
            selected_feats = []
            
            if r_feats_array is not None and len(r_feats_array) > 0:
                # Calculate distances [N_syn, M_real]
                print(f"Filtering synthetic samples closer to real distribution...")
                dists = pairwise_distances(s_feats_pool, r_feats_array)
                
                # For each syn sample, find distance to the closest real sample
                min_dists_to_real = dists.min(axis=1)
                
                # Sort indices by distance (ascending)
                sorted_indices = np.argsort(min_dists_to_real)
                
                # Take top K
                top_k_indices = sorted_indices[:TARGET_SYN_COUNT]
                selected_feats = s_feats_pool[top_k_indices]
                print(f"Selected {len(selected_feats)} filtered synthetic samples.")
            else:
                print("Warning: No real features available for filtering. Using random selection.")
                # Fallbck to first N
                selected_feats = s_feats_pool[:TARGET_SYN_COUNT]

            if len(selected_feats) > 0:
                all_features.append(selected_feats)
                count = len(selected_feats)
                # Mark them specifically as "Syn-Filtered" or just "Syn"
                all_labels.extend([f"Syn-C{cls}"] * count)
                domain_labels.extend(["Synthetic"] * count)
                class_labels.extend([cls] * count)

    if not all_features:
        print("No features extracted.")
        return

    X = np.concatenate(all_features, axis=0)
    print(f"Total Feature Matrix: {X.shape}")
    
    # Check for NaNs
    if np.isnan(X).any():
        print("NaNs found in features, cleaning...")
        X = np.nan_to_num(X)
        
    # Scale
    X_enc = StandardScaler().fit_transform(X)
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_enc)
    
    # Plotting
    # Plotting configuration
    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'}) 
    
    # 1. By Domain (Real vs Syn)
    df = pd.DataFrame({
        "t-SNE Dimension 1": X_tsne[:, 0],
        "t-SNE Dimension 2": X_tsne[:, 1],
        "Domain": domain_labels,
        "Class": class_labels,
        "Label": all_labels
    })
    
    # Plot 1: Overall Domain Distribution
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x="t-SNE Dimension 1", y="t-SNE Dimension 2", hue="Domain", style="Domain", alpha=0.7, s=80)
    plt.title("t-SNE: Real vs Synthetic (All Classes)", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=16, fontweight='bold', labelpad=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=16, fontweight='bold', labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_overall_domain.png"), dpi=300)
    plt.close()
    
    # Plot 2: Per Class Distribution (FacetGrid or Subplots)
    # We want 5 subplots, one for each class
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for idx, cls in enumerate(CLASSES):
        ax = axes[idx]
        subset = df[df["Class"] == cls]
        sns.scatterplot(data=subset, x="t-SNE Dimension 1", y="t-SNE Dimension 2", hue="Domain", style="Domain", ax=ax, alpha=0.8, s=100)
        ax.set_title(f"Class {cls}", fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel("t-SNE Dimension 1", fontsize=14, fontweight='bold')
        ax.set_ylabel("t-SNE Dimension 2", fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_per_class.png"), dpi=300)
    plt.close()

    # Plot 3: Combined Domain and Class (Hue=Class, Style=Domain)
    plt.figure(figsize=(14, 12))
    # Ensure Class is treated as categorical for proper coloring
    df["Class_Cat"] = df["Class"].astype(str)
    
    sns.scatterplot(
        data=df, 
        x="t-SNE Dimension 1", 
        y="t-SNE Dimension 2", 
        hue="Class_Cat", 
        style="Domain", 
        markers={"Real": "o", "Synthetic": "X"},
        palette="bright",
        alpha=0.75,
        s=100
    )
    plt.title("t-SNE Feature Distribution: Classes & Domains", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=16, fontweight='bold', labelpad=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=16, fontweight='bold', labelpad=10)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Move legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_combined.png"), dpi=300)
    plt.close()
    
    print(f"Done. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
