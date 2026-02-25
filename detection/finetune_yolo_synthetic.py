
from ultralytics import YOLO
import os

# Configuration
PRETRAINED_WEIGHTS = "/home/AI_CAC/results/yolo_training/train_run/weights/best.pt"
DATA_YAML = "/home/AI_CAC/datasets/dagm_yolo_synthetic/data.yaml"
OUTPUT_DIR = "/home/AI_CAC/results/yolo_finetune_synthetic"

def main():
    print(f"Loading model from {PRETRAINED_WEIGHTS}...")
    model = YOLO(PRETRAINED_WEIGHTS)
    
    print(f"Starting finetuning on synthetic dataset for 5 epochs...")
    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=5,
        imgsz=512, # Matching synthesis size
        batch=16, # Adjust based on VRAM
        project=OUTPUT_DIR,
        name="finetune_run",
        exist_ok=True,
        optimizer='AdamW', # Good for fine-tuning
        lr0=1e-4, # Lower LR for fine-tuning
        warmup_epochs=0, # No warmup needed for fine-tuning usually
        plots=True
    )
    
    print(f"Training complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
