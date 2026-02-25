Detection artifacts and suggested commands

This project uses YOLO-based training scripts under `src/` (finetune_yolo_synthetic.py, etc.). Suggested workflows:

- Fine-tune YOLO on synthetic dataset (example):

```bash
python3 src/finetune_yolo_synthetic.py --data data/synth_dataset.yaml --weights yolov8s.pt --epochs 5 --project results/yolo_training
```

- Extract features for t-SNE and filtering (used during paper analysis):

```bash
python3 src/analysis/generate_tsne_yolo_features.py --weights results/yolo_training/train_run/weights/best.pt --output results/paper_tsne
```

Notes:
- Several pretrained weights are present in repository root (e.g., `mfec_domain_adapted.pth`, `yolo12s.pt`). Use these as starting points.
- If using CUDA, ensure `torch` and `torchvision` are installed matching your CUDA version.
