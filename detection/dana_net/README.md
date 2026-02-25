DANA-Net (Density-Aware Nuclear Analysis Network)

This folder contains a compact, reproducibility-focused implementation of DANA-Net and a small smoke-test trainer.

Files:
- `model.py` : compact PyTorch implementation of the two-stage DANA-Net (backbone, density head, detector head).
- `train_dana.py` : small script that uses the repository's `DataSynthesizer` to generate quick synthetic patches and runs a 1-epoch smoke test training loop.
- `dana_smoke.pth` : (created by running `train_dana.py`) lightweight checkpoint example.

Quick test (from repo root):

```bash
python3 repro_package_pami/detection/dana_net/train_dana.py
```

Notes:
- The training script is deliberately minimal and intended to verify end-to-end functionality and provide a starting point for full training pipelines (adapt to dataset loaders and box labels as needed).
- For full-scale training, integrate `DANA_Net` into the project's existing `finetune_yolo_synthetic.py` workflow or write a dataset class that reads detection boxes/annotations.
