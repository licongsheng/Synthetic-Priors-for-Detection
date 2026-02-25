DANA / Synthesis Reproducibility Package

This folder collects the synthesis (CAC & DAGM) and detection-related artifacts for reproducibility alongside the PAMI submission.

Structure:
- synthesis/CAC: nucleus + multi-channel spot synthesis (DataSynthesizer snapshot)
- synthesis/DAGM: scratch defect synthesizer and blending code
- detection: placeholder for detection architectures and training scripts (YOLO-based)
- docs: technical reports and html supplement
- assets: generated demo images used in the HTML document

Quick start (recommended):

1) Install dependencies (see requirements.txt)

2) Generate DAGM demo images (uses original code in src/generation):

```bash
python3 /home/AI_CAC/src/generation/generate_html_assets.py
```

3) Generate CAC samples (example using original DataSynthesizer):

```bash
python3 -c "from src.generation.DataSynthesizer import DataSynthesizer; s=DataSynthesizer(); s.generate_full_sample('CAC', 'demo01', '/tmp')"
```

Notes:
- This package intentionally references original source files in `src/generation/` to avoid duplication; snapshots are provided for archival. Prefer running the original scripts to ensure latest updates are used.
- For detection training we use existing training scripts in `src/` (e.g., `finetune_yolo_synthetic.py`). See `detection/README_detect.md` for suggested commands.
