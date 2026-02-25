#!/usr/bin/env python3
"""Generate example synthetic samples for CAC and DAGM and save into the reproducibility package.

Saves into:
- repro_package_pami/assets/CAC/<class>/*.png (+ params JSON)
- repro_package_pami/assets/DAGM/Class<id>/*.png

Uses canonical synth code under /home/AI_CAC/src/generation and config in /home/AI_CAC/configs
"""
import os
import sys
import shutil
import random
from pathlib import Path

# ensure repo src/generation is importable
sys.path.append('/home/AI_CAC/src/generation')
from DataSynthesizer import DataSynthesizer
from generate_scratch_defects import ScratchSynthesizer
import cv2
import numpy as np
import json

ROOT = Path('/home/AI_CAC/repro_package_pami')
ASSETS = ROOT / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)

# CAC generation
CAC_CLASSES = ['Normal','Deletion','Gain','CAC']
CAC_OUT = ASSETS / 'CAC'
CAC_OUT.mkdir(exist_ok=True)
for c in CAC_CLASSES:
    (CAC_OUT / c).mkdir(parents=True, exist_ok=True)

# DAGM generation (classes 1..10)
DAGM_OUT = ASSETS / 'DAGM'
DAGM_OUT.mkdir(exist_ok=True)
for i in range(1,11):
    (DAGM_OUT / f'Class{i}').mkdir(parents=True, exist_ok=True)

# instantiate synthesizers
synth = DataSynthesizer(config_path='/home/AI_CAC/configs/synthesis_config_final.json')
scratch = ScratchSynthesizer(canvas_size=(512,512))

# DAGM dataset root for backgrounds
DAGM_ROOT = Path('/home/AI_CAC/OtherDataset4PAMI/DAGM_KaggleUpload')

# Utility: find a normal background for class
import glob

def load_random_bg_for_class(class_id):
    class_path = DAGM_ROOT / f'Class{class_id}' / 'Train'
    imgs = list(class_path.glob('*.PNG'))
    imgs = [p for p in imgs if '_label' not in p.name]
    if not imgs:
        return None
    return str(random.choice(imgs))

# simple blend functions
def blend_dark(img, mask, opacity=0.8):
    img_f = img.astype(np.float32)
    shift = 200.0 * opacity
    out = img_f - mask * shift
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def blend_ridge(img, mask, opacity=0.8):
    img_f = img.astype(np.float32)
    shift = int(3)
    h,w = mask.shape
    M = np.float32([[1,0,shift],[0,1,shift]])
    offset = cv2.warpAffine(mask, M, (w,h))
    max_shift = 200.0 * opacity
    out = img_f - mask * max_shift
    out = out + offset * (max_shift * 0.8)
    out = np.clip(out,0,255).astype(np.uint8)
    return out

# Generate CAC samples: 8 per class
N_CAC = 8
print('Generating CAC samples...')
for cls in CAC_CLASSES:
    out_dir = CAC_OUT / cls
    for i in range(N_CAC):
        sid = f'{cls}_demo_{i:03d}'
        sample_dir = out_dir / sid
        sample_dir.mkdir(exist_ok=True)
        # generate_full_sample writes files; use temporary dir then copy selected files
        tmp = '/tmp/cac_demo_tmp'
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        try:
            synth.generate_full_sample(cls, sid, tmp)
        except Exception as e:
            print('Error generating sample', cls, e)
            continue
        # move generated images into sample_dir
        for f in Path(tmp).glob(f'{sid}_*.png'):
            shutil.move(str(f), str(sample_dir / f.name))
        # move params if present
        p = Path(tmp) / f'{sid}_params.json'
        if p.exists():
            shutil.move(str(p), str(sample_dir / p.name))
        shutil.rmtree(tmp, ignore_errors=True)
print('CAC generation done')

# Generate DAGM fused samples: 6 per class
N_DAGM = 6
print('Generating DAGM fused samples...')
for cls in range(1,11):
    out_dir = DAGM_OUT / f'Class{cls}'
    for i in range(N_DAGM):
        bg_path = load_random_bg_for_class(cls)
        if bg_path is None:
            print('No background for class', cls)
            continue
        bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
        if bg is None:
            continue
        bg = cv2.resize(bg, (512,512))
        # choose parameters
        if cls == 8:
            mode = 'ridge'
            width = random.uniform(2.0,3.5)
            wav = random.uniform(0.0,0.6)
            op = 0.8
        elif cls in [2,4,6,10]:
            mode='dark'
            if cls==6:
                width = random.uniform(5.0,9.0); wav=0.0; op=1.0
            elif cls==4:
                width = random.uniform(2.0,3.5); wav=random.uniform(0.0,0.5); op=0.4
            elif cls==2:
                width = random.uniform(2.0,4.0); wav=random.uniform(0.0,1.0); op=0.9
            else:
                width = random.uniform(2.5,4.5); wav=random.uniform(0.0,1.0); op=0.65
        else:
            mode='dark'
            width = random.uniform(2.0,4.0); wav=random.uniform(0.0,1.0); op=random.uniform(0.4,0.9)
        # pick a random start/end
        sx = random.randint(100,400); sy=random.randint(100,400)
        ang = random.uniform(0,2*np.pi); length = random.uniform(80,240)
        ex = sx + np.cos(ang)*length; ey = sy + np.sin(ang)*length
        mask = scratch.generate_scratch(start_pos=(sx,sy), end_pos=(ex,ey), width=width, waviness=wav, intensity=1.0, roughness=0.6)
        # mask values in [0,1]
        mask = np.clip(mask, 0.0, 1.0)
        if mode=='ridge':
            out = blend_ridge(bg, mask, op)
        else:
            out = blend_dark(bg, mask, op)
        fname = out_dir / f'class{cls}_synth_{i:03d}.png'
        cv2.imwrite(str(fname), out)
print('DAGM generation done')

print('All examples generated into', ASSETS)
