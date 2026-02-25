import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import tempfile

# Use the canonical synthesizer from src/generation
import sys
sys.path.append('/home/AI_CAC/src/generation')
from DataSynthesizer import DataSynthesizer
from model import DANA_Net

class SynthPatchDataset(Dataset):
    """Generates small DAPI patches and derives a binary presence label from the nucleus mask."""
    def __init__(self, n_samples=50, patch_size=128):
        self.synth = DataSynthesizer(config_path='synthesis_config_final.json')
        self.n = n_samples
        self.patch_size = patch_size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        # generate one full sample and take DAPI channel
        img_dapi, mask, meta = self.synth.generate_nucleus(return_meta=True)
        # resize to patch_size
        img = cv2.resize(img_dapi, (self.patch_size, self.patch_size))
        m = cv2.resize(mask, (self.patch_size, self.patch_size))
        img = img.astype(np.float32)/255.0
        m = (m>127).astype(np.float32)
        # label: is there a nucleus at center tile (simple presence flag)
        center = self.patch_size//2
        label = 1.0 if m[center, center] > 0 else 0.0
        x = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return x, y

def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        out = model(xb)
        cls_logits = out['cls_logits']
        # pool logits to a single score (global presence) for this smoke test
        pooled = cls_logits.mean(dim=[2,3]).squeeze(1) # (B)
        loss = nn.BCEWithLogitsLoss()(pooled, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    return total/len(loader)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DANA_Net(in_ch=1).to(device)
    ds = SynthPatchDataset(n_samples=20, patch_size=128)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print('Starting quick smoke training...')
    for epoch in range(1):
        loss = train_one_epoch(model, dl, opt, device)
        print(f'Epoch {epoch} loss {loss:.4f}')
    # save a tiny checkpoint in package
    out_p = '/home/AI_CAC/repro_package_pami/detection/dana_net/dana_smoke.pth'
    torch.save(model.state_dict(), out_p)
    print('Saved model to', out_p)

if __name__ == '__main__':
    main()
