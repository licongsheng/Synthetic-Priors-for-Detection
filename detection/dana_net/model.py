import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightweight DANA-Net snapshot (reproducibility snapshot)
# This is a compact, well-documented implementation matching the paper description
# (Density-Aware Localization + Microscopy Feature Enhancement stages).

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch//r),
            nn.ReLU(inplace=True),
            nn.Linear(ch//r, ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        b,c,_,_ = x.shape
        w = self.pool(x).view(b,c)
        w = self.fc(w).view(b,c,1,1)
        return x * w

class DANA_Backbone(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_ch, base_ch, ks=3),
            ConvBlock(base_ch, base_ch, ks=3)
        )
        self.layer1 = nn.Sequential(ConvBlock(base_ch, base_ch*2, ks=3, stride=2), ConvBlock(base_ch*2, base_ch*2))
        self.layer2 = nn.Sequential(ConvBlock(base_ch*2, base_ch*4, ks=3, stride=2), ConvBlock(base_ch*4, base_ch*4))
        self.layer3 = nn.Sequential(ConvBlock(base_ch*4, base_ch*8, ks=3, stride=2), ConvBlock(base_ch*8, base_ch*8))
    def forward(self, x):
        c0 = self.stem(x)   # /1
        c1 = self.layer1(c0) # /2
        c2 = self.layer2(c1) # /4
        c3 = self.layer3(c2) # /8
        return [c0, c1, c2, c3]

class DensityHead(nn.Module):
    """Predicts a low-resolution density map used for spatial re-weighting."""
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_ch, in_ch//2, ks=3),
            nn.Conv2d(in_ch//2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: feature map (e.g., c2)
        return self.conv(x)

class DetectorHead(nn.Module):
    """Lightweight detection head: classification + bbox regression.
    Uses feature fusion and SE-like attention to enhance microscopy features.
    """
    def __init__(self, in_chs=[32,64,128,256], num_anchors=1):
        super().__init__()
        # project to common channel
        out_ch = 128
        self.projs = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.fuse = nn.Sequential(ConvBlock(out_ch*len(in_chs), out_ch, ks=3), SEBlock(out_ch))
        # classification and bbox
        self.cls = nn.Conv2d(out_ch, 1 * num_anchors, kernel_size=1)
        self.reg = nn.Conv2d(out_ch, 4 * num_anchors, kernel_size=1)
    def forward(self, feats):
        # feats: list of feature maps c0..c3, upsample to largest spatial size
        sizes = [f.shape[-2:] for f in feats]
        target_size = sizes[0]
        ups = []
        for i,f in enumerate(feats):
            if f.shape[-2:] != target_size:
                f_up = F.interpolate(self.projs[i](f), size=target_size, mode='bilinear', align_corners=False)
            else:
                f_up = self.projs[i](f)
            ups.append(f_up)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        cls_logits = self.cls(x)
        bbox_reg = self.reg(x)
        return cls_logits, bbox_reg

class DANA_Net(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.backbone = DANA_Backbone(in_ch=in_ch, base_ch=32)
        # density head on an intermediate scale
        self.density_head = DensityHead(in_ch=128) # expecting c2 channel dim 128
        # detector head uses multi-resolution fusion
        self.detector_head = DetectorHead(in_chs=[32,64,128,256], num_anchors=1)
    def forward(self, x):
        c0,c1,c2,c3 = self.backbone(x)
        density = self.density_head(c2)
        cls_logits, bbox_reg = self.detector_head([c0,c1,c2,c3])
        # apply density-aware re-weighting: upsample density to cls size and multiply (simple gating)
        density_up = F.interpolate(density, size=cls_logits.shape[-2:], mode='bilinear', align_corners=False)
        cls_logits = cls_logits * (1.0 + density_up)
        return {
            'density': density,       # low-res map
            'cls_logits': cls_logits, # per-pixel presence logit
            'bbox_reg': bbox_reg      # per-pixel bbox param
        }

if __name__ == '__main__':
    # quick smoke test
    m = DANA_Net(in_ch=1)
    x = torch.randn(2,1,128,128)
    out = m(x)
    print('density', out['density'].shape)
    print('cls', out['cls_logits'].shape)
    print('reg', out['bbox_reg'].shape)
