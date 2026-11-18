import monai
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLNCC(nn.Module):
    """Image loss of MPM-Reg V2: local normalised cross-correlation (MONAI) on an image pyramid with larger windows at
    coarser levels (pooling factor, window), averaged over the levels. Runs in fp32 outside autocast."""
    def __init__(self, levels=((4, 13), (2, 11), (1, 9))):
        super().__init__()
        self.levels = levels
        self.losses = nn.ModuleList(
            monai.losses.LocalNormalizedCrossCorrelationLoss(spatial_dims=3, kernel_size=w) for _, w in levels)

    def forward(self, pred, target):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            terms = [loss(F.avg_pool3d(pred, f) if f > 1 else pred, F.avg_pool3d(target, f) if f > 1 else target)
                     for (f, _), loss in zip(self.levels, self.losses)]
            return sum(terms) / len(terms)
