"""Dense head of MPM-Reg V2: turns the decoder tokens into a dense displacement field (all outputs in voxels,
components in (D, H, W) order)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def spline_kernel(stride):
    """Separable 3D kernel (stride^3, 1, 4, 4, 4): the cubic B-spline as a convolution over the control lattice;
    output voxel r of a cell depends on 4 consecutive control points per axis."""
    t = torch.arange(stride, dtype=torch.float64) / stride
    w = torch.stack([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4, -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3], 1) / 6
    return torch.einsum('ai,bj,ck->abcijk', w, w, w).reshape(stride ** 3, 1, 4, 4, 4).float()


class DenseHead(nn.Module):
    """Control displacements + tokens -> dense field as Conv3d(kernel 4) + pixel shuffle, the way deepali evaluates a
    cubic B-spline (model/bspline.py), but with learnable weights: every voxel of a patch has its own weights over the
    4^3 neighbouring tokens.

    The output is the fixed B-spline of the control displacements plus a learned convolution (`delta`) over the
    control displacements and `n_extra` projected token channels. `delta` starts at zero, so an untrained head gives
    exactly the B-spline field of the stage-2 model.
    """

    def __init__(self, stride, embed_dim, n_extra=13):
        super().__init__()
        self.stride = stride
        self.register_buffer('spline', spline_kernel(stride), persistent=False)
        self.extra_proj = nn.Linear(embed_dim, n_extra)
        nn.init.normal_(self.extra_proj.weight, std=0.02)
        nn.init.zeros_(self.extra_proj.bias)
        self.delta = nn.Parameter(torch.zeros(3 * stride ** 3, n_extra + 3, 4, 4, 4))

    def _shuffle(self, cells, grid):
        s = self.stride
        gd, gh, gw = (g + 1 for g in grid)
        x = cells.reshape(3, s, s, s, gd, gh, gw).permute(0, 4, 1, 5, 2, 6, 3).reshape(3, gd * s, gh * s, gw * s)
        o = s // 2
        return x[:, o:-o, o:-o, o:-o]

    def forward(self, ctrl, tokens, grid):
        """ctrl (3, *grid) control displacements, tokens (N, E) in raster order -> field (3, D, H, W)."""
        pad = lambda v: F.pad(v, (1, 3, 1, 3, 1, 3), mode='replicate')
        extra = self.extra_proj(tokens).T.reshape(1, -1, *grid)
        ctrl_p = pad(ctrl[None])
        cells = F.conv3d(torch.cat([ctrl_p, pad(extra)], 1), self.delta)[0]
        cells = cells + F.conv3d(ctrl_p[0][:, None], self.spline).reshape_as(cells)
        return self._shuffle(cells, grid)
