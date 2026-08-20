import torch
import torch.nn as nn
from deepali.core.grid import Grid
from deepali.spatial.bspline import FreeFormDeformation
from deepali.core.bspline import evaluate_cubic_bspline


class BSpline(nn.Module):
    """Cubic B-spline interpolation to generate a dense displacement field.

    Forward:
        forward(
            control_displacements: Tensor,    # shape (1, n, 3)
            output_shape: tuple | torch.Size, # (1, Z, Y, X) or (Z, Y, X)
            stride: int | tuple,              # int or (sz, sy, sx)
            control_positions: Optional[Tensor] = None  # shape (1, n, 3)
        ) -> Tensor

    Arguments:
        - control_displacements: (1, n, 3)
            Displacement vectors at control points.
        - output_shape: (1, Z, Y, X) or (Z, Y, X)
            Desired dense field spatial size in (Z, Y, X) order. A leading
            batch dimension of 1 is accepted and ignored.
        - stride: int or (sz, sy, sx)
            Control point spacing in voxels along each axis.
        - control_positions: (1, n, 3)
            Voxel-space positions of the control points. For a regular lattice,
            these typically start at stride/2 and increment by stride along each
            axis, e.g., (8, 8, 8), (24, 8, 8), ... for stride=16.

    Returns:
        - dense_field: Tensor of shape (Z, Y, X, 3)
            Dense displacement field aligned with the requested output shape,
            with last dimension being the displacement components (dz, dy, dx).
    """

    def __init__(self, device=None):
        """Create the B-spline evaluator.

        Args:
            device: Optional torch device where to allocate and compute. If None,
                    the device is taken from the `control_displacements` at forward.
        """
        super().__init__()
        self.device = device

    def _normalize_stride(self, stride):
        if isinstance(stride, int):
            return (stride, stride, stride)
        if isinstance(stride, (tuple, list)) and len(stride) == 3:
            return tuple(int(s) for s in stride)
        raise ValueError("stride must be int or tuple/list of length 3")

    def _normalize_output_shape(self, output_shape):
        # accept (1, Z, Y, X) or (Z, Y, X)
        if isinstance(output_shape, torch.Size):
            output_shape = tuple(output_shape)
        if isinstance(output_shape, (tuple, list)):
            if len(output_shape) == 4 and output_shape[0] == 1:
                return (int(output_shape[1]), int(output_shape[2]), int(output_shape[3]))
            if len(output_shape) == 3:
                return (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
        raise ValueError("output_shape must be (1, Z, Y, X) or (Z, Y, X)")

    def forward(self, control_displacements: torch.Tensor, output_shape, stride, control_positions: torch.Tensor = None):
        """Evaluate the dense field for the given control points.

        Args:
            control_displacements: Tensor of shape (1, n, 3).
            output_shape: (1, Z, Y, X) or (Z, Y, X) specifying spatial size.
            stride: int or (sz, sy, sx) spacing between control points.
            control_positions: Tensor of shape (1, n, 3) with voxel-space
                               positions of the control points.

        Returns:
            Tensor of shape (Z, Y, X, 3): dense displacement field.
        """
        # Validate input shape
        if control_displacements.ndim != 3 or control_displacements.shape[0] != 1 or control_displacements.shape[2] != 3:
            raise ValueError("control_displacements must have shape (1, n, 3)")

        device = self.device or control_displacements.device
        (Z_out, Y_out, X_out) = self._normalize_output_shape(output_shape)
        stride_3 = self._normalize_stride(stride)

        crtl_size_unpadded = [
            Z_out // stride_3[0],
            Y_out // stride_3[1],
            X_out // stride_3[2],
        ]

        crtl_pos_unpadded = control_positions.clone()
        crtl_pos_unpadded[..., 0] = (crtl_pos_unpadded[..., 0] - stride_3[0]//2) // stride_3[0]
        crtl_pos_unpadded[..., 1] = (crtl_pos_unpadded[..., 1] - stride_3[1]//2) // stride_3[1]
        crtl_pos_unpadded[..., 2] = (crtl_pos_unpadded[..., 2] - stride_3[2]//2) // stride_3[2]

        # cast to int
        crtl_pos_unpadded = crtl_pos_unpadded.to(torch.int32)

        crtl_unpadded = torch.zeros((1, 3, crtl_size_unpadded[0], crtl_size_unpadded[1], crtl_size_unpadded[2]), device=device, dtype=control_displacements.dtype)

        cx = torch.zeros_like(crtl_unpadded[0, 0]).index_put((crtl_pos_unpadded[..., 0], crtl_pos_unpadded[..., 1], crtl_pos_unpadded[..., 2]), control_displacements[..., 0])
        cy = torch.zeros_like(crtl_unpadded[0, 1]).index_put((crtl_pos_unpadded[..., 0], crtl_pos_unpadded[..., 1], crtl_pos_unpadded[..., 2]), control_displacements[..., 1])
        cz = torch.zeros_like(crtl_unpadded[0, 2]).index_put((crtl_pos_unpadded[..., 0], crtl_pos_unpadded[..., 1], crtl_pos_unpadded[..., 2]), control_displacements[..., 2])
        crtl_unpadded = torch.stack([cx, cy, cz], dim=0).unsqueeze(0)

        ctrl = nn.functional.pad(crtl_unpadded, (1, 1, 1, 1, 1, 1), mode='reflect')

        n_given = control_displacements.shape[1]

        Z_out_padded = Z_out + stride_3[0]
        Y_out_padded = Y_out + stride_3[1]
        X_out_padded = X_out + stride_3[2]

        ctrl = nn.functional.pad(ctrl, (0, 2, 0, 2, 0, 2), mode='replicate')
        Z_out_extend = Z_out_padded + 2*stride_3[0]
        Y_out_extend = Y_out_padded + 2*stride_3[1]
        X_out_extend = X_out_padded + 2*stride_3[2]

        # Prepare grid and kernel
        grid = Grid(size=(Z_out_extend, Y_out_extend, X_out_extend), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), device=device)
        ffd = FreeFormDeformation(grid, stride=stride_3, transpose=False, params=True).to(device)

        # Evaluate dense field: returns (1, 3, Z, Y, X)
        dense = evaluate_cubic_bspline(ctrl, stride=stride_3, shape=(Z_out_extend, Y_out_extend, X_out_extend), kernel=ffd.kernel())
        offset_z = stride_3[0] // 2
        offset_y = stride_3[1] // 2
        offset_x = stride_3[2] // 2
        dense = dense[..., offset_z:-offset_z, offset_y:-offset_y, offset_x:-offset_x]

        # To (Z, Y, X, 3)
        dense = dense.permute(0, 2, 3, 4, 1).squeeze(0)
        return dense
