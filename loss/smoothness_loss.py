import torch.nn as nn


class SmoothnessLoss(nn.Module):
    def __init__(self, penalty="l2"):
        super(SmoothnessLoss, self).__init__()
        assert penalty in ["l1", "l2"], "Penalty must be 'l1' or 'l2'"
        self.penalty = penalty

    def forward(self, flow):
        """
        flow: tensor of shape [B, D, H, W, 3]
              displacement field in 3D
        """
        dx = flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :]
        dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        dz = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]

        if self.penalty == "l1":
            diffs = [dx.abs(), dy.abs(), dz.abs()]
        else:  # L2
            diffs = [dx.pow(2), dy.pow(2), dz.pow(2)]

        loss = sum([d.mean() for d in diffs]) / len(diffs)
        return loss

    def __call__(self, flow):
        return self.forward(flow)
