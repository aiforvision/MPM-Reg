import torch


class PositionalEncoding_posEmb_nd:
    def __init__(self, d_hid, half_precision=True):
        self.d_hid = d_hid
        self.d = None
        self.h = None
        self.w = None
        self.num_parameters = None
        self.half_precision = half_precision

        self.divisor_2d = torch.pow(10000, 4*torch.arange(self.d_hid//2)/self.d_hid)
        self.divisor_3d = torch.pow(10000, 8*torch.arange(self.d_hid//4)/self.d_hid)

    def get_position_angle_vec_n2(self, position):
        x, y = position
        pos_x = x / self.divisor_2d
        pos_y = y / self.divisor_2d
        return torch.concat((pos_x, pos_y))

    def get_position_angle_vec_n3(self, position):
        x, y, z = position
        pos_x = x / self.divisor_3d
        pos_y = y / self.divisor_3d
        pos_z = z / self.divisor_3d
        pos_none = torch.zeros_like(pos_x)
        return torch.concat((pos_x, pos_y, pos_z, pos_none))

    def encode_positions(self, pos, img_size):
        assert len(pos.shape) == 2, f"pos shape not recognized: {pos.shape}"
        n_patches, num_param = pos.shape
        if self.num_parameters is None:
            self.num_parameters = num_param

        # normalization to range [0, 1]
        if self.num_parameters == 2:
            self.h = img_size[0]
            self.w = img_size[1]
            pos = torch.stack([pos[..., 0]/self.h, pos[..., 1]/self.w], axis=-1)
        elif self.num_parameters == 3:
            self.d = img_size[0]
            self.h = img_size[1]
            self.w = img_size[2]
            pos = torch.stack([pos[..., 0]/self.d, pos[..., 1]/self.h, pos[..., 2]/self.w], axis=-1)
        else:
            raise ValueError("Only 2 or 3 parameters are supported")

        pe = torch.zeros((n_patches, self.d_hid))

        if self.num_parameters == 2:
            for n in range(n_patches):
                pe[n] = self.get_position_angle_vec_n2(pos[n])
                pe[n, 0::2] = torch.sin(pe[n, 0::2])  # dim 2i
                pe[n, 1::2] = torch.cos(pe[n, 1::2])  # dim 2i+1
        elif self.num_parameters == 3:
            for n in range(n_patches):
                pe[n] = self.get_position_angle_vec_n3(pos[n])
                pe[n, 0::2] = torch.sin(pe[n, 0::2])
                pe[n, 1::2] = torch.cos(pe[n, 1::2])

        self.encoded_pe = pe

        if self.half_precision:
            pe = pe.half()

        return pe
