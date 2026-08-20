# --------------------------------------------------------
# Based on BEiT, timm, DINO, DeiT and MAE-pytorch code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/pengzhiliang/MAE-pytorch
# --------------------------------------------------------'

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from model.blocks import Block, PatchEmbed
from model.model_utils import PositionalEncoding_posEmb_nd
from model.bspline import BSpline
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, pos=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.img_size = None

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed = None
        self.pe = PositionalEncoding_posEmb_nd(embed_dim,)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, x_augm, pe_enc_x, mask_target, mask_source, positions=None):
        self.pos_embed = pe_enc_x

        B, D, H, W = x.shape
        x = self.patch_embed(x)
        x_augm = self.patch_embed(x_augm)

        B, N, C = x.shape

        x_pos = x[mask_target].reshape(B, -1, C)
        x_noPos = x_augm[mask_source].reshape(B, -1, C)

        pos = self.pos_embed[mask_target].reshape(B, -1, C)

        x = torch.concat([x_pos + pos, self.mask_token + x_noPos], dim=1)  # image tokens with real positional embedding and image tokens with mask tokens

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward(self, x, x_augm, pe_enc_x, mask_target, mask_source, positions=None):
        B, D, H, W = x.shape
        self.img_size = [D, H, W]
        x = self.forward_features(x, x_augm, pe_enc_x, mask_target, mask_source, positions)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))  # [B, N, 3*16^2]

        return x


class PretrainVisionTransformer(nn.Module):
    """ Masked Position Modeling pre-training model (stage 1).

    Patches keep their content; a subset of positional embeddings is masked
    (replaced by a mask token in the encoder, and by perturbed spatial cues in
    the decoder). The decoder reconstructs the position of each masked patch.
    """
    def __init__(self,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 pos=None,
                 training_scheme=None,
                 give_spatial_cues=True,
                 ):

        if training_scheme == "single" or training_scheme == 'single_random' or training_scheme == 'single_ImgNet':
            pass
        elif training_scheme == "synth":
            pass
        elif training_scheme == "unpaired":
            pass
        else:
            assert False, "Unknown training scheme"

        self.training_scheme = training_scheme
        self.img_size = None
        self.give_spatial_cues = give_spatial_cues

        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        # this mask token is the decoder mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, x_augm, pe_enc_x, pe_dec_x, pe_dec_augm, mask_target, mask_source, positions=None, regularization_positions=None):
        self.pos_embed = pe_dec_x

        x_orig = x.clone()
        B, D, H, W = x.shape
        self.img_size = [D, H, W]

        x_vis = self.encoder(x, x_augm, pe_enc_x, mask_target, mask_source, positions)  # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, N, C = x_vis.shape

        pos_emd_vis = self.pos_embed[mask_target].reshape(B, -1, C)

        if self.give_spatial_cues:
            pos_embed_masked = pe_dec_augm[mask_source].reshape(B, -1, C)
            n_masked = pos_embed_masked.shape[1]
        else:
            n_masked = mask_source.sum(dim=1)[0].int().item()
            pos_embed_masked = self.mask_token.expand(B, n_masked, -1)

        pos_embed = torch.cat([pos_emd_vis, pos_embed_masked], dim=1)

        x_full = x_vis + pos_embed
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, n_masked)  # [B, N_mask, 3 * 16 * 16]

        # scale the predicted positions to the original image size
        x[..., 0] = x[..., 0] * self.img_size[0]
        x[..., 1] = x[..., 1] * self.img_size[1]
        x[..., 2] = x[..., 2] * self.img_size[2]

        return x


class PretrainVisionTransformer_img2img_torchInterpolation_bSpline(nn.Module):
    """ Registration model (stages 2 and 3): predicts per-patch positions,
    converts them to a dense displacement field via cubic B-spline interpolation and
    warps the first input image towards the second.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 pos=None,
                 encoding="FT",
                 training_scheme=None,
                 interpolation="bilinear",
                 give_spatial_cues=True,
                 ):

        if training_scheme == "synth":
            pass
        elif training_scheme == "img2img":
            pass
        elif training_scheme == "paired":
            pass
        elif training_scheme == "unpaired":
            pass
        elif training_scheme == "vxm":
            pass
        else:
            assert False, "Unknown training scheme - single or single_random not allowed"

        self.training_scheme = training_scheme
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.give_spatial_cues = give_spatial_cues

        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            )

        self.img_size = img_size

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.bspline = BSpline()

        self.pos_embed = None
        self.encoding = encoding
        self.pe = PositionalEncoding_posEmb_nd(decoder_embed_dim, )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, x_augm, pe_enc_x, pe_dec_x, pe_dec_augm, mask_target, mask_source, positions, regularization_positions):
        self.pos_embed = pe_dec_x

        x_orig = x.clone().requires_grad_(True)
        B, D, H, W = x.shape
        self.img_size = [D, H, W]

        x_vis = self.encoder(x, x_augm, pe_enc_x, mask_target, mask_source, positions)  # [B, N_vis, C_e]

        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]

        B, N, C = x_vis.shape

        # x_vis is already x_full
        pos_emd_vis = self.pos_embed[mask_target].reshape(B, -1, C)

        if self.give_spatial_cues:
            pos_embed_masked = pe_dec_augm[mask_source].reshape(B, -1, C)
            n_masked = pos_embed_masked.shape[1]
        else:
            # mask_source is a torch tensor while training but a numpy array in the
            # validation/inference engines, so avoid the torch-only `dim=` kwarg.
            n_masked = int(mask_source[0].sum())
            pos_embed_masked = self.mask_token.expand(B, n_masked, -1)

        pos_embed = torch.cat([pos_emd_vis, pos_embed_masked], dim=1)
        x_full = x_vis + pos_embed
        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, n_masked)  # [B, N_mask, 3 * 16 * 16]

        # scale the predicted positions to the original image size
        x[..., 0] = x[..., 0] * self.img_size[0]
        x[..., 1] = x[..., 1] * self.img_size[1]
        x[..., 2] = x[..., 2] * self.img_size[2]

        # bc we removed inplace operation in forward path
        x = x.unsqueeze(1)
        x_augm = x_augm.unsqueeze(1)

        b, c, d, h, w = x_augm.shape
        n_dims = len(x_augm.shape) - 2
        flow = torch.zeros((b, d, h, w, n_dims)).to(x.device)

        x = torch.flip(x, dims=[-1])
        regularization_positions = torch.flip(regularization_positions, dims=[-1])

        x = x.squeeze(0)

        flow = regularization_positions - x

        # BSpline expects components and positions in (z, y, x) order
        flow_for_bspline = flow[..., [2, 1, 0]]
        control_pos_zyx = regularization_positions[..., [2, 1, 0]]

        flow = self.bspline(flow_for_bspline, x_orig.shape, tuple(self.patch_size), control_pos_zyx)  # shape torch.Size([160, 224, 192, 3])
        # BSpline returns components ordered as (dz, dy, dx). Reorder to (dx, dy, dz)
        # to match grid_sample's expected (x, y, z) convention used below.
        flow = flow[..., [2, 1, 0]]
        flow = flow.unsqueeze(0)

        regularization_positions = regularization_positions.reshape(B, d//self.patch_size[0], h//self.patch_size[1], w//self.patch_size[2], n_dims)

        x = x.reshape(B, d//self.patch_size[0], h//self.patch_size[1], w//self.patch_size[2], n_dims)

        flow_px = flow.clone()
        # scale the flow from original image size to [-1, 1]
        flow[..., 0] = flow[..., 0] / (w/2)
        flow[..., 1] = flow[..., 1] / (h/2)
        flow[..., 2] = flow[..., 2] / (d/2)

        # create a grid thats size [h, w] and ranges from -1 to 1
        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0).repeat(b, 1, 1), x_orig.unsqueeze(1).size(), align_corners=True).to(x.device)

        grid = grid - flow

        x_orig = x_orig.unsqueeze(1)  # add channel dim

        x_augm = F.grid_sample(x_orig, grid, align_corners=True, padding_mode='border')

        x = torch.flip(x, dims=[-1])

        x = x.reshape(B, -1, n_dims)

        return x, x_augm, flow, flow_px


def mpm_reg_pretrain_3D(patch_size, projection_dim, channels, output_parameters, training_scheme, pretrained=False, encoder_num_heads=12, encoder_depth=12, decoder_num_heads=6, decoder_depth=4, give_spatial_cues=True, **kwargs):
    """MPM-Reg stage-1 model (single-image masked position modeling pre-training)."""
    encoder_embed_dim = patch_size[0]*patch_size[1]*patch_size[2] if projection_dim is None else projection_dim
    print(f"Creating model with patch_size={patch_size}, channels={channels}, d_hidd={encoder_embed_dim}",)

    model = PretrainVisionTransformer(
        patch_size=patch_size,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_num_classes=0,
        encoder_in_chans=channels,
        decoder_num_classes=output_parameters,
        decoder_embed_dim=512,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        training_scheme=training_scheme,
        give_spatial_cues=give_spatial_cues,
        **kwargs)

    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


def mpm_reg_registration_3D_bspline(patch_size, projection_dim, channels, output_parameters, training_scheme, pretrained=False, encoder_num_heads=12, decoder_num_heads=6, encoder_depth=12, decoder_depth=4, **kwargs):
    """MPM-Reg stage-2/3 model (unpaired registration with B-spline interpolation)."""
    encoder_embed_dim = patch_size[0]*patch_size[1]*patch_size[2] if projection_dim is None else projection_dim
    print(f"Creating model with patch_size={patch_size}, channels={channels}, d_hidd={encoder_embed_dim}",)

    model = PretrainVisionTransformer_img2img_torchInterpolation_bSpline(
        patch_size=patch_size,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        encoder_num_classes=0,
        encoder_in_chans=channels,
        decoder_num_classes=output_parameters,
        decoder_embed_dim=512,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        encoding="posEmb",
        training_scheme=training_scheme,
        **kwargs)

    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
