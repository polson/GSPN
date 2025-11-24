# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.gaterecurrent import GateRecurrent2dnoind


def normalize_w(Gl, Gm, Gr):
    """Normalize weights for GSPN"""
    Gl_s = torch.sigmoid(Gl)
    Gm_s = torch.sigmoid(Gm)
    Gr_s = torch.sigmoid(Gr)

    sum_s = Gl_s + Gm_s + Gr_s

    sum_s[:, :, 0, :] = Gm_s[:, :, 0, :] + Gr_s[:, :, 0, :]
    sum_s[:, :, -1, :] = Gl_s[:, :, -1, :] + Gm_s[:, :, -1, :]

    sum_s = sum_s.clamp(min=1e-7)

    return Gl_s / sum_s, Gm_s / sum_s, Gr_s / sum_s


class Linear2d(nn.Linear):
    """Linear layer that works with channel-first tensors"""
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for channel-first tensors"""
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GRN(nn.Module):
    """Global Response Normalization layer"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class GSPN(nn.Module):
    """
    Generalized Spatial Propagation Network (GSPN) Module

    A parallel sequence modeling layer that processes 2D spatial data efficiently.

    Args:
        d_model (int): Number of input channels
        feat_size (int): Spatial size of the feature map (height or width)
        items_each_chunk (int): Chunk size for processing. Default: 8
        ssm_ratio (float): Expansion ratio for inner dimension. Default: 2.0
        ssm_d_state (int): State dimension. Default: 16
        d_conv (int): Depthwise convolution kernel size. Default: 3
        conv_bias (bool): Whether to use bias in convolution. Default: True
        dropout (float): Dropout rate. Default: 0.0
        bias (bool): Whether to use bias in linear layers. Default: False
        n_directions (int): Number of scanning directions. Default: 4
        act_layer: Activation function. Default: nn.SiLU
        channel_first (bool): Whether input is channel-first (B,C,H,W). Default: True

    Input shape: (B, C, H, W) if channel_first=True
    Output shape: (B, C, H, W) if channel_first=True

    Example:
        >>> gspn = GSPN(d_model=96, feat_size=56)
        >>> x = torch.randn(2, 96, 56, 56)
        >>> out = gspn(x)
        >>> out.shape
        torch.Size([2, 96, 56, 56])
    """
    def __init__(
        self,
        d_model=96,
        feat_size=56,
        items_each_chunk=8,
        ssm_ratio=2.0,
        ssm_d_state=16,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        n_directions=4,
        act_layer=nn.SiLU,
        channel_first=True,
        **kwargs,
    ):
        super().__init__()

        self.d_state = ssm_d_state
        self.channel_first = channel_first
        self.c_group = 12
        self.n_directions = n_directions
        self.items_each_chunk = items_each_chunk

        d_inner = int(ssm_ratio * d_model)

        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        # Input projection
        self.in_proj = Linear(d_model, d_inner, bias=bias)
        self.act = act_layer()

        # Depthwise convolution
        self.d_spn = d_inner
        self.conv2d = nn.Conv2d(
            in_channels=self.d_spn,
            out_channels=self.d_spn,
            groups=self.d_spn,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )

        # SPN core module
        self.spn_core = GateRecurrent2dnoind(self.items_each_chunk)

        # Projection layers
        ks = 1
        self.x_conv_down = nn.Conv2d(self.d_spn, self.d_state, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.w_conv_up = nn.Conv2d(self.d_state, self.c_group * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.l_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.u_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.d_conv = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks-1)//2, bias=False)
        self.m_conv = nn.Conv2d(self.n_directions, 1, kernel_size=1, bias=False)

        # Output layers
        self.grn = GRN(d_inner)
        self.out_act = nn.Identity()
        self.out_norm = LayerNorm(self.d_spn)
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def spn_block(self, X, l, u, Gl, Gm, Gr, D, spn_module):
        """Apply spatial propagation block"""
        Gl, Gm, Gr = normalize_w(Gl, Gm, Gr)

        Gl = Gl.to(X.dtype)
        Gm = Gm.to(X.dtype)
        Gr = Gr.to(X.dtype)

        out = spn_module(X, l, Gl, Gm, Gr)
        if D is not None:
            out = out * u + X * D
        else:
            out = out * u
        return out

    def forward_core(self, x: torch.Tensor):
        """Core forward pass"""
        B, D, H, W = x.shape

        x_proxy = self.x_conv_down(x)
        ws = self.w_conv_up(x_proxy)
        Ls = self.l_conv_up(x_proxy).contiguous()
        Us = self.u_conv_up(x_proxy).contiguous()
        Ds = self.d_conv(x_proxy).contiguous()

        x_hwwh = torch.stack([x, x.transpose(2, 3).contiguous()], dim=1)
        xs = torch.cat([x_hwwh, x_hwwh.flip(dims=[-1]).contiguous()], dim=1)
        xs = xs.view(B, -1, H, W)
        xs = xs.contiguous()

        Gs = torch.split(ws, D*self.n_directions, dim=1)
        G3 = [g.contiguous() for g in Gs]

        out_y = self.spn_block(xs, Ls, Us, G3[0], G3[1], G3[2], Ds, self.spn_core)

        out_y = out_y.view(B, self.n_directions, D*H, W)
        out_y = self.m_conv(out_y).view(B, D, H, W)

        y = self.out_norm(out_y)

        return y

    def forward(self, x: torch.Tensor):
        """
        Forward pass of GSPN

        Args:
            x: Input tensor of shape (B, C, H, W) if channel_first=True

        Returns:
            Output tensor of same shape as input
        """
        x = self.in_proj(x)
        x = self.conv2d(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = self.grn(y)
        out = self.dropout(self.out_proj(y))
        return out
