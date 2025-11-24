# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
GSPN: Generalized Spatial Propagation Network

A parallel sequence modeling framework for efficient 2D spatial processing.

Example usage:
    >>> import torch
    >>> from gspn import GSPN
    >>>
    >>> # Create a GSPN module for 96-channel features
    >>> model = GSPN(d_model=96, feat_size=56)
    >>>
    >>> # Process a batch of feature maps
    >>> x = torch.randn(2, 96, 56, 56)
    >>> out = model(x)
    >>> print(out.shape)  # torch.Size([2, 96, 56, 56])
"""

from .core import GSPN

__version__ = "0.1.0"
__all__ = ["GSPN"]
