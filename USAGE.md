# GSPN Package Usage

This package provides a simplified installation containing only the core GSPN module for easy integration into your PyTorch projects.

## Installation

```bash
# Install from local directory
pip install -e .

# Or from GitHub (after pushing)
pip install git+https://github.com/whj363636/GSPN.git
```

## Quick Start

```python
import torch
from gspn import GSPN

# Create a GSPN module
gspn_layer = GSPN(d_model=96, feat_size=56)

# Process a batch of feature maps (B, C, H, W)
x = torch.randn(2, 96, 56, 56)
output = gspn_layer(x)

print(output.shape)  # torch.Size([2, 96, 56, 56])
```

## API Reference

### GSPN Module

```python
GSPN(
    d_model=96,           # Number of input/output channels
    feat_size=56,         # Spatial size (height or width)
    items_each_chunk=8,   # Chunk size for processing
    ssm_ratio=2.0,        # Expansion ratio for inner dimension
    ssm_d_state=16,       # State dimension
    d_conv=3,             # Depthwise conv kernel size
    conv_bias=True,       # Use bias in convolution
    dropout=0.0,          # Dropout rate
    bias=False,           # Use bias in linear layers
    n_directions=4,       # Number of scanning directions
    act_layer=nn.SiLU,    # Activation function
    channel_first=True,   # Input format (B,C,H,W)
)
```

### Parameters

- **d_model** (int): Number of input and output channels. This should match your feature dimension.
- **feat_size** (int): The spatial size of your feature map (assumes square, i.e., H=W=feat_size).
- **items_each_chunk** (int): Controls the chunk size for the spatial propagation. Larger values may be faster but use more memory.
- **ssm_ratio** (float): Expansion ratio for the inner dimension. Higher values give more capacity.
- **dropout** (float): Dropout rate applied to the output.

### Input/Output

- **Input**: Tensor of shape `(B, C, H, W)` where:
  - B = batch size
  - C = channels (should equal d_model)
  - H, W = height and width (should be close to feat_size)

- **Output**: Tensor of shape `(B, C, H, W)` (same as input)

## Integration Example

Use GSPN as a drop-in replacement for attention or as a spatial processing layer:

```python
import torch.nn as nn
from gspn import GSPN

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.gspn = GSPN(d_model=96, feat_size=224)
        self.conv2 = nn.Conv2d(96, 64, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gspn(x)  # Spatial propagation
        x = self.conv2(x)
        return x

model = MyModel()
```

## CUDA Requirements

The GSPN module uses CUDA-optimized operations for efficiency. Make sure you have:
- CUDA-capable GPU
- PyTorch with CUDA support installed
- CUDA toolkit for compilation during installation

If CUDA is not available during installation, the package will still install but may have limited functionality.

## Examples

See `example_usage.py` for more detailed examples.

## Citation

If you use GSPN in your research, please cite:

```bibtex
@inproceedings{wang2025parallel,
    author    = {Wang, Hongjun and Byeon, Wonmin and Xu, Jiarui and Gu, Jinwei and Cheung, Ka Chun and Wang, Xiaolong and Han, Kai and Kautz, Jan and Liu, Sifei},
    title     = {Parallel Sequence Modeling via Generalized Spatial Propagation Network},
    journal   = {CVPR},
    year      = {2025}
}
```

## License

NVIDIA NC License - See LICENSE file for details.
