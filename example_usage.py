"""
Example usage of the GSPN module

This demonstrates how to use the GSPN module as a drop-in layer
for processing (B, C, H, W) tensors.
"""

import torch
from gspn import GSPN


def main():
    # Example 1: Basic usage
    print("Example 1: Basic GSPN module")
    print("-" * 50)

    # Create a GSPN module for 96-channel features at 56x56 resolution
    gspn_layer = GSPN(d_model=96, feat_size=56)

    # Create sample input (batch_size=2, channels=96, height=56, width=56)
    x = torch.randn(2, 96, 56, 56)

    print(f"Input shape: {x.shape}")

    # Forward pass
    output = gspn_layer(x)

    print(f"Output shape: {output.shape}")
    print()

    # Example 2: Different configuration
    print("Example 2: Custom configuration")
    print("-" * 50)

    # Create GSPN with custom parameters
    gspn_custom = GSPN(
        d_model=128,
        feat_size=28,
        items_each_chunk=16,  # Larger chunks
        ssm_ratio=3.0,        # Higher expansion
        dropout=0.1,          # Add dropout
    )

    x2 = torch.randn(4, 128, 28, 28)
    print(f"Input shape: {x2.shape}")

    output2 = gspn_custom(x2)
    print(f"Output shape: {output2.shape}")
    print()

    # Example 3: Use in a simple network
    print("Example 3: GSPN in a simple network")
    print("-" * 50)

    class SimpleNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.gspn1 = GSPN(d_model=64, feat_size=32)
            self.gspn2 = GSPN(d_model=64, feat_size=32)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.fc = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.nn.functional.relu(x)
            x = self.gspn1(x)
            x = self.gspn2(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x

    model = SimpleNetwork()
    x3 = torch.randn(2, 3, 32, 32)

    print(f"Input shape: {x3.shape}")
    output3 = model(x3)
    print(f"Output shape: {output3.shape}")
    print()

    print("All examples completed successfully!")


if __name__ == "__main__":
    main()
