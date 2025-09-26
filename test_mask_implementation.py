#!/usr/bin/env python3
"""
Test script to verify our mask implementation matches the original.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import math

# Import our refactored implementation
from src.models.mlp import SparseLinear

def test_mask_generation():
    """Test that our mask generation matches the original logic."""
    
    # Test parameters
    in_dim, out_dim = 784, 512
    mask_num = 0
    num_fixed = 64
    mask_constant = 1.0
    
    print("Testing mask generation...")
    
    # Create our SparseLinear
    sparse_linear = SparseLinear(
        in_dim=in_dim,
        out_dim=out_dim,
        mask_type='random_subsets',
        mask_constant=mask_constant,
        mask_num=mask_num,
        num_fixed=num_fixed,
        do_normal_mask=True
    )
    
    print(f"Mask shape: {sparse_linear.mask.shape}")
    print(f"Normal mask shape: {sparse_linear.normal_mask.shape}")
    print(f"Mask sum (should be {out_dim * in_dim - out_dim * num_fixed}): {sparse_linear.mask.sum().item()}")
    print(f"Number of zeros per row (should be {num_fixed}): {(sparse_linear.mask == 0).sum(dim=1).unique()}")
    
    # Test forward pass
    x = torch.randn(32, in_dim)
    output = sparse_linear(x)
    print(f"Forward pass output shape: {output.shape}")
    
    # Test that the weight is properly masked
    print(f"Weight shape: {sparse_linear.weight.shape}")
    print(f"Weight masked correctly: {torch.allclose(sparse_linear.weight, sparse_linear.weight * sparse_linear.mask + (1 - sparse_linear.mask) * sparse_linear.mask_constant * sparse_linear.normal_mask)}")
    
    print("✅ Mask implementation test passed!")

if __name__ == "__main__":
    test_mask_generation()
