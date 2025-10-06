"""Basic tests for gpu-poor"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from gpu_poor import make_it_work, GPUPoor

def test_import():
    """Test that package imports correctly"""
    assert make_it_work is not None
    assert GPUPoor is not None
    print("[PASS] Import test passed")

def test_simple_tensor():
    """Test optimization on a simple tensor"""
    tensor = torch.randn(100, 100)
    optimized = make_it_work(tensor, verbose=False)
    assert optimized is not None
    print("[PASS] Tensor test passed")

if __name__ == "__main__":
    test_import()
    test_simple_tensor()
    print("\n[SUCCESS] All tests passed!")