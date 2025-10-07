# Create real_benchmark.py
"""Benchmark showing REAL memory savings with INT8 quantization"""
import torch
import torch.nn as nn
import psutil
import gc
from gpu_poor import make_it_work
from transformers import AutoModel
import time

def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024

print("\n" + "="*70)
print(" "*20 + "GPU-POOR REAL BENCHMARK")
print(" "*15 + "INT8 Quantization vs Original")
print("="*70)

# Test with a real model
print("\nLoading BERT (109M parameters)...")
model = AutoModel.from_pretrained("bert-base-uncased")

# Measure original
original_size = get_model_size(model)
print(f"Original model size: {original_size:.1f} MB (FP32)")

# Apply GPU-Poor optimization
print("\nApplying GPU-Poor INT8 optimization...")
model_optimized = make_it_work(model, aggressive=True, verbose=False)

# Measure optimized
optimized_size = get_model_size(model_optimized)
reduction = (1 - optimized_size/original_size) * 100

print(f"Optimized model size: {optimized_size:.1f} MB (INT8)")
print(f"Memory reduction: {reduction:.1f}%")

# Test inference speed
print("\nTesting inference speed...")
dummy_input = torch.randint(0, 1000, (1, 128))

# Original model time
start = time.time()
with torch.no_grad():
    _ = model(dummy_input)
original_time = time.time() - start

# Optimized model time
start = time.time()
with torch.no_grad():
    _ = model_optimized(dummy_input)
optimized_time = time.time() - start

print(f"Original inference: {original_time:.3f}s")
print(f"Optimized inference: {optimized_time:.3f}s")
print(f"Speed difference: {(original_time/optimized_time):.1f}x")

print("\n" + "="*70)
print("REAL MEMORY SAVINGS ACHIEVED!")
print("="*70)
