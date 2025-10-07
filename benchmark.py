"""Benchmark memory savings"""
import torch
import psutil
import gc
from gpu_poor import make_it_work
import time

def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process().memory_info().rss / 1024 / 1024

print("\n" + "="*60)
print(" "*15 + "GPU-POOR BENCHMARK")
print("="*60)

# Test with different tensor sizes
sizes = [
    (1000, 1000, "Small Model"),
    (2000, 2000, "Medium Model"),
    (3000, 3000, "Large Model")
]

results = []

for size1, size2, name in sizes:
    print(f"\nTesting {name} ({size1}x{size2})...")
    
    # Create a fake model
    class FakeModel:
        def __init__(self):
            self.weight1 = torch.randn(size1, size2, dtype=torch.float32)
            self.weight2 = torch.randn(size1, size2, dtype=torch.float32)
        
        def half(self):
            self.weight1 = self.weight1.half()
            self.weight2 = self.weight2.half()
            return self
        
        def to(self, device):
            return self
    
    # Measure before
    gc.collect()
    time.sleep(0.5)
    mem_before = get_memory_usage()
    model = FakeModel()
    mem_with_model = get_memory_usage()
    
    # Optimize
    model = make_it_work(model, verbose=False)
    gc.collect()
    time.sleep(0.5)
    mem_after = get_memory_usage()
    
    # Calculate savings
    model_size = mem_with_model - mem_before
    optimized_size = mem_after - mem_before
    savings = (1 - optimized_size/model_size) * 100 if model_size > 0 else 0
    
    results.append({
        'name': name,
        'original': f"{model_size:.1f} MB",
        'optimized': f"{optimized_size:.1f} MB",
        'savings': f"{savings:.1f}%"
    })
    
    print(f"  Original size: {model_size:.1f} MB")
    print(f"  Optimized size: {optimized_size:.1f} MB")
    print(f"  Memory saved: {savings:.1f}%")

# Print summary table
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<15} {'Original':<12} {'Optimized':<12} {'Savings':<10}")
print("-"*60)
for r in results:
    print(f"{r['name']:<15} {r['original']:<12} {r['optimized']:<12} {r['savings']:<10}")
print("="*60)
print("\nGPU-Poor makes AI accessible to everyone!")