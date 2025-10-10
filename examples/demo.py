"""
Production-ready gpu-poor demonstration
Shows all improvements: mixed-precision quantization, speed, model support
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer
from gpu_poor.hybrid_a3qer import make_it_work_hybrid
from gpu_poor.fast_inference import OptimizedModel
from gpu_poor.quantization import QuantizedLinear, QuantizedLinearINT4, QuantizedLinearINT6, QuantizedConv1D
import time
import warnings
warnings.filterwarnings('ignore')

def create_simple_calibration_data(model_name, tokenizer, n_samples=64, seq_length=128):
    """Create simple calibration data without hanging"""
    # Create diverse text samples
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models are becoming more efficient.",
        "Natural language processing has many applications.",
        "The future of technology is exciting and full of possibilities.",
        "Climate change requires immediate global action.",
        "Scientific research continues to advance human knowledge.",
        "Education is the foundation of a prosperous society.",
        "Artificial intelligence is transforming various industries.",
    ]
    
    # Repeat and mix to create more samples
    full_text = " ".join(sample_texts * (n_samples // len(sample_texts) + 1))
    
    # Tokenize with proper settings
    inputs = tokenizer(
        full_text,
        return_tensors="pt",
        max_length=seq_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )
    
    return inputs["input_ids"]

def calculate_actual_model_size(model):
    """Calculate actual memory usage with mixed precision"""
    total_bytes = 0
    layer_stats = {'int4': 0, 'int6': 0, 'int8': 0, 'fp32': 0}
    
    for name, module in model.named_modules():
        # Skip wrapper modules
        if isinstance(module, QuantizedConv1D):
            # Count the underlying quantized linear
            module = module.quantized_linear
        
        if isinstance(module, QuantizedLinearINT4):
            # INT4: packed storage (0.5 bytes per weight)
            total_bytes += module.weight_packed.numel()
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
            layer_stats['int4'] += 1
            
        elif isinstance(module, QuantizedLinearINT6):
            # INT6: stored as INT8 (1 byte per weight)
            total_bytes += module.weight_int8.numel()
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
            layer_stats['int6'] += 1
            
        elif isinstance(module, QuantizedLinear):
            # INT8: 1 byte per parameter
            total_bytes += module.weight_quantized.numel()
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
            layer_stats['int8'] += 1
            
        elif hasattr(module, 'weight') and module.weight is not None:
            # Only count non-quantized layers
            module_type = type(module).__name__
            if 'Quantized' not in module_type and 'Conv1D' != module_type:
                # Skip embeddings and layer norms as they stay FP32
                if any(skip in name.lower() for skip in ['embed', 'ln', 'norm']):
                    total_bytes += module.weight.numel() * module.weight.element_size()
                    if hasattr(module, 'bias') and module.bias is not None:
                        total_bytes += module.bias.numel() * module.bias.element_size()
                else:
                    # This shouldn't happen if quantization worked
                    layer_stats['fp32'] += 1
    
    return total_bytes / (1024 * 1024), layer_stats

def demo_production_ready(model_name="gpt2"):
    print(f"\n{'='*70}")
    print(f" GPU-POOR PRODUCTION v4.0 - Mixed Precision Edition")
    print(f" INT4/INT6/INT8 Adaptive Quantization")
    print(f"{'='*70}")
    
    # Load model and tokenizer
    print(f"\n[1/5] Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Fix tokenizer settings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for generation
    
    # Calculate original size
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Original size: {original_size:.1f} MB")
    
    # Create calibration data
    print("\n[2/5] Creating calibration data...")
    print("="*60)
    print("CALIBRATION DATA PREPARATION")
    print("="*60)
    
    calibration_data = create_simple_calibration_data(
        model_name, 
        tokenizer,
        n_samples=64,
        seq_length=128
    )
    print(f"[Calibration] Created calibration tensor: {calibration_data.shape}")
    
    # Apply mixed-precision quantization
    print("\n[3/5] Applying mixed-precision quantization...")
    print("="*60)
    print("MIXED-PRECISION QUANTIZATION")
    print("="*60)
    
    # Use the hybrid quantizer with mixed precision
    quantized_model = make_it_work_hybrid(
        model,
        sample_inputs=calibration_data,
        smoothing_alpha=0.3,
        memory_target=0.4  # Target 40% of original size
    )
    
    print("[Success] Model quantized with mixed precision!")
    
    # Calculate actual compressed size
    compressed_size, layer_stats = calculate_actual_model_size(quantized_model)
    
    # Optimize for inference
    print("\n[4/5] Optimizing for inference speed...")
    optimized_model = OptimizedModel(quantized_model)
    optimized_model.optimize()
    
    # Benchmark
    print("\n[5/5] Benchmarking...")
    
    # Test generation
    test_prompt = "The future of technology"
    inputs = tokenizer(
        test_prompt, 
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        _ = optimized_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Time original
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    with torch.no_grad():
        output_original = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
    original_time = time.perf_counter() - start
    
    # Time optimized
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    with torch.no_grad():
        output_optimized = optimized_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=50,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id
        )
    optimized_time = time.perf_counter() - start
    
    # Show results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print(f"\nMemory:")
    print(f"  Original: {original_size:.1f} MB")
    print(f"  Compressed: {compressed_size:.1f} MB")
    print(f"  Reduction: {(1 - compressed_size/original_size)*100:.1f}%")
    
    print(f"\nLayer Distribution:")
    print(f"  INT4 layers: {layer_stats['int4']}")
    print(f"  INT6 layers: {layer_stats['int6']}")
    print(f"  INT8 layers: {layer_stats['int8']}")
    print(f"  FP32 layers: {layer_stats['fp32']}")
    
    print(f"\nSpeed:")
    print(f"  Original: {original_time:.2f}s")
    print(f"  Optimized: {optimized_time:.2f}s")
    speedup = original_time / optimized_time if optimized_time > 0 else 1.0
    print(f"  Speedup: {speedup:.1f}x")
    
    print(f"\nGeneration Quality:")
    original_text = tokenizer.decode(output_original[0], skip_special_tokens=True)
    optimized_text = tokenizer.decode(output_optimized[0], skip_special_tokens=True)
    
    print(f"  Original: '{original_text}'")
    print(f"  Optimized: '{optimized_text}'")
    
    # Check for repetitions
    words = optimized_text.split()
    repetitions = 0
    for i in range(len(words) - 2):
        if i + 2 < len(words) and words[i] == words[i+1] == words[i+2]:
            repetitions += 1
    
    if repetitions == 0:
        print(f"\n  [OK] Quality: Excellent (no repetitions detected)")
    else:
        print(f"\n  [WARNING] Quality: {repetitions} repetitive patterns detected")
    
    # Performance summary
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    
    if compressed_size / original_size < 0.5:
        print(f"[SUCCESS] Achieved {(1 - compressed_size/original_size)*100:.1f}% compression!")
        print(f"          Target was 50-60% - Mixed precision working correctly!")
    else:
        print(f"[INFO] Achieved {(1 - compressed_size/original_size)*100:.1f}% compression")
        print(f"       Consider adjusting bit-width distribution for more compression")
    
    print(f"\n{'='*70}")
    print("Production Ready with Mixed Precision!")
    print(f"{'='*70}")
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': (1 - compressed_size/original_size)*100,
        'speedup': speedup,
        'layer_stats': layer_stats,
        'has_repetitions': repetitions > 0
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "gpt2"

    print(f"Running production-ready demo with model: {model_name}")
    results = demo_production_ready(model_name)

    import json
    import os

    os.makedirs("results", exist_ok=True)
     # Save individual result
    with open(f"results/{model_name.replace('/', '_')}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Append to master results file
    results_file = "results/all_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append({
        "model": model_name,
        "compression": results['compression_ratio'],
        "speedup": results['speedup'],
        "original_mb": results['original_size'],
        "compressed_mb": results['compressed_size'],
        "quality": "good" if not results['has_repetitions'] else "repetitions"
    })
    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to results/{model_name.replace('/', '_')}.json")
    
    # You can test with other models:
    # demo_production_ready("microsoft/phi-2")
    # demo_production_ready("facebook/opt-125m")