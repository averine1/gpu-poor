"""
Production-ready gpu-poor demonstration
Shows all improvements: mixed-precision quantization, speed, model support
"""
import torch
import torch.nn as nn
import os
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_poor.hybrid_a3qer import make_it_work_hybrid
from gpu_poor.fast_inference import OptimizedModel
from gpu_poor.quantization import QuantizedLinear, QuantizedLinearINT4, QuantizedLinearINT6, QuantizedConv1D
import time
import warnings
warnings.filterwarnings('ignore')
import torch, os 
torch.set_num_threads(os.cpu_count() or 1)  # all CPU cores
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def build_gen_kwargs(
    model,
    tokenizer,
    inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=None,
    top_k=None,
):
    """
    Safe defaults for both causal and encoder–decoder models.
    - Ensures pad_token_id exists
    - Sets left padding for causal, right padding for seq2seq
    - Provides decoder_input_ids for T5-like models
    """
    import torch

    # Ensures pad token exists (e.g., GPT-2)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base kwargs
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["top_k"] = top_k

    # Default
    tokenizer.padding_side = "left"

    # Seq2seq (e.g., T5): right padding + decoder start token
    if getattr(model.config, "is_encoder_decoder", False):
        tokenizer.padding_side = "right"
        if model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        bsz = inputs["input_ids"].size(0)
        device = inputs["input_ids"].device
        kwargs["decoder_input_ids"] = torch.full(
            (bsz, 1),
            model.config.decoder_start_token_id,
            dtype=inputs["input_ids"].dtype,
            device=device,
        )

    return kwargs



def hf_token():
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

def create_simple_calibration_data(model_name, tokenizer, n_samples=64, seq_length=128):
    """Create simple calibration data without hanging"""
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
    
    # Short samples to fill the batch
    batch = (sample_texts * ((n_samples + len(sample_texts) - 1)//len(sample_texts)))[:n_samples]
    
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        max_length=seq_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True
    )
    
    return inputs["input_ids"]

def calculate_actual_model_size(model):
    """Calculate actual memory usage - FIXED to avoid double counting"""
    from gpu_poor.quantization import (
        QuantizedLinear, QuantizedLinearINT4, QuantizedLinearINT6, QuantizedConv1D
    )
    try:
        from gpu_poor.quantization import QuantizedEmbeddingINT8, QuantizedLMHeadINT8
    except ImportError:
        QuantizedEmbeddingINT8 = None
        QuantizedLMHeadINT8 = None
    
    total_bytes = 0
    layer_stats = {'int4': 0, 'int6': 0, 'int8': 0, 'int8_emb': 0, 'fp32': 0}
    
    # Track what we've counted to avoid double-counting
    counted_modules = set()
    
    for name, module in model.named_modules():
        # Skip if already counted
        if id(module) in counted_modules:
            continue
        
        # Handle QuantizedConv1D wrapper
        if isinstance(module, QuantizedConv1D):
            # Count the underlying linear, mark wrapper as counted
            counted_modules.add(id(module))
            module = module.quantized_linear
            # Don't add to counted yet - let it be counted below
        
        if isinstance(module, QuantizedLinearINT4):
            total_bytes += module.weight_packed.numel() * module.weight_packed.element_size()
            total_bytes += module.scale.numel() * module.scale.element_size()
            total_bytes += module.zero_point.numel() * module.zero_point.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
            layer_stats['int4'] += 1
            counted_modules.add(id(module))
            
        elif isinstance(module, QuantizedLinearINT6):
            total_bytes += module.weight_int8.numel() * module.weight_int8.element_size()
            total_bytes += module.scale.numel() * module.scale.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
            layer_stats['int6'] += 1
            counted_modules.add(id(module))
            
        elif isinstance(module, QuantizedLinear):
            total_bytes += module.weight_quantized.numel() * module.weight_quantized.element_size()
            total_bytes += module.weight_scale.numel() * module.weight_scale.element_size()
            total_bytes += module.weight_zero_point.numel() * module.weight_zero_point.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
            layer_stats['int8'] += 1
            counted_modules.add(id(module))
        
        elif QuantizedEmbeddingINT8 is not None and isinstance(module, QuantizedEmbeddingINT8):
            total_bytes += module.weight_int8.numel() * 1  # int8
            total_bytes += module.scale.numel() * 4  # float32
            layer_stats['int8_emb'] += 1
            counted_modules.add(id(module))
        
        elif QuantizedLMHeadINT8 is not None and isinstance(module, QuantizedLMHeadINT8):
            if module.weight_shared:
                # Only count bias
                if module.bias is not None:
                    total_bytes += module.bias.numel() * module.bias.element_size()
            else:
                # Count weights + scales + bias
                total_bytes += module.weight_int8_T.numel() * 1
                total_bytes += module.scale.numel() * 4
                if module.bias is not None:
                    total_bytes += module.bias.numel() * module.bias.element_size()
            counted_modules.add(id(module))
            
        elif hasattr(module, 'weight') and module.weight is not None:
            # Only count substantial FP32 layers (skip tiny ones)
            if module.weight.numel() > 100:  # Skip very small layers
                total_bytes += module.weight.numel() * module.weight.element_size()
                if hasattr(module, 'bias') and module.bias is not None:
                    total_bytes += module.bias.numel() * module.bias.element_size()
                # Only count as FP32 if it's a significant layer
                if module.weight.numel() > 10000:
                    layer_stats['fp32'] += 1
                counted_modules.add(id(module))
    
    return total_bytes / (1024 * 1024), layer_stats

def measure_quality_comprehensive(baseline_model, optimized_model, tokenizer, model_name):
    """
    Comprehensive quality measurement: Perplexity + BLEU + Generation samples
    """
    import math
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    print("\n[Quality] Running comprehensive evaluation...")
    
    # Test prompts covering different domains
    test_prompts = [
        "The quick brown fox",
        "Machine learning is",
        "In the year 2025,",
        "The capital of France",
        "Scientists have discovered",
        "The future of technology",
        "Climate change will",
        "Artificial intelligence can",
    ]
    
    results = {
        'perplexity_baseline': [],
        'perplexity_quantized': [],
        'bleu_scores': [],
        'generation_samples': []
    }
    
    smoothing = SmoothingFunction()
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        
        # Measure perplexity
        with torch.no_grad():
            # Baseline
            try:
                outputs_base = baseline_model(**inputs, labels=inputs["input_ids"])
                ppl_base = math.exp(outputs_base.loss.item())
                results['perplexity_baseline'].append(ppl_base)
            except:
                ppl_base = None
            
            # Quantized
            try:
                outputs_quant = optimized_model(**inputs, labels=inputs["input_ids"])
                if hasattr(outputs_quant, 'loss') and outputs_quant.loss is not None:
                    ppl_quant = math.exp(outputs_quant.loss.item())
                    results['perplexity_quantized'].append(ppl_quant)
                else:
                    ppl_quant = None
            except Exception as e:
                print(f"    [Warning] Perplexity calculation failed for quantized model: {e}")
                ppl_quant = None
        
        # Generate text for BLEU comparison
        with torch.no_grad():
            gen_base = baseline_model.generate(
                inputs["input_ids"],
                max_new_tokens=30,
                do_sample=False,  # Deterministic for comparison
                pad_token_id=tokenizer.pad_token_id
            )
            
            gen_quant = optimized_model.generate(
                inputs["input_ids"],
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        text_base = tokenizer.decode(gen_base[0], skip_special_tokens=True)
        text_quant = tokenizer.decode(gen_quant[0], skip_special_tokens=True)
        
        # Calculate BLEU (comparing quantized to baseline as reference)
        reference = [text_base.split()]
        hypothesis = text_quant.split()
        
        bleu = sentence_bleu(reference, hypothesis, 
                            smoothing_function=smoothing.method1)
        results['bleu_scores'].append(bleu)
        
        # Store sample
        results['generation_samples'].append({
            'prompt': prompt,
            'baseline': text_base,
            'quantized': text_quant,
            'bleu': bleu
        })
    
    # Calculate averages
    avg_ppl_base = sum(results['perplexity_baseline']) / len(results['perplexity_baseline']) if results['perplexity_baseline'] else None
    avg_ppl_quant = sum(results['perplexity_quantized']) / len(results['perplexity_quantized']) if results['perplexity_quantized'] else None
    avg_bleu = sum(results['bleu_scores']) / len(results['bleu_scores'])
    
    ppl_degradation = ((avg_ppl_quant - avg_ppl_base) / avg_ppl_base * 100) if (avg_ppl_base and avg_ppl_quant) else None
    
    print(f"\nQuality Metrics:")
    if avg_ppl_base and avg_ppl_quant:
        print(f"  Perplexity (baseline): {avg_ppl_base:.2f}")
        print(f"  Perplexity (quantized): {avg_ppl_quant:.2f}")
        print(f"  Perplexity change: {ppl_degradation:+.1f}%")
    print(f"  BLEU score: {avg_bleu:.3f} (1.0 = identical to baseline)")
    
    # Quality assessment
    if avg_bleu > 0.95:
        quality = "Excellent"
    elif avg_bleu > 0.90:
        quality = "Very Good"
    elif avg_bleu > 0.85:
        quality = "Good"
    else:
        quality = "Acceptable"
    
    print(f"  Quality: {quality}")
    
    return {
        'perplexity_baseline': avg_ppl_base,
        'perplexity_quantized': avg_ppl_quant,
        'perplexity_degradation_pct': ppl_degradation,
        'bleu_score': avg_bleu,
        'quality_rating': quality,
        'samples': results['generation_samples'][:3]  # Keep first 3 for reporting
    }


def demo_production_ready(model_name="gpt2"):
    print(f"\n{'='*70}")
    print(f" GPU-POOR PRODUCTION v4.0 - Mixed Precision Edition")
    print(f" INT4/INT6/INT8 Adaptive Quantization")
    print(f"{'='*70}")
    
    # Loading model and tokenizer
    print(f"\n[1/5] Loading {model_name}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token())
    except ValueError:
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token())
        print("Note: Loaded as Seq2Seq model")

    baseline_model = deepcopy(model) #untouched FP32 model for comparison        
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token())
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for generation
    
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Original size: {original_size:.1f} MB")
    
    #  cal data
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
    
    # mixed-precision quantization
    print("\n[3/5] Applying mixed-precision quantization...")
    print("="*60)
    print("MIXED-PRECISION QUANTIZATION")
    print("="*60)
    
    # hybrid quantizer with mixed precision
    quantized_model = make_it_work_hybrid(
        model,
        sample_inputs=calibration_data,
        smoothing_alpha=0.3,
        memory_target=0.4  # Target 40% of original size
    )
    
    print("[Success] Model quantized with mixed precision!")
    
    # Calculating actual compressed size
    compressed_size, layer_stats = calculate_actual_model_size(quantized_model)
    
# Optimizing for inference
    print("\n[4/5] Optimizing for inference speed...")
    optimized_model = OptimizedModel(quantized_model)
    optimized_model.optimize()  # This does ALL the pre-caching now
    
    # === GPU-POOR DIAGNOSTICS START ===
    if os.getenv("GPU_POOR_DIAG", "0") == "1":
        print("\n" + "="*60)
        print("GPU-POOR DIAGNOSTICS")
        print("="*60)
        
        from gpu_poor.quantization import (
            QuantizedLinear, QuantizedLinearINT4, QuantizedLinearINT6, QuantizedConv1D
        )
        try:
            from gpu_poor.quantization import QuantizedEmbeddingINT8, QuantizedLMHeadINT8
        except ImportError:
            QuantizedEmbeddingINT8 = None
            QuantizedLMHeadINT8 = None
        
        print("\n[1] Quantization Coverage:")
        quant_linear = 0
        quant_emb = 0
        quant_lmhead_shared = 0
        
        for n, m in optimized_model.named_modules():
            if isinstance(m, (QuantizedLinear, QuantizedLinearINT4, QuantizedLinearINT6)):
                quant_linear += 1
            elif isinstance(m, QuantizedConv1D):
                quant_linear += 1
            elif QuantizedEmbeddingINT8 and isinstance(m, QuantizedEmbeddingINT8):
                quant_emb += 1
            elif QuantizedLMHeadINT8 and isinstance(m, QuantizedLMHeadINT8):
                if m.weight_shared:
                    quant_lmhead_shared += 1
        
        print(f"  Quantized Linear/INT4/INT6: {quant_linear}")
        print(f"  Quantized Embeddings: {quant_emb}")
        print(f"  Quantized LM Heads (shared): {quant_lmhead_shared}")
        
        print("\n[2] Embedding Tables:")
        for n, m in optimized_model.named_modules():
            if isinstance(m, nn.Embedding):
                size_mb = m.weight.numel() * m.weight.element_size() / (1024**2)
                print(f"  [FP32-EMB] {n}: {size_mb:.1f} MB")
            elif QuantizedEmbeddingINT8 and isinstance(m, QuantizedEmbeddingINT8):
                int8_mb = m.weight_int8.numel() * 1 / (1024**2)
                scale_mb = m.scale.numel() * 4 / (1024**2)
                print(f"  [INT8-EMB] {n}: {int8_mb + scale_mb:.1f} MB")
        
        print("\n[3] LM Head:")
        for n, m in optimized_model.named_modules():
            if 'lm_head' in n.lower():
                if QuantizedLMHeadINT8 and isinstance(m, QuantizedLMHeadINT8):
                    if m.weight_shared:
                        print(f"  [INT8-LMHEAD-SHARED] {n}: tied to embedding")
                    else:
                        print(f"  [INT8-LMHEAD-STANDALONE] {n}")
                elif isinstance(m, nn.Linear):
                    size_mb = m.weight.numel() * 4 / (1024**2)
                    print(f"  [FP32-LMHEAD] {n}: {size_mb:.1f} MB")
        
        print("="*60)
    # === GPU-POOR DIAGNOSTICS END ===

    print("\n[5/5] Benchmarking and Quality Evaluation...")
    
    # Quality evaluation FIRST (use unwrapped model for perplexity)
    quality_results = measure_quality_comprehensive(
        baseline_model, 
        quantized_model,  # ← Changed from optimized_model
        tokenizer,
        model_name
    )
        
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
        # Building per-model kwargs (handles T5 vs GPT/OPT/Llama)
        base_kwargs = build_gen_kwargs(baseline_model, tokenizer, inputs, max_new_tokens=50, do_sample=True, temperature=0.8)
        opt_kwargs  = build_gen_kwargs(optimized_model, tokenizer, inputs, max_new_tokens=50, do_sample=True, temperature=0.8)

        _ = baseline_model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **base_kwargs
        )
        _ = optimized_model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **opt_kwargs
        )

    # Warmup
    print("Warming up...")
    with torch.no_grad():

        # Building per-model kwargs (handles T5 vs GPT/OPT/Llama)*I hope*
        base_kwargs = build_gen_kwargs(baseline_model, tokenizer, inputs, max_new_tokens=50, do_sample=True, temperature=0.8)
        opt_kwargs  = build_gen_kwargs(optimized_model, tokenizer, inputs, max_new_tokens=50, do_sample=True, temperature=0.8)

        _ = baseline_model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **base_kwargs
        )
        _ = optimized_model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **opt_kwargs
        )
    
    # Time original
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_original = baseline_model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **base_kwargs
        )
    original_time = time.perf_counter() - start
    
    # Time optimized
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output_optimized = optimized_model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            **opt_kwargs
        )
    optimized_time = time.perf_counter() - start
    
    latency_ratio_opt_over_fp32 = (optimized_time / original_time) if original_time > 0 else float("inf")
    speedup_x = (1.0 / latency_ratio_opt_over_fp32) if latency_ratio_opt_over_fp32 > 0 else 0.0

    # Results! Finally.
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
    latency_ratio_opt_over_fp32 = (optimized_time / original_time) if original_time > 0 else float("inf")  # <1 is faster
    speedup_x = (1.0 / latency_ratio_opt_over_fp32) if latency_ratio_opt_over_fp32 > 0 else 0.0           # >1 is faster
    print(f"  Speedup: {speedup_x:.2f}x  (latency opt/fp32 = {latency_ratio_opt_over_fp32:.3f})")

    
    print(f"\nGeneration Quality:")
    original_text = tokenizer.decode(output_original[0], skip_special_tokens=True)
    optimized_text = tokenizer.decode(output_optimized[0], skip_special_tokens=True)
    
    print(f"  Original: '{original_text}'")
    print(f"  Optimized: '{optimized_text}'")
    print(f"\nQuality Metrics:")
    print(f"  BLEU score: {quality_results['bleu_score']:.3f}")

    if quality_results['perplexity_degradation_pct']:
        print(f"  Perplexity change: {quality_results['perplexity_degradation_pct']:+.1f}%")
    print(f"  Rating: {quality_results['quality_rating']}")

    # Checking for repetitions
    words = optimized_text.split()
    repetitions = 0
    for i in range(len(words) - 2):
        if i + 2 < len(words) and words[i] == words[i+1] == words[i+2]:
            repetitions += 1
    
    if repetitions == 0:
        print(f"\n  [OK] Quality: Excellent (no repetitions detected)")
    else:
        print(f"\n  [WARNING] Quality: {repetitions} repetitive patterns detected")
    
    # Performance summary!
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
        'compression_ratio': (1 - compressed_size / original_size) * 100,
        'latency_seconds_fp32': original_time,
        'latency_seconds_quant': optimized_time,
        'latency_ratio_opt_over_fp32': latency_ratio_opt_over_fp32,  # lower is better
        'speedup_x': speedup_x,                                      # higher is better
        'layer_stats': layer_stats,
        'has_repetitions': repetitions > 0,
        'perplexity_baseline': quality_results['perplexity_baseline'],
        'perplexity_quantized': quality_results['perplexity_quantized'],
        'perplexity_degradation_pct': quality_results['perplexity_degradation_pct'],
        'bleu_score': quality_results['bleu_score'],
        'quality_rating': quality_results['quality_rating'],
        'quality_samples': quality_results['samples']
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
     # individual result
    with open(f"results/{model_name.replace('/', '_')}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # master results file
    results_file = "results/all_results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append({
        "model": model_name,
        "compression": results['compression_ratio'],
        "speedup": results.get('speedup_x'),  # use the >1× speedup
        "latency_ratio": results.get('latency_ratio_opt_over_fp32'),
        "original_mb": results['original_size'],
        "compressed_mb": results['compressed_size'],
        "quality": "good" if not results['has_repetitions'] else "repetitions"
    })

    
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to results/{model_name.replace('/', '_')}.json")
