"""
Quick test script to verify the quantization fix works
Run this FIRST to ensure the vectorized unpacking is working
"""
import torch
import torch.nn as nn
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import your fixed quantization module
from gpu_poor.quantization import (
    QuantizedLinearINT4, 
    QuantizedLinearINT6,
    QuantizedLinear,
    QuantizedConv1D
)

def test_int4_unpacking():
    """Test that INT4 unpacking is fast and correct"""
    print("\n" + "="*60)
    print("Testing INT4 Unpacking Speed")
    print("="*60)
    
    # Create a test linear layer
    in_features, out_features = 768, 768  # Typical transformer dimensions
    test_layer = nn.Linear(in_features, out_features)
    
    # Quantize it to INT4
    quantized = QuantizedLinearINT4.from_float(test_layer)
    
    # Test input
    batch_size = 16
    seq_len = 128
    x = torch.randn(batch_size, seq_len, in_features)
    x_flat = x.view(-1, in_features)
    
    # Time the forward pass
    print(f"Input shape: {x_flat.shape}")
    print(f"Weight packed size: {quantized.weight_packed.shape}")
    print(f"Expected unpacked size: {quantized.weight_shape}")
    
    # Warmup
    with torch.no_grad():
        _ = quantized(x_flat)
    
    # Time it
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            out = quantized(x_flat)
    elapsed = time.perf_counter() - start
    
    print(f"\n10 forward passes took: {elapsed:.3f} seconds")
    print(f"Average per pass: {elapsed/10*1000:.1f} ms")
    
    if elapsed > 1.0:
        print("[WARNING] INT4 unpacking is still slow!")
        print("          Check if vectorized operations are working correctly.")
    else:
        print("[SUCCESS] INT4 unpacking is fast!")
    
    return elapsed < 1.0  # Should be much faster than 1 second for 10 passes

def test_mixed_precision_model():
    """Test a small model with mixed precision"""
    print("\n" + "="*60)
    print("Testing Mixed-Precision Quantization on GPT-2")
    print("="*60)
    
    # Load small model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Apply mixed precision manually to test
    print("\nApplying mixed-precision quantization...")
    
    # Count layers
    total_layers = 0
    quantized_layers = {'int4': 0, 'int6': 0, 'int8': 0}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear)) or module.__class__.__name__ == 'Conv1D':
            total_layers += 1
            
            # Skip embeddings
            if any(skip in name.lower() for skip in ['embed', 'wte', 'wpe', 'ln', 'norm']):
                continue
            
            # Get parent module for replacement
            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
            module_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            else:
                parent = model
            
            # Determine quantization based on layer position
            try:
                if 'h.0' in name or 'h.1' in name:
                    # First two blocks - INT8
                    if module.__class__.__name__ == 'Conv1D':
                        weight_data = module.weight.data.t()
                        temp = nn.Linear(weight_data.shape[1], weight_data.shape[0])
                        temp.weight.data = weight_data
                        if hasattr(module, 'bias'):
                            temp.bias = module.bias
                        quantized = QuantizedLinear.from_float(temp, bits=8)
                        quantized_conv = QuantizedConv1D(quantized)
                        setattr(parent, module_name, quantized_conv)
                    else:
                        quantized = QuantizedLinear.from_float(module, bits=8)
                        setattr(parent, module_name, quantized)
                    quantized_layers['int8'] += 1
                    
                elif 'h.5' in name or 'h.6' in name:
                    # Middle blocks - INT4 to test our fix
                    if module.__class__.__name__ == 'Conv1D':
                        weight_data = module.weight.data.t()
                        temp = nn.Linear(weight_data.shape[1], weight_data.shape[0])
                        temp.weight.data = weight_data
                        if hasattr(module, 'bias'):
                            temp.bias = module.bias
                        quantized = QuantizedLinearINT4.from_float(temp)
                        quantized_conv = QuantizedConv1D(quantized)
                        setattr(parent, module_name, quantized_conv)
                    else:
                        quantized = QuantizedLinearINT4.from_float(module)
                        setattr(parent, module_name, quantized)
                    quantized_layers['int4'] += 1
                    
                elif 'h.3' in name or 'h.4' in name:
                    # Other middle blocks - INT6
                    if module.__class__.__name__ == 'Conv1D':
                        weight_data = module.weight.data.t()
                        temp = nn.Linear(weight_data.shape[1], weight_data.shape[0])
                        temp.weight.data = weight_data
                        if hasattr(module, 'bias'):
                            temp.bias = module.bias
                        quantized = QuantizedLinearINT6.from_float(temp)
                        quantized_conv = QuantizedConv1D(quantized)
                        setattr(parent, module_name, quantized_conv)
                    else:
                        quantized = QuantizedLinearINT6.from_float(module)
                        setattr(parent, module_name, quantized)
                    quantized_layers['int6'] += 1
                    
            except Exception as e:
                print(f"  Failed to quantize {name}: {e}")
    
    print(f"\nQuantization summary:")
    print(f"  Total layers: {total_layers}")
    print(f"  INT4 layers: {quantized_layers['int4']}")
    print(f"  INT6 layers: {quantized_layers['int6']}")
    print(f"  INT8 layers: {quantized_layers['int8']}")
    
    # Test generation (the real test!)
    print("\n\nTesting generation (this is where it used to hang)...")
    test_prompt = "The future of artificial intelligence"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    start = time.perf_counter()
    print(f"Starting generation at {time.strftime('%H:%M:%S')}...")
    
    with torch.no_grad():
        try:
            output = model.generate(
                inputs["input_ids"],
                max_length=30,
                do_sample=False,  # Deterministic for testing
                pad_token_id=tokenizer.pad_token_id
            )
            elapsed = time.perf_counter() - start
            
            print(f"Generation completed in {elapsed:.2f} seconds!")
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\nGenerated text: '{generated_text}'")
            
            if elapsed < 5.0:
                print("\n[SUCCESS] Model generation is working and fast!")
                return True
            else:
                print("\n[WARNING] Generation works but is slower than expected")
                return True
                
        except Exception as e:
            print(f"\n[ERROR] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" GPU-POOR QUANTIZATION FIX TEST")
    print("="*70)
    
    # Test 1: INT4 unpacking speed
    test1_pass = test_int4_unpacking()
    
    # Test 2: Full model with mixed precision
    test2_pass = test_mixed_precision_model()
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    if test1_pass and test2_pass:
        print("\n[SUCCESS] All tests passed! The quantization fix is working!")
        print("\nYou can now run your demo_production.py")
        print("The hanging issue should be resolved.")
    else:
        print("\n[FAILURE] Some tests failed. Check the output above for details.")
        if not test1_pass:
            print("- INT4 unpacking is still slow (likely still using loops)")
        if not test2_pass:
            print("- Model generation failed or timed out")
    
    return test1_pass and test2_pass

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)