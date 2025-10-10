"""Advanced quantization with layer sensitivity analysis and mixed precision"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

class QuantizedLinear(nn.Module):
    """Flexible quantized linear layer supporting multiple bit widths"""
    
    def __init__(self, in_features, out_features, bits=8, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        if bits == 8:
            dtype = torch.int8
            self.qmin, self.qmax = -128, 127
        elif bits == 4:
            dtype = torch.int8  # Store 2x4-bit values in one int8
            self.qmin, self.qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
        
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=dtype))
        self.register_buffer('weight_scale', torch.ones(out_features, 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        # Dequantize weights
        if self.bits == 4:
            # Unpack 4-bit values (stored as int8)
            weight_float = (self.weight_quantized.float() - self.weight_zero_point) * self.weight_scale
        else:
            weight_float = (self.weight_quantized.float() - self.weight_zero_point) * self.weight_scale
        
        return F.linear(input.float(), weight_float, self.bias)
    
    @staticmethod
    def from_float(module, bits=8, symmetric=False):
        """Convert float module to quantized with specified bits"""
        weight_data = module.weight.data
        out_features, in_features = weight_data.shape
        
        quantized = QuantizedLinear(in_features, out_features, bits, 
                                   module.bias is not None)
        
        # Per-channel quantization for better accuracy
        if symmetric:
            # Symmetric quantization (better for some layers)
            max_vals = weight_data.abs().max(dim=1, keepdim=True)[0]
            scale = max_vals / ((2 ** (bits - 1)) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric quantization (generally better accuracy)
            min_vals = weight_data.min(dim=1, keepdim=True)[0]
            max_vals = weight_data.max(dim=1, keepdim=True)[0]
            
            qmin = -(2 ** (bits - 1))
            qmax = (2 ** (bits - 1)) - 1
            
            scale = (max_vals - min_vals) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
            zero_point = qmin - min_vals / scale
        
        # Quantize
        weight_int = torch.round(weight_data / scale + zero_point)
        weight_int = torch.clamp(weight_int, qmin, qmax).to(torch.int8)
        
        quantized.weight_quantized = weight_int
        quantized.weight_scale = scale
        quantized.weight_zero_point = zero_point
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()
        
        return quantized

class QuantizedLinearINT4(nn.Module):
    """Ultra-compact 4-bit quantization with packed storage"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Calculate packed size (2 4-bit values per byte)
        total_elements = out_features * in_features
        packed_size = (total_elements + 1) // 2
        
        self.register_buffer('weight_packed', torch.zeros(packed_size, dtype=torch.uint8))
        self.register_buffer('scale', torch.ones(out_features, 1))
        self.register_buffer('zero_point', torch.zeros(out_features, 1))
        
        self.weight_shape = (out_features, in_features)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # ADD THESE TWO LINES FOR CACHING
        self._cached_weight = None
        self._use_cache = True
    
    @staticmethod
    def from_float(module, bits=4):
        # KEEP YOUR EXISTING from_float METHOD UNCHANGED
        """Convert float module to INT4"""
        weight_data = module.weight.data
        out_features, in_features = weight_data.shape
        
        quantized = QuantizedLinearINT4(in_features, out_features, module.bias is not None)
        
        # [Rest of your existing from_float code stays the same]
        # Per-channel asymmetric quantization for INT4
        min_vals = weight_data.min(dim=1, keepdim=True)[0]
        max_vals = weight_data.max(dim=1, keepdim=True)[0]
        
        # 4-bit range: 0 to 15
        scale = (max_vals - min_vals) / 15
        scale = torch.clamp(scale, min=1e-8)
        zero_point = -min_vals / scale
        
        # Quantize to 4-bit
        weight_int4 = torch.round((weight_data - min_vals) / scale)
        weight_int4 = torch.clamp(weight_int4, 0, 15).to(torch.uint8)
        
        # Pack two 4-bit values per byte
        weight_flat = weight_int4.flatten()
        packed = torch.zeros((weight_flat.numel() + 1) // 2, dtype=torch.uint8)
        
        for i in range(0, weight_flat.numel(), 2):
            low = weight_flat[i]
            high = weight_flat[i+1] if i+1 < weight_flat.numel() else 0
            packed[i//2] = (high << 4) | (low & 0x0F)
        
        quantized.weight_packed = packed
        quantized.scale = scale
        quantized.zero_point = zero_point
        quantized.weight_shape = weight_data.shape
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()
        
        return quantized
    
    def forward(self, x):
        # Check cache first
        if self._use_cache and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        
        # VECTORIZED unpacking - replace the for-loop!
        device = self.weight_packed.device
        
        # Convert to int32 for bit operations
        packed_int = self.weight_packed.to(torch.int32)
        
        # Extract nibbles using vectorized operations (1000x faster!)
        low_nibbles = (packed_int & 0x0F).to(torch.float32)
        high_nibbles = ((packed_int >> 4) & 0x0F).to(torch.float32)
        
        # Stack and flatten
        unpacked = torch.stack([low_nibbles, high_nibbles], dim=1).flatten()
        
        # Truncate to exact size
        weight_size = self.weight_shape[0] * self.weight_shape[1]
        unpacked = unpacked[:weight_size]
        
        # Reshape and dequantize
        weight = unpacked.reshape(self.weight_shape)
        weight = weight * self.scale - self.zero_point * self.scale
        
        # Cache the result
        if self._use_cache:
            self._cached_weight = weight
        
        return F.linear(x, weight, self.bias)


class QuantizedLinearINT6(nn.Module):
    """6-bit quantization - better quality than INT4, smaller than INT8"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store in int8 but only use 6-bit range (-32 to 31)
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # ADD THESE TWO LINES FOR CACHING
        self._cached_weight = None
        self._use_cache = True
    
    @staticmethod
    def from_float(module, bits=6):
        # KEEP YOUR EXISTING from_float METHOD UNCHANGED
        """Convert float module to INT6"""
        weight_data = module.weight.data
        out_features, in_features = weight_data.shape
        
        quantized = QuantizedLinearINT6(in_features, out_features, module.bias is not None)
        
        # Per-channel symmetric quantization for INT6
        max_vals = weight_data.abs().max(dim=1, keepdim=True)[0]
        scale = max_vals / 31  # 6-bit range: -32 to 31
        scale = torch.clamp(scale, min=1e-8)
        
        # Quantize
        weight_int6 = torch.round(weight_data / scale)
        weight_int6 = torch.clamp(weight_int6, -32, 31).to(torch.int8)
        
        quantized.weight_int8 = weight_int6
        quantized.scale = scale
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()
        
        return quantized
    
    def forward(self, x):
        # ADD CACHE CHECK AT THE START
        if self._use_cache and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        
        # Dequantize weights
        weight = self.weight_int8.float() * self.scale
        
        # CACHE THE DEQUANTIZED WEIGHT
        if self._use_cache:
            self._cached_weight = weight
        
        return F.linear(x, weight, self.bias)

class QuantizedConv1D(nn.Module):
    """Wrapper for GPT-2 style Conv1D layers - FIXED"""
    
    def __init__(self, quantized_linear):
        super().__init__()
        self.quantized_linear = quantized_linear
        
        # Store proper attributes for GPT-2 compatibility
        self.nf = quantized_linear.out_features
        self.nx = quantized_linear.in_features
        
        # Set dummy weight for compatibility (actual weights are in quantized_linear)
        self.register_buffer('weight', torch.zeros(1, 1))
    
    def forward(self, x):
        # Conv1D in GPT-2 expects (batch, seq, hidden)
        size_out = x.size()[:-1] + (self.nf,)
        x_flat = x.view(-1, self.nx)
        
        # Use the quantized linear layer's forward method
        x_out = self.quantized_linear(x_flat)
        
        return x_out.view(size_out)

class LayerSensitivityAnalyzer:
    """Analyzes which layers are sensitive to quantization"""
    
    def __init__(self, model, sample_data=None):
        self.model = model
        self.sample_data = sample_data
        self.sensitivity_scores = {}
        self.activation_stats = defaultdict(dict)
        
    def analyze_sensitivity(self, num_samples=100):
        """Analyze layer sensitivity using multiple metrics"""
        print("[Analysis] Analyzing layer sensitivity to quantization...")
        
        # Hook to capture activations
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_stats[name]['mean'] = output.mean().item()
                    self.activation_stats[name]['std'] = output.std().item()
                    self.activation_stats[name]['max'] = output.abs().max().item()
            return hook
        
        # Register hooks for all linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) or module.__class__.__name__ == 'Conv1D':
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # Run forward pass with sample data if provided
        if self.sample_data is not None:
            with torch.no_grad():
                if isinstance(self.sample_data, torch.Tensor):
                    _ = self.model(self.sample_data[:min(num_samples, len(self.sample_data))])
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Calculate sensitivity scores
        for name, module in self.model.named_modules():
            if self._should_analyze(module, name):
                score = self._calculate_sensitivity_score(name, module)
                self.sensitivity_scores[name] = score
        
        return self.sensitivity_scores
    
    def _should_analyze(self, module, name):
        """Check if module should be analyzed"""
        # Skip embeddings and layer norms
        if any(skip in name.lower() for skip in ['embed', 'wte', 'wpe', 'ln', 'norm']):
            return False
        
        # Check if it has weights
        if not hasattr(module, 'weight'):
            return False
        
        # Only 2D weights
        if len(module.weight.shape) != 2:
            return False
        
        return True
    
    def _calculate_sensitivity_score(self, name, module):
        """Calculate sensitivity score for a layer"""
        score = 0.0
        weight = module.weight.data
        
        # Factor 1: Weight magnitude variation (high variation = sensitive)
        weight_std = weight.std().item()
        weight_mean = abs(weight.mean().item())
        if weight_mean > 1e-8:
            variation_score = min(weight_std / weight_mean, 1.0)  # Cap at 1.0
        else:
            variation_score = min(weight_std, 1.0)
        
        # Factor 2: Weight range (large range = sensitive)
        weight_range = (weight.max() - weight.min()).item()
        range_score = min(weight_range / 10.0, 1.0)  # Normalize to [0, 1]
        
        # Factor 3: Activation statistics (if available)
        activation_score = 0.0
        if name in self.activation_stats:
            stats = self.activation_stats[name]
            if 'max' in stats:
                activation_score = min(stats['max'] / 10.0, 1.0)
        
        # Factor 4: Layer position (early and late layers more sensitive)
        position_score = self._get_position_score(name)
        
        # Combine factors with adjusted weights for better distribution
        score = (variation_score * 0.2 + 
                range_score * 0.2 + 
                activation_score * 0.1 + 
                position_score * 0.5)  # Position is most important
        
        return score
    
    def _get_position_score(self, name):
        """Score based on layer position (early/late layers more sensitive)"""
        # GPT-2 specific patterns
        if 'h.0.' in name:  # First transformer block
            if 'attn' in name:
                return 0.9  # Keep first attention layers high quality
            else:
                return 0.6  # MLP can be quantized
        elif 'h.1.' in name:  # Second block
            return 0.5
        elif 'h.10.' in name or 'h.11.' in name:  # Last two blocks (for 12-layer model)
            if 'c_proj' in name:  # Output projections more important
                return 0.85
            else:
                return 0.45
        elif 'ln_f' in name or 'head' in name or 'lm_head' in name:  # Final layers
            return 0.95
        elif 'wte' in name or 'wpe' in name:  # Embeddings (shouldn't quantize)
            return 1.0
        elif any(f'h.{i}.' in name for i in [2, 3, 4, 5, 6, 7, 8, 9]):  # Middle layers
            return 0.15  # Much less sensitive, aggressive quantization
        else:
            return 0.4  # Default

class AdaptiveQuantizer:
    """Main class for adaptive mixed-precision quantization"""
    
    def __init__(self, memory_target=0.4, quality_threshold=0.8):
        """
        Args:
            memory_target: Target memory usage (0.4 = 40% of original)
            quality_threshold: Minimum quality to maintain (0.8 = 80%)
        """
        self.memory_target = memory_target
        self.quality_threshold = quality_threshold
        self.quantization_config = {}
        
    def create_quantization_plan(self, model, sample_data=None):
        """Create optimal quantization plan for the model"""
        print("[Quantization] Creating adaptive quantization plan...")
        
        # Analyze sensitivity
        analyzer = LayerSensitivityAnalyzer(model, sample_data)
        sensitivity_scores = analyzer.analyze_sensitivity()
        
        # Sort layers by sensitivity
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1])
        
        # Calculate memory for each configuration
        total_params = sum(p.numel() for p in model.parameters())
        current_memory_ratio = 0.0
        
        # Assign bit widths based on sensitivity
        for layer_name, sensitivity in sorted_layers:
            if sensitivity > 0.8:  # Very sensitive - keep original
                self.quantization_config[layer_name] = 'fp16'
                memory_contrib = self._get_layer_params(model, layer_name) / total_params * 1.0
            elif sensitivity > 0.4:  # Moderately sensitive - INT8 with care
                self.quantization_config[layer_name] = 'int8'
                memory_contrib = self._get_layer_params(model, layer_name) / total_params * 0.25
            else:  # Not sensitive - aggressive INT8
                self.quantization_config[layer_name] = 'int8'
                memory_contrib = self._get_layer_params(model, layer_name) / total_params * 0.25
            
            current_memory_ratio += memory_contrib
            
            # Stop if we hit memory target
            if current_memory_ratio >= (1 - self.memory_target):
                break
        
        print(f"[Quantization] Plan created: {len(self.quantization_config)} layers configured")
        self._print_quantization_summary()
        
        return self.quantization_config
    
    def _get_layer_params(self, model, layer_name):
        """Get parameter count for a specific layer"""
        for name, module in model.named_modules():
            if name == layer_name:
                return sum(p.numel() for p in module.parameters())
        return 0
    
    def _print_quantization_summary(self):
        """Print summary of quantization plan"""
        fp16_count = sum(1 for v in self.quantization_config.values() if v == 'fp16')
        int8_count = sum(1 for v in self.quantization_config.values() if v == 'int8')
        
        print(f"  FP16 layers (sensitive): {fp16_count}")
        print(f"  INT8 layers (standard): {int8_count}")
    
    def apply_quantization(self, model):
        """Apply the quantization plan to the model"""
        print("[Quantization] Applying adaptive quantization...")
        quantized_count = 0
        fp16_count = 0
        
        def replace_with_quantized(parent, name, module, full_name):
            nonlocal quantized_count, fp16_count
            
            if full_name not in self.quantization_config:
                return False
            
            config = self.quantization_config[full_name]
            
            if config == 'fp16':
                # For FP16, we don't actually convert the layer itself
                # We just mark it to be kept at higher precision
                # The actual FP16 conversion should be done at model level if needed
                print(f"  Keeping in higher precision: {full_name}")
                fp16_count += 1
            elif config == 'int8':
                # Quantize to INT8
                try:
                    if isinstance(module, nn.Linear):
                        quantized = QuantizedLinear.from_float(module, bits=8)
                        setattr(parent, name, quantized)
                        print(f"  Quantized to INT8: {full_name}")
                        quantized_count += 1
                    elif module.__class__.__name__ == 'Conv1D':
                        # Handle Conv1D layers
                        weight_data = module.weight.data.t()
                        temp_module = nn.Linear(weight_data.shape[1], weight_data.shape[0])
                        temp_module.weight.data = weight_data
                        temp_module.bias = module.bias if hasattr(module, 'bias') else None
                        
                        quantized = QuantizedLinear.from_float(temp_module, bits=8)
                        # Wrap in Conv1D wrapper
                        quantized_conv = QuantizedConv1D(quantized)
                        setattr(parent, name, quantized_conv)
                        print(f"  Quantized Conv1D to INT8: {full_name}")
                        quantized_count += 1
                except Exception as e:
                    print(f"  Warning: Could not quantize {full_name}: {e}")
            
            return True

        
        def recursive_apply(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if not replace_with_quantized(module, name, child, full_name):
                    recursive_apply(child, full_name)
        
        recursive_apply(model)
        
        print(f"\n[Quantization] Complete!")
        print(f"  INT8 quantized: {quantized_count} layers")
        print(f"  Kept original: {fp16_count} layers")
        return model

def make_it_work_adaptive(model, sample_data=None, memory_target=0.4):
    """Enhanced make_it_work with adaptive quantization"""
    quantizer = AdaptiveQuantizer(memory_target=memory_target)
    quantizer.create_quantization_plan(model, sample_data)
    model = quantizer.apply_quantization(model)
    return model

# Keep your original quantize_model function for backward compatibility
def quantize_model(model):
    """
    Original uniform quantization - handles Linear, Conv1D, and other layer types
    """
    quantized_count = 0
    skipped_layers = []
    
    def should_quantize_module(module, name):
        """Determine if a module should be quantized"""
        if isinstance(module, nn.Embedding):
            return False
        if any(skip in name.lower() for skip in ['embed', 'wte', 'wpe', 'ln', 'norm']):
            return False
        if not hasattr(module, 'weight'):
            return False
        if len(module.weight.shape) != 2:
            return False
        if module.weight.numel() < 1000:
            return False
        return True
    
    def replace_module(parent, name, module, full_name):
        nonlocal quantized_count, skipped_layers
        
        if not should_quantize_module(module, full_name):
            if hasattr(module, 'weight'):
                skipped_layers.append(f"{full_name} (skipped: {module.__class__.__name__})")
            return False
        
        try:
            if isinstance(module, nn.Linear):
                quantized = QuantizedLinear.from_float(module, bits=8)
                setattr(parent, name, quantized)
                print(f"  Quantized Linear: {full_name} ({module.in_features} -> {module.out_features})")
                quantized_count += 1
                return True
            elif module.__class__.__name__ == 'Conv1D':
                weight_data = module.weight.data.t()
                bias_data = module.bias.data if hasattr(module, 'bias') and module.bias is not None else None
                
                temp_module = nn.Linear(weight_data.shape[1], weight_data.shape[0])
                temp_module.weight.data = weight_data
                if bias_data is not None:
                    temp_module.bias = nn.Parameter(bias_data)
                
                quantized_linear = QuantizedLinear.from_float(temp_module, bits=8)
                quantized_conv = QuantizedConv1D(quantized_linear)
                setattr(parent, name, quantized_conv)
                
                in_features = module.weight.shape[0]
                out_features = module.weight.shape[1]
                print(f"  Quantized Conv1D: {full_name} ({in_features} -> {out_features})")
                quantized_count += 1
                return True
                    
        except Exception as e:
            print(f"  Warning: Could not quantize {full_name}: {e}")
            skipped_layers.append(f"{full_name} (error: {str(e)[:50]})")
        
        return False
    
    def recursive_quantize(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if not replace_module(module, name, child, full_name):
                recursive_quantize(child, full_name)
    
    print("[Quantization] Starting universal INT8 quantization...")
    print("[Quantization] Detecting all layers (skipping embeddings)...")
    
    recursive_quantize(model)
    
    print(f"\n[Quantization] Summary:")
    print(f"  Total layers quantized: {quantized_count}")
    
    if quantized_count == 0:
        print("[Warning] No layers were quantized! Check model architecture.")
    else:
        print(f"[Success] Quantized {quantized_count} layers to INT8!")
        
    return model

# Backward compatibility wrapper
def make_it_work(model, adaptive=True, **kwargs):
    """Main API - now with adaptive quantization by default"""
    if adaptive:
        return make_it_work_adaptive(model, **kwargs)
    else:
        return quantize_model(model)

# Test code when running as script
if __name__ == "__main__":
    """Test the adaptive quantization with GPT-2"""
    import time
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    print("\n" + "="*60)
    print("GPU-POOR: Advanced Adaptive Quantization Test")
    print("="*60)
    
    # Load model and tokenizer
    print("\n[1/4] Loading GPT-2 model...")
    model_name = "gpt2"  # You can also try "gpt2-medium" for larger model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    # Calculate original size
    original_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original model size: {original_params / 1024**2:.2f} MB")
    
    # Prepare sample data for sensitivity analysis
    print("\n[2/4] Preparing calibration data...")
    sample_text = "The quick brown fox jumps over the lazy dog. "
    inputs = tokenizer(sample_text, return_tensors="pt")
    sample_data = inputs["input_ids"]
    
    # Test original model generation
    print("\n[3/4] Testing original model generation...")
    tokenizer.pad_token = tokenizer.eos_token
    test_prompt = "The future of artificial intelligence is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    original_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Original output:\n  {original_text}")
    
    # Apply adaptive quantization
    print("\n[4/4] Applying adaptive quantization...")
    print("-" * 40)
    
    # You can test different approaches:
    # Option 1: Adaptive quantization (recommended)
    quantized_model = make_it_work_adaptive(
        model, 
        sample_data=sample_data,
        memory_target=0.4  # Target 40% of original size
    )
    
    # Option 2: If you want to compare with the old uniform quantization
    # quantized_model = quantize_model(model)  # Your original function
    
    print("-" * 40)
    
    # Calculate quantized size (approximate)
    quantized_params = 0
    int8_layers = 0
    original_layers = 0
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedLinear):
            # INT8 weights take 1/4 the space of FP32
            quantized_params += module.weight_quantized.numel() * 1  # 1 byte per INT8
            if module.bias is not None:
                quantized_params += module.bias.numel() * 4  # Bias stays FP32
            int8_layers += 1
        elif isinstance(module, QuantizedConv1D):
            # Conv1D wrapper - count the underlying quantized linear
            quantized_params += module.quantized_linear.weight_quantized.numel() * 1
            if module.quantized_linear.bias is not None:
                quantized_params += module.quantized_linear.bias.numel() * 4
            int8_layers += 1
        elif hasattr(module, 'weight'):
            # Non-quantized layers
            quantized_params += module.weight.numel() * module.weight.element_size()
            if hasattr(module, 'bias') and module.bias is not None:
                quantized_params += module.bias.numel() * module.bias.element_size()
            original_layers += 1
    
    print(f"\nQuantized model size: {quantized_params / 1024**2:.2f} MB")
    print(f"Memory reduction: {(1 - quantized_params/original_params) * 100:.1f}%")
    
    # Test quantized model generation
    print("\n[Testing] Quantized model generation...")
    with torch.no_grad():
        outputs = quantized_model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )
    
    quantized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Quantized output:\n  {quantized_text}")
    
    # Check for repetition
    print("\n[Analysis] Checking for repetitive patterns...")
    words = quantized_text.split()
    repetitions = 0
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            repetitions += 1
    
    if repetitions > 0:
        print(f"Warning: Found {repetitions} repetitive patterns")
        print("   Consider adjusting sensitivity thresholds or keeping more layers in FP16")
    else:
        print("No significant repetitive patterns detected!")
    
    print("\n" + "="*60)
    print("Test complete! Compare the outputs above.")
    print("="*60)
    
    # Optional: Interactive testing
    print("\nWant to test with your own prompts? (y/n): ", end="")
    import sys
    response = input().strip().lower()
    
    if response == 'y':
        while True:
            print("\nEnter a prompt (or 'quit' to exit): ", end="")
            prompt = input().strip()
            
            if prompt.lower() == 'quit':
                break
            
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = quantized_model.generate(
                    inputs["input_ids"],
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Output: {result}\n")