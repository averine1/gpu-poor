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
            dtype = torch.int8  
            self.qmin, self.qmax = -8, 7
        else:
            raise ValueError(f"Unsupported bit width: {bits}")
        
        self.register_buffer('weight_quantized', torch.zeros(out_features, in_features, dtype=dtype))
        self.register_buffer('weight_scale', torch.ones(out_features, 1))
        self.register_buffer('weight_zero_point', torch.zeros(out_features, 1))
        
        #add caching 
        self._cached_weight = None
        self._use_cache = True

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        #(massive speedup for generation)
        if self._use_cache and self._cached_weight is not None:
            return F.linear(input.float(), self._cached_weight, self.bias)
        
        #dequantize weights
        weight_float = (self.weight_quantized.float() - self.weight_zero_point) * self.weight_scale
        

        if self._use_cache:
            self._cached_weight = weight_float.detach()
        
        return F.linear(input.float(), weight_float, self.bias)
    
    @staticmethod
    def from_float(module, bits=8, symmetric=False):
        """Convert float module to quantized with specified bits"""
        weight_data = module.weight.data
        out_features, in_features = weight_data.shape
        
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

        quantized = QuantizedLinear(in_features, out_features, bits, 
                                   module.bias is not None)
        
        #per-channel quantization for better accuracy
        if symmetric:
            #symmetric quantization (better for some layers)
            max_vals = weight_data.abs().max(dim=1, keepdim=True)[0]
            scale = max_vals / ((2 ** (bits - 1)) - 1)
            zero_point = torch.zeros_like(scale)
        else:
            #asymmetric quantization (generally better accuracy)
            min_vals = weight_data.min(dim=1, keepdim=True)[0]
            max_vals = weight_data.max(dim=1, keepdim=True)[0]
            
            qmin = -(2 ** (bits - 1))
            qmax = (2 ** (bits - 1)) - 1
            
            scale = (max_vals - min_vals) / (qmax - qmin)
            scale = torch.clamp(scale, min=1e-8) 
            zero_point = qmin - min_vals / scale
        
        #quantize
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
        
      
        self._cached_weight = None
        self._use_cache = True
    
    @staticmethod
    def from_float(module, bits=4):
  
        """Convert float module to INT4"""
        weight_data = module.weight.data
        out_features, in_features = weight_data.shape
        
        quantized = QuantizedLinearINT4(in_features, out_features, module.bias is not None)
        
      
        # per-channel asymmetric quantization for INT4
        min_vals = weight_data.min(dim=1, keepdim=True)[0]
        max_vals = weight_data.max(dim=1, keepdim=True)[0]
        
        #0 to 15
        scale = (max_vals - min_vals) / 15
        scale = torch.clamp(scale, min=1e-8)
        zero_point = -min_vals / scale
        
       
        weight_int4 = torch.round((weight_data - min_vals) / scale)
        weight_int4 = torch.clamp(weight_int4, 0, 15).to(torch.uint8)
        
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
       
        if self._use_cache and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        
        #VECTORIZED unpacking 
        device = self.weight_packed.device
        
        #to int32 for bit operations
        packed_int = self.weight_packed.to(torch.int32)
        
        #extract nibbles
        low_nibbles = (packed_int & 0x0F).to(torch.float32)
        high_nibbles = ((packed_int >> 4) & 0x0F).to(torch.float32)
        

        unpacked = torch.stack([low_nibbles, high_nibbles], dim=1).flatten()
        
        weight_size = self.weight_shape[0] * self.weight_shape[1]
        unpacked = unpacked[:weight_size]
        
        #reshape and dequantize
        weight = unpacked.reshape(self.weight_shape)
        weight = weight * self.scale - self.zero_point * self.scale

        if self._use_cache:
            self._cached_weight = weight
        
        return F.linear(x, weight, self.bias)


class QuantizedLinearINT6(nn.Module):
    """6-bit quantization - better quality than INT4, smaller than INT8"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        #in int8 but only use 6-bit range (-32 to 31)
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features, 1))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        

        self._cached_weight = None
        self._use_cache = True
    
    @staticmethod
    def from_float(module, bits=6):

        """Convert float module to INT6"""
        weight_data = module.weight.data
        out_features, in_features = weight_data.shape
        
        quantized = QuantizedLinearINT6(in_features, out_features, module.bias is not None)
        
        #per-channel symmetric quantization for INT6
        max_vals = weight_data.abs().max(dim=1, keepdim=True)[0]
        scale = max_vals / 31  # 6-bit range: -32 to 31
        scale = torch.clamp(scale, min=1e-8)
        
        #quantize
        weight_int6 = torch.round(weight_data / scale)
        weight_int6 = torch.clamp(weight_int6, -32, 31).to(torch.int8)
        
        quantized.weight_int8 = weight_int6
        quantized.scale = scale
        
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()
        
        return quantized
    
    def forward(self, x):
        if self._use_cache and self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        
        #dequantize weights
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
        
        self.nf = quantized_linear.out_features
        self.nx = quantized_linear.in_features
        
        #dummy weight for compatibility 
        self.register_buffer('weight', torch.zeros(1, 1))
    
    def forward(self, x):
        # Conv1D in GPT-2 expects (batch, seq, hidden)
        size_out = x.size()[:-1] + (self.nf,)
        x_flat = x.reshape(-1, self.nx)
        
        #quantized linear layer's forward method
        x_out = self.quantized_linear(x_flat)
        
        return x_out.view(size_out)

class QuantizedEmbeddingINT8(nn.Module):
    """
    Minimal INT8 embedding with per-row symmetric scales.
    - Stores weights as int8 and one float32 scale per row.
    - Dequantizes only the rows used by the current batch (fast on CPU).
    """
    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        # [num_embeddings, embedding_dim]
        self.register_buffer("weight_int8", weight_int8)  # int8
        self.register_buffer("scale", scale)              # float32
       
        self.num_embeddings = weight_int8.size(0)
        self.embedding_dim = weight_int8.size(1)

    @classmethod
    def from_float(cls, emb: nn.Embedding) -> "QuantizedEmbeddingINT8":
        W = emb.weight.detach()
        # symmetric per-row scales: max(|row|)/127 (avoid zero scale)
        scale = W.abs().amax(dim=1).clamp(min=1e-8) / 127.0          # [N]
        Wq = torch.round(W / scale.unsqueeze(1)).to(torch.int8)     # [N, D]
        return cls(Wq, scale)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        # Dequantize only the rows we need this step
        flat = input_ids.view(-1)                                     # [B*T]
        rows = self.weight_int8.index_select(0, flat).float()      # [B*T, D]
        s = self.scale.index_select(0, flat).unsqueeze(1)       # [B*T, 1]
        rows = rows * s                    # dequant
        B, T = input_ids.shape
        D = rows.shape[-1]
        return rows.view(B, T, D)
    
class QuantizedLMHeadINT8(nn.Module):
    """
    INT8 LM head with weight tying to embedding.
    - Stores transposed weight [embed_dim, vocab_size] for efficient matmul
    - Shares buffers with QuantizedEmbeddingINT8 when weight tying is used
    """
    def __init__(self, weight_int8_T: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight_int8_T", weight_int8_T)  # [embed_dim, vocab_size]
        self.register_buffer("scale", scale)  # [vocab_size]
        
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)
        
        self.embed_dim = weight_int8_T.size(0)
        self.vocab_size = weight_int8_T.size(1)
        self.weight_shared = False

    @classmethod
    def from_embedding(cls, qemb: "QuantizedEmbeddingINT8", bias: Optional[torch.Tensor] = None):
        """Create lm_head from quantized embedding with weight tying."""
        weight_int8_T = qemb.weight_int8.t().contiguous()  # [vocab, embed] → [embed, vocab]
        scale = qemb.scale.clone()
        instance = cls(weight_int8_T, scale, bias)
        instance.weight_shared = True
        return instance

    @classmethod
    def from_float(cls, module: nn.Linear):
        """Quantize a standalone linear layer to INT8."""
        W = module.weight.detach()  # [vocab_size, embed_dim]
        B = module.bias.detach() if module.bias is not None else None
        
        scale = W.abs().amax(dim=1).clamp(min=1e-8) / 127.0  # [vocab_size]
        Wq = torch.round(W / scale.unsqueeze(1)).to(torch.int8)
        weight_int8_T = Wq.t().contiguous()  # [embed, vocab]
        
        return cls(weight_int8_T, scale, B)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward: (B, T, D) @ (D, V) → (B, T, V)"""
        weight_float = self.weight_int8_T.float()  # [D, V]
        logits = hidden_states @ weight_float  # [..., vocab]
        logits = logits * self.scale.unsqueeze(0)
        if self.bias is not None:
            logits = logits + self.bias
        return logits
    
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
        
        handles = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_stats[name]['mean'] = output.mean().item()
                    self.activation_stats[name]['std'] = output.std().item()
                    self.activation_stats[name]['max'] = output.abs().max().item()
            return hook
        
       
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)) or module.__class__.__name__ == 'Conv1D':
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        #forward pass with sample data
        if self.sample_data is not None:
            with torch.no_grad():
                if isinstance(self.sample_data, torch.Tensor):
                    _ = self.model(self.sample_data[:min(num_samples, len(self.sample_data))])
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        
        for name, module in self.model.named_modules():
            if self._should_analyze(module, name):
                score = self._calculate_sensitivity_score(name, module)
                self.sensitivity_scores[name] = score
        
        return self.sensitivity_scores
    
    def _should_analyze(self, module, name):
        """Check if module should be analyzed"""
        
        if any(skip in name.lower() for skip in ['embed', 'wte', 'wpe', 'ln', 'norm']):
            return False
        
  
        if not hasattr(module, 'weight'):
            return False
        
      
        if len(module.weight.shape) != 2:
            return False
        
        return True
    
    def _calculate_sensitivity_score(self, name, module):
        """Calculate sensitivity score for a layer"""
        score = 0.0
        weight = module.weight.data
        
        #weight magnitude variation (high variation = sensitive)
        weight_std = weight.std().item()
        weight_mean = abs(weight.mean().item())
        if weight_mean > 1e-8:
            variation_score = min(weight_std / weight_mean, 1.0)  # Cap at 1.0
        else:
            variation_score = min(weight_std, 1.0)
        
        #weight range (large range = sensitive)
        weight_range = (weight.max() - weight.min()).item()
        range_score = min(weight_range / 10.0, 1.0)  # Normalize to [0, 1]
        
        #activation statistics
        activation_score = 0.0
        if name in self.activation_stats:
            stats = self.activation_stats[name]
            if 'max' in stats:
                activation_score = min(stats['max'] / 10.0, 1.0)
        
        #(early and late layers more sensitive)
        position_score = self._get_position_score(name)
        
        score = (variation_score * 0.2 + 
                range_score * 0.2 + 
                activation_score * 0.1 + 
                position_score * 0.5)  
        
        return score
    
    def _get_position_score(self, name):
        """Score based on layer position (early/late layers more sensitive)"""
        #GPT-2 specific patterns
        if 'h.0.' in name: 
            if 'attn' in name:
                return 0.9  
            else:
                return 0.6  # MLP can be quantized
        elif 'h.1.' in name: 
            return 0.5
        elif 'h.10.' in name or 'h.11.' in name: 
            if 'c_proj' in name:  
                return 0.85
            else:
                return 0.45
        elif 'ln_f' in name or 'head' in name or 'lm_head' in name:  
            return 0.95
        elif 'wte' in name or 'wpe' in name:  #embeddings 
            return 1.0
        elif any(f'h.{i}.' in name for i in [2, 3, 4, 5, 6, 7, 8, 9]):  
            return 0.15  #much less sensitive, aggressive quantization
        else:
            return 0.4  

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
        
        #analyze sensitivity
        analyzer = LayerSensitivityAnalyzer(model, sample_data)
        sensitivity_scores = analyzer.analyze_sensitivity()
        
        #sort layers by sensitivity
        sorted_layers = sorted(sensitivity_scores.items(), key=lambda x: x[1])
        
        
        total_params = sum(p.numel() for p in model.parameters())
        current_memory_ratio = 0.0
        
      
        for layer_name, sensitivity in sorted_layers:
            if sensitivity > 0.8:  
                self.quantization_config[layer_name] = 'fp16'
                memory_contrib = self._get_layer_params(model, layer_name) / total_params * 1.0
            elif sensitivity > 0.4:  
                self.quantization_config[layer_name] = 'int8'
                memory_contrib = self._get_layer_params(model, layer_name) / total_params * 0.25
            else:  
                self.quantization_config[layer_name] = 'int8'
                memory_contrib = self._get_layer_params(model, layer_name) / total_params * 0.25
            
            current_memory_ratio += memory_contrib
            
            #if we hit memory target
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
              
                print(f"  Keeping in higher precision: {full_name}")
                fp16_count += 1
            elif config == 'int8':
                #quantize to INT8
                try:
                    if isinstance(module, nn.Linear):
                        quantized = QuantizedLinear.from_float(module, bits=8)
                        setattr(parent, name, quantized)
                        print(f"  Quantized to INT8: {full_name}")
                        quantized_count += 1
                    elif module.__class__.__name__ == 'Conv1D':
                        #handle Conv1D layers
                        weight_data = module.weight.data.t()
                        temp_module = nn.Linear(weight_data.shape[1], weight_data.shape[0])
                        temp_module.weight.data = weight_data
                        temp_module.bias = module.bias if hasattr(module, 'bias') else None
                        
                        quantized = QuantizedLinear.from_float(temp_module, bits=8)
                        #wrap in Conv1D wrapper
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

#backward compatibility wrapper
def make_it_work(model, adaptive=True, **kwargs):
    """Main API - now with adaptive quantization by default"""
    if adaptive:
        return make_it_work_adaptive(model, **kwargs)
    else:
        return quantize_model(model)

if __name__ == "__main__":
    """Test the adaptive quantization with GPT-2"""
    import time
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("\n" + "="*60)
    print("GPU-POOR: Advanced Adaptive Quantization Test")
    print("="*60)
    
    
    print("\n[1/4] Loading GPT-2 model...")
    model_name = "gpt2" 
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
  
    original_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original model size: {original_params / 1024**2:.2f} MB")
    
   
    print("\n[2/4] Preparing calibration data...")
    sample_text = "The quick brown fox jumps over the lazy dog. "
    inputs = tokenizer(sample_text, return_tensors="pt")
    sample_data = inputs["input_ids"]
    
    #test original model generation
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
    
    #adaptive quantization
    print("\n[4/4] Applying adaptive quantization...")
    print("-" * 40)
    

    quantized_model = make_it_work_adaptive(
        model, 
        sample_data=sample_data,
        memory_target=0.4  #target 40%
    )
    
    #old uniform quantization
    print("-" * 40)
    
   
    quantized_params = 0
    int8_layers = 0
    original_layers = 0
    
    for name, module in quantized_model.named_modules():
        if isinstance(module, QuantizedLinear):
            #INT8 weights take 1/4 the space of FP32
            quantized_params += module.weight_quantized.numel() * 1  
            if module.bias is not None:
                quantized_params += module.bias.numel() * 4  
            int8_layers += 1
        elif isinstance(module, QuantizedConv1D):
            quantized_params += module.quantized_linear.weight_quantized.numel() * 1
            if module.quantized_linear.bias is not None:
                quantized_params += module.quantized_linear.bias.numel() * 4
            int8_layers += 1
        elif isinstance(module, QuantizedEmbeddingINT8):   #<<< NEW
            #int8 weights (1 byte each) + per-row float32 scale
            quantized_params += module.weight_int8.numel() * 1
            quantized_params += module.scale.numel() * 4
            int8_layers += 1
        elif hasattr(module, 'weight'):
            #non-quantized layers
            quantized_params += module.weight.numel() * module.weight.element_size()
            if hasattr(module, 'bias') and module.bias is not None:
                quantized_params += module.bias.numel() * module.bias.element_size()
            original_layers += 1
    
    print(f"\nQuantized model size: {quantized_params / 1024**2:.2f} MB")
    print(f"Memory reduction: {(1 - quantized_params/original_params) * 100:.1f}%")
    

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
    
    #Interactive testing
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