"""Universal model adapter for gpu-poor
Automatically detects and optimizes any Transformers model
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import re

class UniversalModelAdapter:
    """
    Automatically adapts to any Hugging Face Transformers model
    No manual configuration needed!
    """
    
    # Pattern matching for different architectures
    PATTERNS = {
        'attention': [
            r'.*attention.*', r'.*attn.*', r'.*self_attn.*',
            r'.*cross_attn.*', r'.*query.*', r'.*key.*', r'.*value.*'
        ],
        'mlp': [
            r'.*mlp.*', r'.*fc\d*$', r'.*dense.*', r'.*feedforward.*',
            r'.*intermediate.*', r'.*output\.dense.*'
        ],
        'embedding': [
            r'.*embed.*', r'.*wte.*', r'.*wpe.*', r'.*word_embeddings.*',
            r'.*position_embeddings.*', r'.*token_type_embeddings.*'
        ],
        'normalization': [
            r'.*norm.*', r'.*ln.*', r'.*layernorm.*', r'.*layer_norm.*'
        ],
        'head': [
            r'.*head.*', r'.*classifier.*', r'.*pooler.*', r'.*pred.*',
            r'.*lm_head.*', r'.*cls.*'
        ]
    }
    
    def __init__(self, model):
        self.model = model
        self.layers = {}  # Initialize layers first!
        self.model_info = self._analyze_model()
        
    def _analyze_model(self) -> Dict[str, Any]:
        """Automatically analyze model architecture"""
        info = {
            'type': self._detect_model_type(),
            'layers': self._map_layers(),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'architecture': self._detect_architecture()
        }
        
        print(f"[Adapter] Detected model: {info['type']}")
        print(f"[Adapter] Architecture: {info['architecture']}")
        print(f"[Adapter] Found {len(info['layers'])} quantizable layers")
        
        return info
    
    def _detect_model_type(self) -> str:
        """Detect model type from class name or config"""
        model_class = self.model.__class__.__name__.lower()
        
        # Check common model types
        if 'gpt' in model_class:
            return 'gpt'
        elif 'bert' in model_class:
            return 'bert'
        elif 't5' in model_class:
            return 't5'
        elif 'llama' in model_class:
            return 'llama'
        elif 'opt' in model_class:
            return 'opt'
        elif 'bloom' in model_class:
            return 'bloom'
        elif 'falcon' in model_class:
            return 'falcon'
        elif 'mistral' in model_class:
            return 'mistral'
        elif 'qwen' in model_class:
            return 'qwen'
        else:
            # Try to detect from config
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'model_type'):
                    return config.model_type
            return 'unknown'
    
    def _detect_architecture(self) -> str:
        """Detect if encoder, decoder, or encoder-decoder"""
        has_encoder = any('encoder' in name for name, _ in self.model.named_modules())
        has_decoder = any('decoder' in name for name, _ in self.model.named_modules())
        
        if has_encoder and has_decoder:
            return 'encoder-decoder'
        elif has_decoder or 'gpt' in self._detect_model_type():
            return 'decoder-only'
        else:
            return 'encoder-only'
    
    def _map_layers(self) -> Dict[str, Dict[str, Any]]:
        """Map all layers and categorize them"""
        layers = {}
        
        for name, module in self.model.named_modules():
            # Skip if not quantizable
            if not self._is_quantizable(module):
                continue
            
            # Categorize layer
            category = self._categorize_layer(name)
            
            # Store layer info temporarily
            layers[name] = {
                'module': module,
                'category': category,
                'size': sum(p.numel() for p in module.parameters()),
                'depth': self._get_layer_depth(name)
            }
        
        # Store layers before computing criticality
        self.layers = layers
        
        # Now compute criticality with layers available
        for name in list(layers.keys()):
            layers[name]['criticality'] = self._assess_criticality(name, layers[name]['category'])
        
        return layers
    
    def _is_quantizable(self, module) -> bool:
        """Check if module can be quantized"""
        # Must have weight parameter
        if not hasattr(module, 'weight'):
            return False
        
        # Must be Linear-like layer
        if not isinstance(module, (nn.Linear, nn.Conv1d)):
            # Check for custom linear layers (like Conv1D in GPT-2)
            if module.__class__.__name__ not in ['Conv1D', 'Linear']:
                return False
        
        # Skip if too small (not worth quantizing)
        if hasattr(module, 'weight'):
            if module.weight.numel() < 1000:
                return False
        
        return True
    
    def _categorize_layer(self, name: str) -> str:
        """Categorize layer type using pattern matching"""
        name_lower = name.lower()
        
        for category, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, name_lower):
                    return category
        
        return 'other'
    
    def _assess_criticality(self, name: str, category: str) -> float:
        """
        Assess layer criticality (0=not critical, 1=very critical)
        Based on empirical findings from research
        """
        criticality = 0.5  # Default
        
        # Get depth information
        depth = self._get_layer_depth(name)
        total_depth = self._get_total_depth()
        relative_position = depth / max(total_depth, 1)
        
        # First and last 10% of layers are critical
        if relative_position < 0.1 or relative_position > 0.9:
            criticality = 0.9
        
        # Embeddings are always critical
        if category == 'embedding':
            criticality = 1.0
        
        # Output heads are critical
        elif category == 'head':
            criticality = 0.95
        
        # Attention in first/last layers is critical
        elif category == 'attention':
            if relative_position < 0.15 or relative_position > 0.85:
                criticality = 0.85
            else:
                criticality = 0.4
        
        # MLPs in middle layers are less critical
        elif category == 'mlp':
            if 0.2 < relative_position < 0.8:
                criticality = 0.3
            else:
                criticality = 0.6
        
        # Normalization layers should not be quantized
        elif category == 'normalization':
            criticality = 1.0
        
        return criticality
    
    def _get_layer_depth(self, name: str) -> int:
        """Extract layer depth from name"""
        # Look for patterns like layer.0, h.0, blocks.0, etc.
        matches = re.findall(r'(?:layer|h|block|blocks|layers)\.(\d+)', name)
        if matches:
            return int(matches[0])
        return 0
    
    def _get_total_depth(self) -> int:
        """Get total number of layers"""
        if not self.layers:
            return 1
        depths = [self._get_layer_depth(name) for name in self.layers.keys()]
        return max(depths) if depths else 1
    
    def get_quantization_config(self, target_reduction: float = 0.5) -> Dict[str, Dict]:
        """
        Generate optimal quantization configuration
        
        Args:
            target_reduction: Target memory reduction (0.5 = 50%)
        
        Returns:
            Configuration dict for each layer
        """
        configs = {}
        
        # Sort layers by criticality (least critical first)
        sorted_layers = sorted(
            self.layers.items(),
            key=lambda x: (x[1]['criticality'], -x[1]['size'])
        )
        
        # Calculate how much we need to quantize
        total_params = self.model_info['total_params']
        target_quantized = total_params * target_reduction
        current_quantized = 0
        
        for name, info in sorted_layers:
            if current_quantized >= target_quantized:
                # Target reached, keep rest in original precision
                configs[name] = {
                    'quantize': False,
                    'reason': 'target_reached'
                }
            elif info['criticality'] > 0.9:
                # Too critical to quantize
                configs[name] = {
                    'quantize': False,
                    'reason': 'critical_layer'
                }
            else:
                # Quantize this layer
                configs[name] = {
                    'quantize': True,
                    'bits': 8,  # Use INT8 for quality
                    'smooth': info['category'] in ['attention', 'mlp'],
                    'category': info['category'],
                    'criticality': info['criticality']
                }
                current_quantized += info['size'] * 0.75  # INT8 saves 75%
        
        # Print summary
        quantized = sum(1 for c in configs.values() if c.get('quantize'))
        kept = sum(1 for c in configs.values() if not c.get('quantize'))
        
        print(f"\n[Adapter] Quantization plan:")
        print(f"  Quantizing {quantized} layers")
        print(f"  Keeping {kept} layers original")
        print(f"  Estimated reduction: {current_quantized/total_params*100:.1f}%")
        
        return configs


def auto_quantize(model, calibration_data=None, target_reduction=0.5):
    """
    Automatically quantize any model with zero configuration
    
    Args:
        model: Any Hugging Face Transformers model
        calibration_data: Optional calibration data
        target_reduction: Target memory reduction
    
    Returns:
        Quantized model
    """
    print(f"\n{'='*60}")
    print("AUTO-QUANTIZATION")
    print(f"{'='*60}")
    
    # Analyze model
    adapter = UniversalModelAdapter(model)
    
    # Get optimal configuration
    config = adapter.get_quantization_config(target_reduction)
    
    # Apply quantization
    from .hybrid_a3qer import HybridA3QER
    
    quantizer = HybridA3QER()
    
    # If no calibration data, create some basic samples
    if calibration_data is None:
        print("[Warning] No calibration data provided, using random data")
        # Create dummy data matching model input shape
        if hasattr(model, 'dummy_inputs'):
            calibration_data = model.dummy_inputs['input_ids']
        else:
            calibration_data = torch.randint(0, 1000, (4, 128))
    
    # Collect calibration stats if needed
    if any(c.get('smooth') for c in config.values()):
        quantizer.collect_calibration_data(model, calibration_data)
    
    # Apply quantization based on config
    from .quantization import QuantizedLinear, QuantizedConv1D
    
    for name, layer_config in config.items():
        if not layer_config.get('quantize'):
            continue
        
        # Get module and parent
        module = adapter.layers[name]['module']
        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
        child_name = name.split('.')[-1]
        
        parent = model
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
        
        # Quantize
        try:
            if module.__class__.__name__ == 'Conv1D':
                temp = nn.Linear(module.weight.shape[0], module.weight.shape[1])
                temp.weight.data = module.weight.data.t()
                if hasattr(module, 'bias'):
                    temp.bias = module.bias
                quantized = QuantizedLinear.from_float(temp, bits=8)
                quantized = QuantizedConv1D(quantized)
            else:
                quantized = QuantizedLinear.from_float(module, bits=8)
            
            setattr(parent, child_name, quantized)
            
        except Exception as e:
            print(f"[Error] Failed to quantize {name}: {e}")
    
    print(f"\n[Success] Model quantized!")
    return model

