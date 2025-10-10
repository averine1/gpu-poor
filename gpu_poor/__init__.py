"""GPU-Poor: Run AI models without a GPU"""
from .core import make_it_work, GPUPoor
from .quantization import quantize_model, make_it_work_adaptive
from .hybrid_a3qer import make_it_work_hybrid
from .universal_adapter import auto_quantize

# Fast inference optimization
try:
    from .fast_inference import OptimizedModel, optimize_for_inference
    __all__ = [
        "make_it_work", 
        "GPUPoor", 
        "quantize_model", 
        "make_it_work_adaptive",
        "make_it_work_hybrid",
        "auto_quantize",
        "OptimizedModel",
        "optimize_for_inference"
    ]
except ImportError:
    __all__ = [
        "make_it_work", 
        "GPUPoor", 
        "quantize_model", 
        "make_it_work_adaptive",
        "make_it_work_hybrid",
        "auto_quantize"
    ]

__version__ = "3.0.0"  