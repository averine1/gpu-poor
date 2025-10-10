"""GPU-Poor Core: Make AI work on potato computers"""
import torch
import torch.nn as nn

class GPUPoor:
    """Main optimization class"""
    
    @staticmethod
    def make_it_work(model, mode="auto", verbose=True, method="adaptive", sample_data=None, **kwargs):
        """
        Optimize any model to run on limited hardware

        Args:
            model: PyTorch model to optimize
            mode: 'basic' (FP16), 'quantized' (INT8), 'a3qer' (new!), or 'auto'
            method: Quantization method - 'a3qer', 'adaptive', or 'uniform'
            verbose: Print progress
            sample_data: Calibration data for A3QER
        """

        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        model_mb = (param_count * 4) / (1024 * 1024)
        
        if verbose:
            print(f"[GPU-Poor] Original model size: {model_mb:.1f} MB")
        
        # Auto mode selection
        if mode == "auto":
            if model_mb > 500:
                mode = "a3qer"  # Use A3QER for large models
            elif model_mb > 100:
                mode = "quantized"
            else:
                mode = "basic"
            
            if verbose:
                print(f"[GPU-Poor] Auto-selected mode: {mode}")
        
        # Apply optimizations based on mode
        if mode == "basic":
            # Just convert to FP16
            model = model.half()
            if verbose:
                print("[GPU-Poor] Converted to FP16")
                
        elif mode == "quantized":
            # Use adaptive quantization
            from .quantization import make_it_work_adaptive, quantize_model
            
            if method == "adaptive":
                model = make_it_work_adaptive(model, sample_data=sample_data, **kwargs)
            else:
                model = quantize_model(model)
            
            if verbose:
                print(f"[GPU-Poor] Applied {method} quantization")
                
        elif mode == "a3qer":
            # Use the new A3QER method
            try:
                from .hybrid_a3qer import make_it_work_hybrid
                model = make_it_work_hybrid(model, sample_inputs=sample_data, **kwargs)
                if verbose:
                    print("[GPU-Poor] Applied hybrid quantization")
            except ImportError:
                print("[GPU-Poor] Warning: A3QER not available, falling back to adaptive")
                from .quantization import make_it_work_adaptive
                model = make_it_work_adaptive(model, sample_data=sample_data, **kwargs)
        
        # Calculate final size
        if verbose:
            # Estimate quantized size
            quantized_mb = (param_count * 1) / (1024 * 1024) if mode in ["quantized", "a3qer"] else model_mb / 2
            print(f"[GPU-Poor] Optimized size: ~{quantized_mb:.1f} MB")
            print(f"[GPU-Poor] Memory saved: ~{(1 - quantized_mb/model_mb)*100:.0f}%")
        
        return model

# Module-level convenience function
def make_it_work(model, mode="auto", verbose=True, method="adaptive", sample_data=None, **kwargs):
    """The magic function that makes any model work on potato computers"""
    return GPUPoor.make_it_work(model, mode=mode, verbose=verbose, method=method, 
                                sample_data=sample_data, **kwargs)

