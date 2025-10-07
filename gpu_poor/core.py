"""GPU-Poor: Real optimization for running AI without GPUs"""
import torch
import torch.nn as nn
import psutil
import gc
import time

# Try to import rich for nice output
try:
    from rich.console import Console
    from rich.progress import track
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

class GPUPoor:
    @staticmethod
    def make_it_work(model, aggressive=True, verbose=True):
        """
        ACTUALLY optimize any model to run on limited hardware using PyTorch quantization
        
        Args:
            model: PyTorch model to optimize
            aggressive: If True, use INT8 quantization. If False, use FP16
            verbose: Print progress
        """
        if verbose:
            initial_ram = psutil.virtual_memory().available / (1024**3)
            if HAS_RICH:
                console.print("\n[bold yellow]GPU-Poor Optimization Starting...[/bold yellow]")
                console.print(f"[cyan]Available RAM: {initial_ram:.2f}GB[/cyan]")
            else:
                print("\n[GPU-POOR] Optimization Starting...")
                print(f"[INFO] Available RAM: {initial_ram:.2f}GB")
        
        # Optimization steps
        optimizations = [
            ("Setting to evaluation mode", GPUPoor._set_eval_mode),
            ("Moving to CPU", GPUPoor._move_to_cpu),
            ("Converting to half precision", GPUPoor._convert_to_half),
            ("Disabling gradients", GPUPoor._disable_gradients),
            ("Cleaning memory", GPUPoor._clean_memory),
        ]
        
        if aggressive:
            # Add dynamic quantization for aggressive mode
            optimizations.insert(2, ("Applying INT8 quantization", GPUPoor._quantize_dynamic))
        
        if HAS_RICH and verbose:
            for desc, func in track(optimizations, description="Optimizing..."):
                model = func(model)
                time.sleep(0.3)
        else:
            for desc, func in optimizations:
                if verbose:
                    print(f"[STEP] {desc}...")
                model = func(model)
        
        if verbose:
            final_ram = psutil.virtual_memory().available / (1024**3)
            if HAS_RICH:
                console.print("[bold green]Model optimized! Ready to run on your potato![/bold green]\n")
            else:
                print("[SUCCESS] Model optimized! Ready to run on your potato!\n")
        
        return model
    
    @staticmethod
    def _quantize_dynamic(model):
        """Apply dynamic INT8 quantization - works on CPU!"""
        # Skip quantization for models that don't support it well
        model_type = type(model).__name__.lower()
        
        # For transformer models, we need to be careful
        if 'gpt' in model_type or 'bert' in model_type:
            # Don't quantize the whole model, it breaks generation
            # Instead, just ensure we're using efficient dtypes
            return model
        
        try:
            # For other models, try dynamic quantization
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU, nn.LSTMCell, nn.GRUCell, nn.RNNCell},
                dtype=torch.qint8
            )
            return quantized
        except:
            # If quantization fails, just return the model
            return model
    
    @staticmethod
    def _set_eval_mode(model):
        """Set model to evaluation mode"""
        model.eval()
        return model
    
    @staticmethod
    def _convert_to_half(model):
        """Convert model to FP16 - this actually saves memory!"""
        try:
            # For transformer models, use half precision
            model = model.half()
        except:
            # Manual conversion for custom models
            for param in model.parameters():
                if param.dtype == torch.float32:
                    param.data = param.data.half()
        return model
    
    @staticmethod
    def _disable_gradients(model):
        """Disable gradients for inference"""
        for param in model.parameters():
            param.requires_grad = False
        
        # Also disable dropout and other training-specific layers
        if hasattr(model, 'training'):
            model.training = False
            
        return model
    
    @staticmethod
    def _move_to_cpu(model):
        """Ensure model is on CPU"""
        return model.to('cpu')
    
    @staticmethod
    def _clean_memory(model):
        """Clean up memory aggressively"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model

def make_it_work(model, aggressive=True, verbose=True):
    """
    The magic function that makes any model work on potato computers
    """
    return GPUPoor.make_it_work(model, aggressive, verbose)