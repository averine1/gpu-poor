"""Core optimization magic"""
import torch
import psutil
import gc
import time

# Try to import rich, but fall back if it fails
try:
    from rich.console import Console
    from rich.progress import track
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[WARNING] Rich library not installed. Install with: pip install rich")

class GPUPoor:
    @staticmethod
    def make_it_work(model, verbose=True):
        """
        Optimize any model to run on limited hardware
        """
        if verbose:
            if HAS_RICH:
                console.print("\n[bold yellow]GPU-Poor Optimization Starting...[/bold yellow]")
                ram_gb = psutil.virtual_memory().available / (1024**3)
                console.print(f"[cyan]Available RAM: {ram_gb:.2f}GB[/cyan]")
            else:
                print("\n[GPU-POOR] Optimization Starting...")
                ram_gb = psutil.virtual_memory().available / (1024**3)
                print(f"[INFO] Available RAM: {ram_gb:.2f}GB")
        
        # Optimization steps
        optimizations = [
            ("Converting to half precision", GPUPoor._convert_to_half),
            ("Enabling gradient checkpointing", GPUPoor._enable_checkpointing),
            ("Moving to CPU", GPUPoor._move_to_cpu),
            ("Cleaning memory", GPUPoor._clean_memory),
        ]
        
        if HAS_RICH and verbose:
            for desc, func in track(optimizations, description="Optimizing..."):
                func(model)
                time.sleep(0.5)
        else:
            for desc, func in optimizations:
                if verbose:
                    print(f"[STEP] {desc}...")
                func(model)
        
        if verbose:
            if HAS_RICH:
                console.print("[bold green]Model optimized! Ready to run on your potato![/bold green]\n")
            else:
                print("[SUCCESS] Model optimized! Ready to run on your potato!\n")
        
        return model
    
    @staticmethod
    def _convert_to_half(model):
        """Convert model to FP16"""
        if hasattr(model, 'half'):
            try:
                model.half()
            except:
                pass
    
    @staticmethod
    def _enable_checkpointing(model):
        """Enable gradient checkpointing if available"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    @staticmethod
    def _move_to_cpu(model):
        """Ensure model is on CPU"""
        if hasattr(model, 'to'):
            model.to('cpu')
    
    @staticmethod
    def _clean_memory(model):
        """Clean up memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def make_it_work(model, verbose=True):
    return GPUPoor.make_it_work(model, verbose)