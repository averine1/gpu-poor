"""Test gpu-poor with various popular models"""
from gpu_poor import make_it_work
import torch
import warnings
warnings.filterwarnings('ignore')

def test_model(name, loader_func):
    """Test a model with gpu-poor"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    try:
        print(f"[1/3] Loading {name}...")
        model = loader_func()
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters()) / 1_000_000
        print(f"     Model has {param_count:.1f}M parameters")
        
        print(f"[2/3] Applying GPU-Poor optimization...")
        model = make_it_work(model)
        
        print(f"[3/3] Success! {name} optimized and ready!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test different models
print("\n GPU-POOR MODEL COMPATIBILITY TEST\n")

results = []

# 1. Test BERT
from transformers import AutoModel
def load_bert():
    return AutoModel.from_pretrained("bert-base-uncased")

results.append(("BERT", test_model("BERT", load_bert)))

# 2. Test GPT-2
from transformers import GPT2Model
def load_gpt2():
    return GPT2Model.from_pretrained("gpt2")

results.append(("GPT-2", test_model("GPT-2", load_gpt2)))

# 3. Test DistilBERT (smaller, faster)
def load_distilbert():
    return AutoModel.from_pretrained("distilbert-base-uncased")

results.append(("DistilBERT", test_model("DistilBERT", load_distilbert)))

# Print summary
print("\n" + "="*60)
print("COMPATIBILITY SUMMARY")
print("="*60)
for model_name, success in results:
    status = "WORKS" if success else "FAILED"
    print(f"{model_name:<15} {status}")
print("="*60)