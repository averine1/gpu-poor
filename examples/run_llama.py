"""Example: Run a language model without GPU"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
sys.path.append('..')

from gpu_poor import make_it_work

def main():
    print("[INFO] Testing GPU-Poor...")
    
    # Create a simple test
    model_name = "microsoft/DialoGPT-small"
    
    try:
        print(f"[LOADING] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("[OPTIMIZING] Optimizing with GPU-Poor...")
        model = make_it_work(model)
        
        print("[SUCCESS] Model optimized.")
        
        # Test generation
        text = "Hello, how are you?"
        inputs = tokenizer.encode(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[INPUT] {text}")
        print(f"[RESPONSE] {response}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()