"""Test text generation with different models"""
from gpu_poor import make_it_work
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def test_generation(model_name, prompt="Hello, I am"):
    """Test text generation with a model"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)
    
    try:
        # Load model and tokenizer
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Optimize with gpu-poor
        print("Optimizing with GPU-Poor...")
        model = make_it_work(model)
        
        # Generate text
        print(f"\nPrompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: '{response}'")
        print("Success!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

import torch

# Test different generation models
models_to_test = [
    "microsoft/DialoGPT-small",    # Conversational
    "gpt2",                         # Classic GPT-2
    "distilgpt2",                   # Smaller GPT-2
    "cerebras/btlm-3b-8k-base",    # If you want to try bigger
]

print("\n GPU-POOR TEXT GENERATION TEST\n")

for model_name in models_to_test:
    try:
        test_generation(model_name)
    except:
        print(f"Skipping {model_name} - not available")