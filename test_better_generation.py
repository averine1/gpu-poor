"""Test generation with better prompts"""
from gpu_poor import make_it_work
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings('ignore')

print("\nGPU-POOR: Better Generation Test\n")
print("="*60)

# Test with GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Optimize
print("Optimizing GPT-2 with GPU-Poor...")
model = make_it_work(model, verbose=False)

# Better prompts
test_prompts = [
    "The weather today is",
    "Python is a programming language that",
    "The best thing about AI is",
    "To make a sandwich, first you need",
]

print("\nGenerating with better prompts:\n")

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=30,  # Shorter responses
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")

print("="*60)
print("SUCCESS! GPT-2 running on CPU with GPU-Poor!")
