"""Better chat demo with GPT-2"""
from gpu_poor import make_it_work
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" " * 20 + "GPU-POOR DEMO")
print("=" * 60)
print("\nLoading GPT-2 for text completion (not chat)...\n")

# Load GPT-2 instead of DialoGPT
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Applying GPU-Poor optimization...")
model = make_it_work(model, verbose=False)

print("\nReady! This will complete your text, not chat.\n")
print("=" * 60)
print("Type 'quit' to exit")
print("=" * 60)

print("\nText Completion Demo:\n")

while True:
    user_input = input("\nStart a sentence: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Goodbye!")
        break
    
    # Generate completion
    inputs = tokenizer.encode(user_input, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=len(inputs[0]) + 30,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"GPT-2 completes: {response}")
