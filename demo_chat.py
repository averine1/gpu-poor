"""Interactive chat demo for GPU-Poor"""
from gpu_poor import make_it_work
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print(" " * 20 + "GPU-POOR DEMO")
print("=" * 60)
print("\nRunning AI on your potato computer...\n")

# Load model
print("[1/3] Loading AI model (this usually needs a GPU)...")
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

print("[2/3] Applying GPU-Poor magic...")
model = make_it_work(model)

print("[3/3] Ready to chat!\n")
print("=" * 60)
print("Type 'quit' to exit")
print("=" * 60)

# Chat loop
chat_history_ids = None
print("\nChat with AI (running on CPU):\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Bot: Goodbye! Thanks for being gpu-poor with me!")
        break
    
    # Encode and generate
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Append to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    
    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        do_sample=True,
        top_k=50
    )
    
    # Decode and print only the new response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Bot: {response}\n")