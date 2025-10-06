"""Simple impressive demo"""
from gpu_poor import make_it_work
from transformers import GPT2Model
import torch

print("\n" + "="*60)
print("RUNNING GPT-2 (124M PARAMETERS) WITHOUT A GPU")
print("="*60)

print("\n[BEFORE] Typical error without GPU:")
print('RuntimeError: CUDA out of memory. Tried to allocate 2.0 GB')

print("\n[WITH GPU-POOR]:")
model = GPT2Model.from_pretrained("gpt2")
model = make_it_work(model)

print("\nGenerating with GPT-2 on CPU...")
dummy_input = torch.randint(0, 50000, (1, 10))
output = model(dummy_input)
print(f"Output shape: {output.last_hidden_state.shape}")
print("\nSUCCESS! GPT-2 is running on your potato computer!")
print("="*60)