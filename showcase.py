"""GPU-Poor Showcase - Working version for Windows/CPU"""
import time
import psutil
from gpu_poor import make_it_work
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("GPU-POOR SHOWCASE")
print("="*70)
print("\nDEMONSTRATION: Running GPT-2 (124M parameters) without a GPU\n")

# Show the problem
print("[THE PROBLEM]")
print("Most people see this error:")
print("  RuntimeError: CUDA out of memory")
print("")

# Show the solution
print("[THE SOLUTION]")
print("With gpu-poor, just one line of code:")
print("  model = make_it_work(model)")
print("")

# Measure memory before
mem_before_gb = psutil.virtual_memory().used / (1024**3)

# Actually run it
print("[LIVE DEMO]")
print("Loading GPT-2 (124 million parameters)...")
start = time.time()

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Measure memory with original model
mem_with_model_gb = psutil.virtual_memory().used / (1024**3)
original_size = mem_with_model_gb - mem_before_gb

print("Optimizing with gpu-poor...")
model = make_it_work(model, verbose=False)

# Measure memory after optimization
mem_optimized_gb = psutil.virtual_memory().used / (1024**3)
optimized_size = mem_optimized_gb - mem_before_gb

load_time = time.time() - start
print(f"Model ready in {load_time:.1f} seconds!")
print("")

# Show memory savings
savings = (1 - optimized_size/original_size) * 100 if original_size > 0 else 0
print(f"[MEMORY STATS]")
print(f"Original model size: {original_size:.2f}GB")
print(f"Optimized model size: {optimized_size:.2f}GB")
print(f"Memory saved: {savings:.1f}%")
print("")

# Generate text
print("[GENERATING TEXT ON CPU]")
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs['input_ids'],
    max_length=30,
    do_sample=True,
    temperature=0.8,
    pad_token_id=tokenizer.pad_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: '{prompt}'")
print(f"GPT-2:  '{response}'")
print("")

print("="*70)
print("SUCCESS! GPT-2 is running on your CPU - No GPU needed!")
print("GitHub: https://github.com/averine1/gpu-poor")
print("="*70 + "\n")
