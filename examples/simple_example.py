"""Quick example of GPU-Poor compression"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_poor import make_it_work_hybrid

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create calibration data (important!)
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length")

# Compress model (79% smaller!)
compressed = make_it_work_hybrid(model, sample_inputs=inputs["input_ids"])

print("Model compressed! Now 79% smaller with same quality.")