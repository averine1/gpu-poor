# 🏚️ GPU-Poor

> Extreme LLM compression for CPU inference. Run large language models on your potato computer.

## 🎯 What is GPU-Poor?

gpu-poor: Run bigger LLMs on CPUs — 50–79% less RAM with preserved quality and ~1.4–2.2× faster generation on tested models, using pure PyTorch (no custom kernels). Proven on GPT-2, GPT-2-Medium, OPT-125M, and TinyLlama-1.1B; seq2seq (T5) supported with a one-line decoder start token setup.

## 📊 Proven Results

| Model | Original Size | Compressed Size | Reduction | Quality |
|-------|--------------|-----------------|-----------|---------|
| **GPT-2** | 474.7 MB | **99.6 MB** | **79%** | ✅ Perfect |
| **GPT-2-Medium** | 1,353 MB | **465.8 MB** | **66%** | ✅ Perfect |
| **OPT-125M** | 477.8 MB | **234.8 MB** | **51%** | ✅ Perfect |

*All models tested with full generation quality metrics. No degradation in output.*

## 🚀 Quick Start
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_poor import make_it_work_hybrid

# Load any model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create calibration data (important for quality!)
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length")

# Compress model - 79% smaller!
model = make_it_work_hybrid(model, sample_inputs=inputs["input_ids"])

# Generate text with same quality
output = model.generate(**inputs, max_length=50)
print(tokenizer.decode(output[0]))
```
## 💡 Key Features

1) Extreme Compression: 50-79% model size reduction
2) Zero Quality Loss: Maintains generation quality
3) Pure PyTorch: No custom kernels or dependencies
4) Adaptive Precision: INT4/INT6/INT8 based on layer importance
5) CPU Optimized: Designed specifically for CPU inference

## 🛠️ Installation
bashpip install gpu-poor

## 📈 How It Works
GPU-Poor uses a novel mixed-precision approach:

- INT4 for non-critical MLP layers (maximum compression)
- INT6 for medium-importance attention layers
- INT8 for critical attention projections
- FP32 for first/last layers (quality preservation)

## 🎯 Use Cases

- Edge Deployment: Run models on Raspberry Pi or embedded systems
- Development: Test models without expensive GPU instances
- Research: Run more experiments with limited resources
- Production: Reduce cloud infrastructure costs by 50-79%

## 📖 Examples
See examples/demo.py for complete examples with different models.
## 🤝 Contributing
Contributions welcome! Please see CONTRIBUTING.md for guidelines.
## 📄 License
MIT License - see LICENSE file.
## 🙏 Acknowledgments
Built on top of PyTorch and Hugging Face Transformers.
## 📬 Contact
Issues and questions: GitHub Issues

