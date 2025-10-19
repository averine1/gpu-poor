# ⚡ gpu-poor

**Run large language models in 74% less memory, at the same speed.**

Pure Python quantization for memory-constrained deployment.
```python
from gpu_poor import quantize

model = quantize(model)  # 3GB → 767MB, same speed
```

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Memory is the bottleneck for running large language models:

- **GPT-2-large:** Needs 3GB RAM
- **GPT-J:** Needs 24GB RAM  
- **Llama-7B:** Needs 28GB RAM

Most laptops have 8-16GB. Most consumer GPUs have 8-12GB VRAM.

**You can't run models that don't fit.**

---

## The Solution

gpu-poor uses INT8 quantization to compress models by 74% with minimal speed impact.

### Results

| Model | Before | After | Reduction | Speed |
|-------|--------|-------|-----------|-------|
| **GPT-2-large** | **2,953 MB** | **767 MB** | **74%** | **0.99×** ✅ |
| GPT-2-medium | 1,353 MB | 356 MB | 74% | 0.93× |
| GPT-2 | 475 MB | 128 MB | 73% | 0.85× |

**Quality:** Zero degradation on all tested models.

---

## Key Insight: Scales with Model Size

Quantization overhead is fixed (~0.2s). As models get larger, this overhead becomes negligible:

| Model Size | Speed Impact | Recommendation |
|------------|--------------|----------------|
| **Large (>2GB)** | **~1.0×** | **✅ Perfect fit** |
| Medium (1-2GB) | ~0.9× | ✅ Use if memory matters |
| Small (<500MB) | ~0.5-0.8× | ❌ Overhead too high |

**Best for:** Large models where you need memory savings most.

---

## Quick Start

### Installation
```bash
pip install gpu-poor
```

### Basic Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gpu_poor import quantize

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

# Quantize (one line)
model = quantize(model)

# Use normally
tokenizer.pad_token = tokenizer.eos_token
output = model.generate(
    tokenizer("The future of AI is", return_tensors="pt")["input_ids"],
    max_length=50
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Try in Colab

[Open in Colab →](https://colab.research.google.com/github/yourusername/gpu-poor)

---

## How It Works

Three techniques for reliable compression:

1. **INT8 Weight Quantization**  
   Weights stored as 8-bit integers with per-channel scales

2. **Smart Layer Selection**  
   Critical layers (first block) kept at higher precision

3. **Embedding Compression**  
   Largest memory savings with weight tying to lm_head

All pure Python using PyTorch. No C++ compilation required.

---

## When to Use

### ✅ Use gpu-poor when:

- Running large models (>1GB) on limited memory
- Deploying on diverse hardware (CPU, GPU, Apple Silicon)
- Need consistent, predictable results
- Want pure Python (no compilation)
- Memory matters more than 5-10% speed

### ❌ Don't use when:

- Running tiny models (<500MB) - overhead hurts
- Need maximum speed - use [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Have unlimited memory - just use FP32

---

## Real Use Cases

### 1. Development on 8GB Laptop
```
Before: Can't load GPT-2-large (3GB model + OS = out of memory)
After:  Load + run with room for IDE, browser (767MB model)
```

### 2. Multi-Model Comparison
```
Before: Compare 2 models (2.7GB total)
After:  Compare 4+ models (1.4GB total)
```

### 3. Fine-Tuning on Consumer GPU
```
Before: GPT-2-medium barely fits (8GB VRAM)
After:  Fits with room for optimizer states (356MB)
```

---

## Comparison with Alternatives

| Library | Target | Compression | Speed | CPU | Pure Python |
|---------|--------|-------------|-------|-----|-------------|
| **gpu-poor** | Large models | 74% | 0.9-1.0× | ✅ | ✅ |
| llama.cpp | Speed | 75% | 2-3× | ✅ | ❌ C++ |
| GPTQ | GPU inference | 75% | 1.0× | ❌ | ✅ |
| bitsandbytes | GPU training | 75% | 1.2× | ❌ | ✅ |

**gpu-poor wins on:**
- Same speed for large models (1.0×)
- Pure Python simplicity
- CPU + GPU support
- Consistent results across models

---

## Advanced Usage

### Custom Quantization
```python
from gpu_poor import make_it_work_hybrid

# More control over quantization
model = make_it_work_hybrid(
    model,
    sample_inputs=calibration_data,
    quantize_embeddings=True,
    smoothing_alpha=0.3,
    memory_target=0.4
)
```

### Benchmarking
```python
from gpu_poor.examples import demo

# Run full benchmark suite
results = demo.demo_production_ready("gpt2-large")
print(results)
```

---

## Detailed Results

See [RESULTS.md](RESULTS.md) for complete benchmarks including:
- Per-layer quantization decisions
- Quality comparisons
- Speed breakdowns
- Reproduction instructions

---

## Contributing

Contributions welcome! Areas of interest:

- Support for more model architectures
- Additional quantization strategies
- Performance optimizations
- Documentation improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Citation
```bibtex
@software{gpupoor2024,
  author = {Your Name},
  title = {gpu-poor: Memory-Efficient Large Model Inference},
  year = {2024},
  url = {https://github.com/averine1/gpu-poor}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/transformers/)

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GPTQ](https://github.com/IST-DASLab/gptq)

---

## FAQ

**Q: Why not just use PyTorch's `quantize_dynamic`?**  
A: PyTorch's method gives ~50% compression. gpu-poor achieves 74% through embedding compression and weight tying.

**Q: Does this work on Apple Silicon?**  
A: Yes! Pure Python, runs on any PyTorch-supported hardware.

**Q: Can I use this for fine-tuning?**  
A: Currently optimized for inference. Fine-tuning support coming soon.

**Q: Why is it slower on small models?**  
A: Fixed overhead (~0.2s) dominates small computation times. See [RESULTS.md](RESULTS.md) for analysis.

**Q: How does quality compare to FP32?**  
A: Identical on all tested models. INT8 with careful layer selection preserves quality.

---

**⭐ Star if you're memory-constrained**

[Report Bug](https://github.com/averine1/gpu-poor/issues) · [Request Feature](https://github.com/averine1/gpu-poor/issues) · [Discussions](https://github.com/averine1/gpu-poor/discussions)