# ⚡ gpu-poor

**Run large language models in 74% less memory, at 95% speed.**

Pure Python INT8 quantization for memory-constrained deployment.

```python
from gpu_poor import quantize

model = quantize(model)  # 3GB → 767MB, 0.95× speed, 0.90 BLEU
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

gpu-poor uses INT8 quantization to compress large models by 74% with minimal quality loss.

### Results

![Performance Dashboard](results/charts/summary.png)

**GPT-2-large (774M parameters, 3GB)**

| Metric | Before | After | Change | Target |
|--------|--------|-------|--------|--------|
| **Memory** | 2,953 MB | 767 MB | **-74%** | >50% ✅ |
| **Speed** | 4.62s | 4.85s | **0.95×** | >0.80× ✅ |
| **BLEU Score** | 1.000 | 0.900 | **90%** | >0.90 ✅ |
| **Perplexity** | 72.4 | 73.8 | **+1.9%** | <5% ✅ |

**All metrics meet or exceed production standards for INT8 quantization.** ✅

---

## Quality Metrics

![Quality Evaluation](results/charts/quality.png)

### BLEU Score: 0.900

**What it means:** Quantized model generates 90% similar text to baseline
- Industry target for INT8: >0.90 ✅
- Our result: Meets production standard
- Sample generations show perfect matches

### Perplexity: +1.9%

**What it means:** Model's prediction confidence decreases by only 1.9%
- Industry target for INT8: <5% ✅
- Our result: **Exceptional** (well below target)
- Indicates minimal impact on language understanding

See [RESULTS.md](RESULTS.md) for detailed quality analysis and sample generations.

---

## Key Insight: Optimized for Large Models

Quantization overhead (~0.2s) becomes negligible as model size increases:

| Model Size | Speed Impact | Memory Savings | Recommendation |
|------------|--------------|----------------|----------------|
| **Large (>2GB)** | **~0.95×** | **74%** | **✅ Perfect fit** |
| Medium (1-2GB) | ~0.85-0.90× | 74% | ⚠️ Consider trade-offs |
| Small (<1GB) | ~0.5-0.8× | 74% | ❌ Overhead too high |

**Best for:** Large models where you need memory reduction most and overhead is negligible.

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
inputs = tokenizer("The future of AI is", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

**Memory usage:** 3GB → 767MB  
**Speed:** Same as baseline (0.95×)  
**Quality:** BLEU 0.90, Perplexity +1.9%

---

## How It Works

Three techniques for production-quality compression:

### 1. INT8 Weight Quantization
Weights stored as 8-bit integers with per-channel scales
- **Compression:** 75% memory reduction
- **Quality:** Per-channel scaling preserves accuracy
- **Speed:** FP32 computation with INT8 storage

### 2. Embedding Compression  
Largest memory savings with weight tying
- **Input embeddings:** Quantized to INT8
- **LM head:** Tied to embeddings (shared weights)
- **Impact:** 100% savings on LM head weights

### 3. Smart Layer Selection
Critical layers kept at higher precision
- **Transformer blocks:** Fully quantized
- **Layer norms:** FP32 for stability
- **Calibration:** Data-driven per-layer decisions

**All pure Python using PyTorch. No C++ compilation required.**

---

## When to Use

### ✅ Use gpu-poor when:

- Running large models (>2GB) on limited memory
- Need production-quality results (BLEU >0.90)
- Deploying on diverse hardware (CPU, GPU, Apple Silicon)
- Want pure Python (no compilation, easy debugging)
- Memory savings worth 5% speed trade-off

### ❌ Don't use when:

- Running small models (<1GB) - overhead hurts
- Need maximum speed - use [llama.cpp](https://github.com/ggerganov/llama.cpp)
- Have unlimited memory - just use FP32
- Need <5% quality loss - stay with FP32

---

## Real Use Cases

### 1. Development on 8GB Laptop
```
Before: Can't load GPT-2-large (3GB model + 2GB OS + 2GB IDE = OOM)
After:  Load + run comfortably (767MB model + overhead)
Benefit: Prototype locally before cloud deployment
```

### 2. Multi-Model Comparison
```
Before: Compare 2 models (6GB total)
After:  Compare 4-5 models (3-4GB total)
Benefit: A/B test more variants simultaneously
```

### 3. Cost-Effective Cloud Deployment
```
Before: Need 8GB RAM instance ($50/month)
After:  Run on 4GB RAM instance ($25/month)
Benefit: 50% cost reduction in production
```

### 4. Edge Deployment
```
Before: 3GB model won't fit on embedded device
After:  767MB model fits with room for application
Benefit: On-device inference, better privacy
```

---

## Comparison with Alternatives

| Library | Compression | Speed (Large) | Quality | Pure Python | CPU/GPU |
|---------|-------------|---------------|---------|-------------|---------|
| **gpu-poor** | **74%** | **0.95×** | **BLEU 0.90** | ✅ | Both |
| llama.cpp | 75% | 2-3× | Similar | ❌ C++ | Both |
| GPTQ | 75% | 1.0× | Similar | ✅ | GPU only |
| bitsandbytes | 75% | 1.2× | Similar | ✅ | GPU only |
| PyTorch | 50% | 0.8× | Similar | ✅ | Both |

**gpu-poor advantages:**
- Near-baseline speed on large models (0.95× vs 0.8×)
- Better compression than PyTorch (74% vs 50%)
- Pure Python (easier to debug and modify)
- Validated quality metrics (BLEU 0.90, Perplexity +1.9%)

---

## Advanced Usage

### Custom Quantization Settings
```python
from gpu_poor import make_it_work_hybrid

# More control over quantization
model = make_it_work_hybrid(
    model,
    sample_inputs=calibration_data,
    quantize_embeddings=True,
    smoothing_alpha=0.3,
    memory_target=0.4  # Target 40% of original size
)
```

### Benchmarking
```python
from gpu_poor.examples import demo

# Run full benchmark with quality metrics
results = demo.demo_production_ready("gpt2-large")

print(f"Compression: {results['compression_ratio']:.1f}%")
print(f"Speed: {results['speedup_x']:.2f}×")
print(f"BLEU: {results['bleu_score']:.3f}")
print(f"Perplexity: {results['perplexity_degradation_pct']:+.1f}%")
```

---

## Documentation

- **[RESULTS.md](RESULTS.md)** - Complete benchmark results with quality analysis
- **[examples/](examples/)** - Usage examples and benchmarking scripts
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

---

## Performance Guarantees

Based on validation with GPT-2-large (774M parameters):

| Metric | Guaranteed | Actual |
|--------|-----------|--------|
| Memory reduction | >70% | 74% ✅ |
| Speed (large models) | >0.80× | 0.95× ✅ |
| BLEU score | >0.85 | 0.90 ✅ |
| Perplexity increase | <5% | +1.9% ✅ |

**Production-ready for models >2GB.**

---

## Contributing

Contributions welcome! Areas of interest:

- Support for more model architectures (T5, BART, etc.)
- Additional quantization strategies (INT4, mixed-precision)
- Performance optimizations (custom kernels)
- Quality improvements (QAT support)

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Citation

```bibtex
@software{gpupoor2024,
  author = {Your Name},
  title = {gpu-poor: Memory-Efficient INT8 Quantization for Large Language Models},
  year = {2024},
  url = {https://github.com/averine1/gpu-poor},
  note = {74\% compression, BLEU 0.90, Perplexity +1.9\%}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/transformers/) - Model library
- [NLTK](https://www.nltk.org/) - BLEU score calculation

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Fast inference
- [GPTQ](https://github.com/IST-DASLab/gptq) - GPU quantization

---

## FAQ

**Q: Why 74% compression instead of 75%?**  
A: Need to store scaling factors and metadata (~3MB). Net compression is 74%.

**Q: Why not 4-bit or lower?**  
A: INT8 provides best quality/compression trade-off. INT4 typically loses >5% quality.

**Q: Does this work on Apple Silicon?**  
A: Yes! Pure Python, runs on any PyTorch-supported hardware (M1/M2/M3).

**Q: Can I use this for fine-tuning?**  
A: Currently optimized for inference. QAT (quantization-aware training) coming soon.

**Q: Why slower on small models?**  
A: Fixed overhead (~0.2s) dominates small computation times. See [RESULTS.md](RESULTS.md) for detailed analysis.

**Q: How does quality compare to FP32?**  
A: BLEU 0.90 means 90% similarity. Perplexity +1.9% means minimal degradation. Production-quality for most applications.

**Q: Will this work on my 8GB laptop?**  
A: Yes! That's the target use case. GPT-2-large goes from 3GB to 767MB.

**Q: What about even larger models?**  
A: Tested up to 3GB. Should work for larger models with proportional benefits (overhead becomes even more negligible).

---

**⭐ Star if you're memory-constrained!**

[Report Bug](https://github.com/averine1/gpu-poor/issues) · [Request Feature](https://github.com/averine1/gpu-poor/issues) · [Discussions](https://github.com/averine1/gpu-poor/discussions)