# 🏚️ gpu-poor

> Run AI models on your potato computer. No GPU required.

## 🎯 Why gpu-poor?

- Tired of "CUDA out of memory" errors?
- Don't have $2000 for a GPU?
- Just want to run AI on your laptop?

This is for you.

## Tested Models

| Model | Parameters | Status | RAM Usage |
|-------|------------|--------|-----------|
| BERT | 109.5M | ✅ WORKS | ~800MB |
| GPT-2 | 124.4M | ✅ WORKS | ~1GB |
| DistilBERT | 66.4M | ✅ WORKS | ~500MB |
| DialoGPT | 117M | ✅ WORKS | ~800MB |

All models tested on a regular laptop with 8GB RAM. No GPU required!

## 🚀 Quick Start
```python
from transformers import AutoModel
from gpu_poor import make_it_work

# Load any model
model = AutoModel.from_pretrained("bert-base-uncased")

# Magic happens here
model = make_it_work(model)

# Now it runs on your laptop!