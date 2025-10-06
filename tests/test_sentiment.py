"""Test sentiment analysis"""
from gpu_poor import make_it_work
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

print("\n🏚️ GPU-POOR SENTIMENT ANALYSIS\n")

# Load model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Optimize
print("Optimizing with GPU-Poor...")
model = make_it_work(model)

# Create pipeline
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test texts
texts = [
    "GPU-Poor is amazing! Now I can run AI on my laptop!",
    "I hate needing expensive GPUs for AI",
    "This tool is okay, nothing special",
    "Finally, AI for everyone! No more CUDA errors!",
    "My potato computer is now an AI powerhouse"
]

print("\nSentiment Analysis Results:\n")
for text in texts:
    result = sentiment(text)[0]
    print(f"Text: '{text}'")
    print(f"→ {result['label']} ({result['score']:.2%})\n")