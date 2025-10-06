"""Test Question-Answering models"""
from gpu_poor import make_it_work
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

print("\n🏚️ GPU-POOR Q&A DEMO\n")

# Load model
print("Loading BERT for Question-Answering...")
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Optimize
print("Optimizing with GPU-Poor...")
model = make_it_work(model)

# Create pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Test it
context = """
GPU-Poor is a tool that makes AI models run on regular computers without expensive GPUs. 
It uses techniques like quantization and memory optimization to reduce model size by 75%.
Created by developers who cant afford GPUs, for developers who cant afford GPUs.
"""

questions = [
    "What is GPU-Poor?",
    "How much does it reduce model size?",
    "Who created GPU-Poor?",
]

print("\nContext:", context)
print("\nQ&A Demo:\n")

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']}")
    print(f"   (Confidence: {result['score']:.2%})\n")