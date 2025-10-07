"""Complete demo of GPU-Poor capabilities"""
from gpu_poor import make_it_work
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print(" "*20 + "GPU-POOR DEMO")
print(" "*15 + "AI Without GPUs!")
print("="*60)

# Text Generation
print("\n[1/3] TEXT GENERATION with GPT-2")
print("-"*40)
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = make_it_work(model, verbose=False)

prompt = "AI should be accessible to"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=20, temperature=0.8)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"GPT-2: {result}")

# Question Answering
print("\n[2/3] QUESTION ANSWERING with BERT")
print("-"*40)
from transformers import AutoModelForQuestionAnswering
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_model = make_it_work(qa_model, verbose=False)
qa = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

context = "GPU-Poor makes AI accessible by reducing model size by 75 percent"
question = "How much does GPU-Poor reduce model size?"
answer = qa(question=question, context=context)
print(f"Context: {context}")
print(f"Question: {question}")
print(f"Answer: {answer['answer']} (Confidence: {answer['score']:.1%})")

# Sentiment Analysis
print("\n[3/3] SENTIMENT ANALYSIS with DistilBERT")
print("-"*40)
from transformers import AutoModelForSequenceClassification
sent_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sent_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sent_model = make_it_work(sent_model, verbose=False)
sentiment = pipeline("sentiment-analysis", model=sent_model, tokenizer=sent_tokenizer)

text = "GPU-Poor is amazing! Finally I can run AI!"
result = sentiment(text)[0]
print(f"Text: {text}")
print(f"Sentiment: {result['label']} ({result['score']:.1%})")

print("\n" + "="*60)
print("ALL MODELS RUNNING ON CPU - NO GPU NEEDED!")
print(f"Sentiment: {result['label']} ({result['score']:.1%})")

print("\n" + "="*60)
print("ALL MOEDELS RUNNING ON CPU - NO GPU NEEDED!")
print("="*60 + "\n")
