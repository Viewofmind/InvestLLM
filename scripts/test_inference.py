import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "models/sentiment/best_model"
LABELS = ["Negative", "Neutral", "Positive"]

def predict(text):
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    
    print(f"\nText: '{text}'")
    print(f"Sentiment: {LABELS[predicted_class.item()]}")
    print(f"Confidence: {confidence.item():.2%}")

# Test with some Indian market examples
examples = [
    "HDFC Bank reports 20% jump in net profit, beats estimates.",
    "Sensex crashes 800 points as inflation worries grip market.",
    "TCS announces dividend of Rs 15 per share.",
    "RBI keeps repo rate unchanged at 6.5%."
]

print("Loading model from local storage...")
for text in examples:
    predict(text)
