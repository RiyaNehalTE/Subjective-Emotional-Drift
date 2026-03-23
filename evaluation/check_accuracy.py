from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

MODEL_PATH = "outputs_deberta/checkpoint-121000"

BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

print("📦 Loading dataset...")
dataset = load_from_disk("hf_dataset")["val"]

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)

preds = []
labels = []

model.eval()

print("⚡ Starting evaluation...")

for i, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    label = batch["labels"]

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    pred = logits.argmax(dim=1).cpu().numpy()

    preds.extend(pred)
    labels.extend(label.numpy())

    if i % 20 == 0:
        print(f"🚀 Processed {i} batches")

acc = accuracy_score(labels, preds)

print("\n🔥 FINAL Validation Accuracy:", acc)