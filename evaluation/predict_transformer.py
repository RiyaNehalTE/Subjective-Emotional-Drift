import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from training.dataset_loader import EmotionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


MODEL_PATH = "outputs_transformer/checkpoint-12872"
TEST_FILE = "data/test_encoded.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


dataset = EmotionDataset(TEST_FILE, tokenizer)

loader = DataLoader(dataset, batch_size=32)


predictions = []

for batch in tqdm(loader):

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    preds = torch.argmax(outputs.logits, dim=1)

    predictions.extend(preds.cpu().numpy())


df = pd.read_csv(TEST_FILE)

df["pred_label"] = predictions

df.to_csv("outputs_transformer/test_predictions.csv", index=False)

print("Predictions saved.")