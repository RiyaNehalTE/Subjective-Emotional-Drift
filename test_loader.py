from transformers import AutoTokenizer
from training.dataset_loader import EmotionDataset

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

dataset = EmotionDataset(
    "data/train_encoded.csv",
    tokenizer
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print("Sample keys:", sample.keys())
print("Input IDs shape:", sample["input_ids"].shape)
print("Label:", sample["labels"])