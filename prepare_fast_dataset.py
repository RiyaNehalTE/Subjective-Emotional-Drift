from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "roberta-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading CSV dataset...")
dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/train_encoded.csv",
        "val": "data/val_encoded.csv",
    }
)

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokens["labels"] = batch["label"]   # keep labels
    return tokens

print("Tokenizing (one-time, cached)...")
dataset = dataset.map(
    tokenize,
    batched=True,
    num_proc=8,
    remove_columns=["conversation_id","persona_id","persona_text","turn_index","text","emotion","label"]
)

dataset.set_format("torch")

print("Saving cached dataset...")
dataset.save_to_disk("hf_dataset")

print("✅ Fast dataset ready.")