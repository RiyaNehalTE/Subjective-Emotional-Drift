import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "roberta-base"
NUM_LABELS = 18


# -------------------------
# Tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -------------------------
# Load FAST pre-tokenized dataset
# -------------------------
dataset = load_from_disk("hf_dataset")
train_dataset = dataset["train"]
val_dataset = dataset["val"]


# -------------------------
# Model
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)


# -------------------------
# Metrics
# -------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1
    }


# -------------------------
# Training config
# -------------------------
training_args = TrainingArguments(
    output_dir="outputs_transformer",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_dir="logs",
    logging_steps=10,

    dataloader_num_workers=8,   # ⭐ IMPORTANT FIX
    report_to="none"
)


# -------------------------
# Trainer
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# -------------------------
# Train
# -------------------------
trainer.train()