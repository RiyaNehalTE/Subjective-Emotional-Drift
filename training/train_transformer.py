import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from training.dataset_loader import EmotionDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "roberta-base"

NUM_LABELS = 18

TRAIN_FILE = "data/train_encoded.csv"
VAL_FILE = "data/val_encoded.csv"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


train_dataset = EmotionDataset(TRAIN_FILE, tokenizer)
val_dataset = EmotionDataset(VAL_FILE, tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)


def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1
    }


training_args = TrainingArguments(

    output_dir="outputs_transformer",

    evaluation_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=32,

    per_device_eval_batch_size=32,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_dir="logs",

    logging_steps=200
)


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    tokenizer=tokenizer,

    compute_metrics=compute_metrics
)


trainer.train()