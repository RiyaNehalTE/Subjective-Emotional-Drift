import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 18


# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Dataset
dataset = load_from_disk("hf_dataset")

full_train = dataset["train"].shuffle(seed=42)
subset_size = int(0.5 * len(full_train))
train_dataset = full_train.select(range(subset_size))

val_dataset = dataset["val"]

print("Subset train size:", len(train_dataset))


# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)


# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {"accuracy": acc, "f1": f1}


# Training args (OPTIMIZED)
training_args = TrainingArguments(

    output_dir="outputs_deberta_best",

    # evaluation
    eval_strategy="epoch",
    save_strategy="epoch",

    # ⭐ save BEST model only
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    save_total_limit=2,

    # ⭐ optimized learning
    learning_rate=1.5e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=50,

    dataloader_num_workers=4,

    report_to="none",

    # stability
    fp16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
)


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,

    # ⭐ early stopping
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)


# Safe training
try:
    trainer.train()
except KeyboardInterrupt:
    print("\nSaving model before exit...")
    trainer.save_model("outputs_deberta_best/manual_save")