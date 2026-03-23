import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "microsoft/deberta-v3-base"
CHECKPOINT_PATH = "outputs_deberta/checkpoint-121000"
NUM_LABELS = 18


# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ✅ Load dataset
dataset = load_from_disk("hf_dataset")
train_dataset = dataset["train"]
val_dataset = dataset["val"]


# ✅ Load model from checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT_PATH,
    num_labels=NUM_LABELS
)


# ✅ Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {"accuracy": acc, "f1": f1}


# ✅ Training args (FINAL STABLE CONFIG)
training_args = TrainingArguments(

    output_dir="outputs_deberta",

    # evaluation
    eval_strategy="epoch",
    save_strategy="steps",

    # ⏱ checkpoint ~15 min
    save_steps=9000,
    save_total_limit=2,

    # 🎯 OPTIMIZED learning
    learning_rate=1.5e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=50,

    dataloader_num_workers=4,

    report_to="none",

    # 🔥 STABILITY (VERY IMPORTANT)
    fp16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
)


# ✅ Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)


# ✅ Resume training properly
trainer.train()