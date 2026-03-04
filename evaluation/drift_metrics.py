import pandas as pd


def compute_drift_for_conversation(labels):
    
    transitions = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transitions += 1

    if len(labels) <= 1:
        return 0

    return transitions / (len(labels) - 1)


def compute_dataset_drift(df):

    drift_scores = []

    grouped = df.sort_values("turn_index").groupby("conversation_id")

    for _, group in grouped:

        labels = group["pred_label"].tolist()

        drift = compute_drift_for_conversation(labels)

        drift_scores.append(drift)

    return sum(drift_scores) / len(drift_scores)