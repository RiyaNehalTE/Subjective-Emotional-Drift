import pandas as pd
from evaluation.drift_metrics import compute_dataset_drift


PRED_FILE = "outputs_transformer/test_predictions.csv"


df = pd.read_csv(PRED_FILE)


drift_score = compute_dataset_drift(df)


print("Average Drift Score:", drift_score)