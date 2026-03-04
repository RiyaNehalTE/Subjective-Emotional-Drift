import pandas as pd
from evaluation.drift_metrics import compute_dataset_drift


data = {
    "conversation_id": ["c1","c1","c1","c1","c1"],
    "turn_index":[0,1,2,3,4],
    "pred_label":[1,1,3,3,5]
}

df = pd.DataFrame(data)

score = compute_dataset_drift(df)

print("Drift score:", score)