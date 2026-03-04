import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load datasets
train = pd.read_csv("data/persona_train.csv")
val = pd.read_csv("data/persona_val.csv")
test = pd.read_csv("data/persona_test.csv")

# Initialize encoder
encoder = LabelEncoder()

# Fit on training labels
train["label"] = encoder.fit_transform(train["emotion"])

# Apply same mapping to val/test
val["label"] = encoder.transform(val["emotion"])
test["label"] = encoder.transform(test["emotion"])

# Save encoded datasets
train.to_csv("data/train_encoded.csv", index=False)
val.to_csv("data/val_encoded.csv", index=False)
test.to_csv("data/test_encoded.csv", index=False)

# Save label mapping for later use
joblib.dump(encoder, "data/label_encoder.pkl")

print("Label encoding complete.")
print("Number of classes:", len(encoder.classes_))
print("Classes:", encoder.classes_)