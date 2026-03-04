import pandas as pd

train = pd.read_csv("data/persona_train.csv")
val = pd.read_csv("data/persona_val.csv")
test = pd.read_csv("data/persona_test.csv")

print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)

print("\nColumns:")
print(train.columns)

print("\nSample rows:")
print(train.head())