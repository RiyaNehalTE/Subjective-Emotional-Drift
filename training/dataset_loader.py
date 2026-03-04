import pandas as pd
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):

        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        text = str(self.data.iloc[idx]["text"])
        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = label

        return item