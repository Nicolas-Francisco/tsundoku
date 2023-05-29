from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import Dataset


PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
BETOTokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BETOModel = AutoModelForMaskedLM.from_pretrained(PRE_TRAINED_MODEL_NAME)


class TsundokuUsersDataset(Dataset):
    def __init__(
        self,
        descriptions,
        location,
        name,
        screen_name,
        url,
        labels,
        tokenizer,
        max_len=200,
    ):
        self.descriptions = descriptions
        self.location = location
        self.name = name
        self.screen_name = screen_name
        self.url = url
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.descriptions)

    def encode(self, text):
        return BETOTokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __getitem__(self, item):
        descriptions = str(self.descriptions[item])
        label = self.labels[item]
        encoding = self.encode(descriptions)

        return {
            "description": descriptions,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }
