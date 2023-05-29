from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
BETOTokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BETOModel = AutoModelForMaskedLM.from_pretrained(PRE_TRAINED_MODEL_NAME)


class BETOTweeterClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BETOTweeterClassifier, self).__init__()
        self.beto = BETOModel
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.beto.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, cls_output = self.beto(input_ids=input_ids, attention_mask=attention_mask)
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output
