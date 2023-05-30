from transformers import BertTokenizer, BertModel
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

PRE_TRAINED_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
BETOTokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
BETOModel = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


class BETOTweeterClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BETOTweeterClassifier, self).__init__()
        self.beto = BETOModel
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.beto.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.beto(input_ids=input_ids, attention_mask=attention_mask)
        print(output)
        cls_output = output.pooler_output
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output
