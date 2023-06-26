from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from tsundoku.models.dataset_class import BETOTokenizer, BETOModel


class BETOTweeterClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BETOTweeterClassifier, self).__init__()
        self.beto = BETOModel
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.beto.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        output = self.beto(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.pooler_output
        drop_output = self.drop(cls_output)
        output = self.linear(drop_output)
        return output
