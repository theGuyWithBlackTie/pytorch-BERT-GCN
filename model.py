from transformers import *
import torch.nn as nn
import config


class BERTBaseUncased(nn.Module):
    def __init__(self, numOfLabels, dropout = 0.1):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        self.dropout = nn.Dropout(dropout)
        self.out     = nn.Linear(768, numOfLabels)

    def forward(self,ids, mask, token_type_ids):
        _, outputLayer = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids, return_dict=False)
        bo             = self.dropout(outputLayer)
        output         = self.out(bo)
        return output


class SciBertUncased(nn.Module):
    def __init__(self, numOfLabels, dropout = 0.1):
        super(SciBertUncased, self).__init__()
        self.sciBert = AutoModel.from_pretrained(config.SCI_BERT_PATH)
        self.dropout = nn.Dropout(dropout)
        self.out     = nn.Linear(768, numOfLabels)

    def forward(self, ids, mask, token_type_ids):
        _, outputLayer = self.sciBert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        bo             = self.dropout(outputLayer)
        output    = self.out(bo)
        return output 