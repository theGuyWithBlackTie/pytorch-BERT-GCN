import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self, dropout = , numOfLabels):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained()
        self.dropout = nn.Droupout()
        self.out     = nn.Linear(768, numOfLabels)

    def forward(self,ids, mask, token_type_ids):
        _, outputLayer = self.bert(ids, attention_mask = mask, token_type_ids=token_type_ids)
        bo             = self.dropout(outputLayer)
        output         = self.out(bo)
        return output