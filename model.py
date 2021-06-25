import torch.nn as nn
from transformers import BertForSequenceClassification

class bert_binary(nn.Module):
    def __init__(self):
        super(bert_binary, self).__init__()
        self.encoder = BertForSequenceClassification.from_pretrained('bert-base-chinese')
    
    def forward(self, text_id, text_mask, label):
        loss, text_fea = self.encoder(text_id, text_mask, labels=label)[:2]
        return loss, text_fea
    