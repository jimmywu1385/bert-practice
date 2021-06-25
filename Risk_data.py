import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch

class Risk_data(Dataset):
    def __init__(self, filename):
        super().__init__()
        df = pd.read_csv(filename)
        self.label = df['label']
        self.text = df['text']      
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    def __getitem__(self, ind: int):
        encode = self.tokenizer.encode_plus(
                                self.text[ind], 
                                add_special_tokens=True,
                                max_length=500,
                                padding= 'max_length',
                                return_attention_mask=True,
                                return_tensors='pt',
                                truncation=True
                            )
        encode['input_ids'] = torch.squeeze(encode['input_ids'])
        encode['attention_mask'] = torch.squeeze(encode['attention_mask'])
        labels = torch.tensor([int(self.label[ind])])
        return labels, encode

    def __len__(self):
        return len(self.label.index)