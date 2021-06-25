from Risk_data import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
from model import bert_binary

def train(model, train_loader, val_loader, optimizer, epoch, device):
    for i in range(epoch):
        total_loss = 0
        model.train()
        print('')
        print('======== Epoch {:} / {:} ========'.format(i + 1, epoch))

        for step, (label, text) in enumerate(train_loader):
            print('  Batch {:>5,}  of  {:>5,}. '.format(step+1, len(train_loader)))
            
            label = label.to(device)
            text_id = text['input_ids'].to(device)
            text_mask = text['attention_mask'].to(device)
            optimizer.zero_grad()
            loss, _ = model(text_id, text_mask, label)
            total_loss += loss.item()
            print("total loss:",total_loss,"\naverage loss:",total_loss/(step+1),"\n-------------------------")
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        print("")
        print("Running Validation...")

        model.eval()
        total_val_loss = 0

        for step, (label, text) in enumerate(val_loader):
            print('  Batch {:>5,}  of  {:>5,}. '.format(step+1, len(val_loader)))
            label = label.to(device)
            text_id = text['input_ids'].to(device)
            text_mask = text['attention_mask'].to(device)
            with torch.no_grad():
                loss, _ = model(text_id, text_mask, label)
                total_val_loss += loss.item()
                print("total loss:",total_val_loss,"\naverage loss:",total_val_loss/(step+1),"\n-------------------------")

        avg_val_loss = total_val_loss / len(val_loader)
        print("")
        print("  Average validing loss: {0:.2f}".format(avg_val_loss))

    return model 


def main():
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    split_ratio = 0.7
    batch_size = 32
    epoch = 1

    total_data = Risk_data('Train_risk_classification_ans.csv')
    train_size, val_size = int(len(total_data)*split_ratio), len(total_data)-int(len(total_data)*split_ratio)
    train_set, val_set = random_split(total_data, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = bert_binary().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps = 1e-8)
    
    model_fit = train(model, train_loader, val_loader, optimizer, epoch, device)    
    torch.save(model_fit.state_dict(), 'bert_binary.pt')

if __name__ == '__main__':
    main()