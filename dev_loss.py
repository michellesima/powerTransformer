import torch
from transformers import *
import pandas as pd
from utils import *
from utils_dr import *
from torch.utils import data
from torch.nn import CrossEntropyLoss
import os.path
from torchsummary import summary
import torch.optim as optim
import sys

random_seed = 7
max_sen_len = 64
batchsize = 4

if __name__ == '__main__':
    train_type = sys.argv[1]
    startmind = int(sys.argv[2])
    
    # CUDA for PyTorch
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    if train_type == 'para':
        dev_ds = parse_file_dr(DEV_DR, para=True)
        savedir = './modelp/savedmodels'
    elif train_type == 'mix':
        dev_ds = parse_file_dr(DEV_DR, para=True)
        dev_roc = parse_file_dr(ROC_DEV)
        dev_ds.append(dev_roc)
        savedir = './modelmix/savedmodels'
    else:
        dev_ds = parse_file_dr(ROC_DEV)
        savedir = './modelr/savedmodels'
    dev_generator = data.DataLoader(dev_ds, batch_size = batchsize, shuffle=True)
    ini = 0
    train_losses = []
    # Loop over epochs
    for epoch in range(startmind, 10):
        # Training
        savepath = savedir + str(epoch + 1)
        if not os.path.exists(savepath):
            break
        model = OpenAIGPTLMHeadModel.from_pretrained(savepath)
        model.to(device_dr)
        model.eval()
        losssum = 0.0
        count = 0
        for local_batch, local_labels in enumerate(dev_generator):
            # Transfer to GPUpri
            x, label = parse_model_inputs_dr(local_labels)
            outputs = model(x.to(device_dr), labels=label.to(device_dr))
            loss, logits = outputs[:2]
            losssum += loss
            count += 1
            loss.backward()
            # Model computations
        avg = losssum / count
        print(epoch)
        print(avg)
        train_losses.append(avg)

    loss_df = pd.DataFrame()
    loss_df["dev_loss"] = train_losses
    loss_df.to_csv('dev_loss_'+train_type+'.csv')
