from transformers import *
import torch
import sys
import pandas as pd
import math
'''
get the perplexity on gpt2
python gpt2per.py <dataset> <method> <toeval('sen' or 'sen0' or 'out')> (<pvalue>)
'''
device_2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def measurepp(df, toeval):
    df = df.sample(n=150)
    outsen = df[toeval].tolist()
    gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2.eval()
    gpt2.to(device_2)
    res = 0
    m = 0
    softmax = torch.nn.Softmax(dim=1)
    count = 0
    for sen in outsen:
        tok_li = gpt2tokenizer.encode(sen, add_special_tokens=False)
        tok_ten = torch.tensor(tok_li, device=device_2)
        loss, logits, past = gpt2(tok_ten, labels=tok_ten)
        count += 1
        res += loss
        '''
        m += len(tok_li)
        count += 1
        for i in range(logits.size()[0]):
            logits = softmax(logits)
            res += math.log2(logits[i][tok_li[i]])
            '''
    res /= count
    print(res)
    res = math.exp(res)
    print(res)

if __name__ == '__main__':
    ds, method = sys.argv[1], sys.argv[2]
    filepath = './gen_sen/res_sen_' + ds + '_' + method + '.csv'
    df = pd.read_csv(filepath)
    toeval = sys.argv[3]
    if len(sys.argv) == 5:
        pval = float(sys.argv[4])
        df = df[df['p-value']==pval]
    if ds == 'para':
        df = df.sample(n=1000)
    measurepp(df, toeval)
