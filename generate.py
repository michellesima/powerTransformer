from transformers import *
import torch
import argparse
from torch.utils.data import DataLoader
import pandas as pd
from utils import *
from utils_dr import *
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from examples.run_generation import *
import sys

max_sen_len = 64
random_seed = 7
numepoch = 10
ps = [0.4]
agen_vector = agen_vector(tokenizer_dr, num_added_token_dr, multi=False)
agen_v = agen_verbs()
softmax = nn.Softmax(dim=0)
REPEAT_PENALTY = 5


def sample_sequence_ivp(model, length, context, verb_vector, num_samples=1, temperature=1, tokenizer=None, top_k=0,
                        top_p=0.0, \
                        repetition_penalty=1.0, xlm_lang=None, device='cpu', multi=True):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    orilen = len(context[0])
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated}
            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)
            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)

            verb_vector = verb_vector.to(device)
            if multi:
                next_token_logits = next_token_logits * verb_vector
            else:
                next_token_logits += verb_vector
            for j in set(generated[0][orilen + 1:].view(-1).tolist()):
                if multi:
                    next_token_logits[j] /= repetition_penalty
                else:
                    next_token_logits[j] /= 1

            next_token_logits = softmax(next_token_logits)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # if temperature == 0: #greedy sampling:
            next_token = torch.argmax(filtered_logits).unsqueeze(0)
            # else:
            # next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def gen_p(model, test_dataset, descat, vocabBoost):
    outlist = []
    outp = []
    print(len(test_dataset))
    if vocabBoost:
        scaling = VER_ADD_VAL
    else:
        scaling = 0
    for i in ps:
        for j in range(len(test_dataset)):
            sen = test_dataset[j]
            senlen = len(sen)
            verb_vector = agen_vector[descat[j]] * scaling
            out = sample_sequence_ivp(
                model=model,
                context=sen,
                verb_vector=verb_vector,
                length=max_sen_len,
                top_p=i,
                multi=False,
                repetition_penalty=REPEAT_PENALTY,
                device=device_dr
            )
            out = out[0, senlen:].tolist()
            text = tokenizer_dr.decode(out, clean_up_tokenization_spaces=True, skip_special_tokens=False)
            end_ind = text.find('<end>')
            if end_ind >= 0:
                text = text[0: end_ind]
            outlist.append(text)
            outp.append(i)
    return outlist, outp

def eval_model(mind, test_dataset, df, vocabBoost, mtd='para'):
    '''
    get generated sentence for a particular model
    '''
    finaldf = pd.DataFrame()
    if mtd == 'para':
        savedir = './modelp/savedmodels'
    elif mtd == 'joint':
        savedir = './modelmix/savedmodels'
    else:
        savedir = './modelr/savedmodels'

    if 'sen0' in df.columns:
        colsen = 'sen0'
    else:
        colsen = 'sen'
    modelpath = savedir + str(mind)
    print(modelpath)
    model = OpenAIGPTLMHeadModel.from_pretrained(modelpath)
    model.to(device_dr)
    model.eval()
    df = repeatN(df, len(ps) - 1)
    outlist, outp = gen_p(model, test_dataset, df['descat'].tolist(), vocabBoost)
    df['out'] = outlist
    df['p-value'] = outp
    df.sort_values(by=[colsen, 'p-value'], inplace=True)
    df['modelind'] = mind
    finaldf = finaldf.append(df, ignore_index=True)
    return finaldf

def gen(mind, ds, vocabBoost, model='para'):
    if ds == 'para':
        f = DEV_DR
    elif ds == 'roc':
        f = ROC_DEV
    elif ds == 'roc-test':
        f = ROC_TEST
    else:
        f = MOVIE_DATA
    test_dataset, df = parse_file_dr(f, train_time=False)
    print(len(df.index))
    finaldf = eval_model(mind, test_dataset, df, vocabBoost, mtd=model)
    savedfile = 'gen_sen/joint-none-test.csv'
    finaldf.to_csv(savedfile, index=False)

def main(ds, mind, mtd, vocabBoost):
    args = {}
    args['n_ctx'] = max_sen_len
    # change to -> load saved dataset
    gen(mind, ds, vocabBoost, mtd)

if __name__ == '__main__':
    # mtd: model trained dataset
    parser = argparse.ArgumentParser(description='Process generation parameters')
    parser.add_argument('--dataset', type=str,
                        help='dataset for generation')
    parser.add_argument('--setup', type=str,
                        help='model setup objective')
    parser.add_argument('--epoch', type=str, default=0,
                        help='the previous trained epoch to load')
    parser.add_argument('--vocabBoost', action='store_true')
    args = parser.parse_args()
    main(args.dataset, args.epoch, args.setup, args.vocabBoost)
