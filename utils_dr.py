import pandas as pd
from utils_dr_pre_word_simi import *
import os
from utils import *
from transformers import *
from dataset import Dataset_dr
import torch
import numpy as np

TRAIN_DR = 'data/parads/train_dr.csv'
DEV_DR = 'data/parads/dev_dr.csv'
TEST_DR = 'data/parads/test_dr.csv'

fw = open('verb_no_simi.txt', 'w')

VER_MAG_RATE = 1.5
VER_ADD_VAL = 5

batchsize_dr = 4
device_dr = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
verb2simi = load_word2simi()
tokenizer_dr = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
token_dict_dr = {
    'bos_token': '<start>',
    'eos_token': '<end>',
    'pad_token': '<pad>',
    'cls_token': '<cls>',
    'additional_special_tokens': ['<pos>', '<neg>', '<equal>', '<VERB>']
}
num_added_token_dr = tokenizer_dr.add_special_tokens(token_dict_dr)
print(tokenizer_dr.vocab_size)
cats = ['pos', 'neg', 'equal']


def agen_vector(tokenizer, num_added, multi=True):
    agen_vectors = {}
    for label, verbset in agen_v.items():
        if multi:
            vector = torch.ones(tokenizer.vocab_size + num_added)
        else:
            vector = torch.zeros(tokenizer.vocab_size + num_added)
        for v in verbset:
            forms = infi2allforms(v)
            for form in forms:
                v_li = tokenizer.encode(form)
                if multi:
                    vector[v_li[0]] *= VER_MAG_RATE
                else:
                    vector[v_li[0]] = VER_ADD_VAL
        agen_vectors[label] = vector
    return agen_vectors

def infi2allforms(word):
    res = []
    row = verb_form[verb_form[0] == word]
    if row.empty:
        res.append(word)
        return res
    row = row.dropna(axis=1)
    for col in row.columns:
        res.append(row[col].iloc[0])
    return res

def get_he_df(df):
    df['cri'] = df['sen'].str.replace(' ', '')
    df['cri'] = df['cri'].str.lower()
    he_df = pd.read_csv(ROC_TEST_HE)
    he_df['cri'] = he_df['sen'].str.replace(' ', '')
    he_df['cri'] = he_df['cri'].str.lower()
    df = df[df['cri'].isin(he_df['cri'])]
    print(len(df.index))
    return df


def simi_word(verb, descat):
    '''
    at train and gen time, get the simi verb with descat
    get the infi form of word
    '''
    infi = word_infinitive(verb)
    row = verb2simi[verb2simi['verb'] == infi]
    li = row[descat].tolist()
    if len(li) > 0:
        return li[0]
    fw.write(verb+'\n')
    return verb

def extract_args(sen, para, train_time):
    if para:
        sen_del = sen['sendel0']
        descat = sen['oricat1']
        verbs = sen['verbs1']
        para_sen = sen['sen1']
    else:
        sen_del = sen['sendel']
        descat = sen['oricat']
        verbs = sen['verbs']
        para_sen = sen['sen']
    if not train_time:
        descat = sen['descat']
    return sen_del, descat, verbs, para_sen

def sen_in(sen, noi_idx, train_time=True, para=False):
    sen_idx = sen[0]
    sen = sen[1]
    sen_del, descat, verbs, para_sen = extract_args(sen, para, train_time)
    ori_verbs = verbs.split()
    add_verbs = ''
    if sen_idx in noi_idx:
        for v in ori_verbs:
            add_verbs += simi_word(v, descat)
    else:
        add_verbs = verbs
    newsen = '<start> ' + sen_del
    if not train_time:
        newsen = newsen + '<cls> ' + descat + '<start>'
    else:
        newsen += '<cls> <' + descat + '> <start> ' + para_sen + ' <end>'
    tok_li = tokenizer_dr.encode(newsen, add_special_tokens=False)
    return tok_li, add_verbs

def sen_in_retr(sen, df, method):
    senavg = df[df['sen']==sen]['glove_avg']
    df['glove_avg'] = df['glove_avg'] - senavg


def parse_file_dr(file, noi_frac=0.1, train_time=True, para=False):
    path = os.path.abspath(file)
    with open(path,encoding='UTF-8') as f:
        df = pd.read_csv(f)
        noi_df = df.sample(frac=noi_frac)
        if train_time:
            tok_li = [sen_in(sen, noi_df.index, train_time=train_time, para=para) for sen in df.iterrows()]
            tok_li = np.array(tok_li)
            df['v_supplied'] = tok_li[:, 1]
            tok_li = tok_li[:, 0]
        else:
            cats = ['pos', 'equal', 'neg']
            tok_li = []
            retdf = pd.DataFrame()
            if False:
                df = dev_he(df)
            for cat in cats:
                subdf = df.copy()
                subdf['descat'] = cat
                if para:
                    subdf['cat'] = df['oricat0']
                else:
                    subdf['cat'] = df['oricat']
                tem = [sen_in(sen, subdf.index, train_time=train_time, para=para) for sen in subdf.iterrows()]
                tem = np.array(tem)
                subdf['v_supplied'] = tem[:, 1]
                tem = tem[:, 0]
                tok_li.extend(tem)
                retdf = retdf.append(subdf)
        if not train_time:
            return tok_li, retdf
        tok_li = add_pad(tok_li, tokenizer_dr)
        dataset = Dataset_dr(list_IDs=tok_li)
        return dataset

def get_label_dr(tokenizer, x, g=False):
    label = x.clone()
    start_inds = ((x == tokenizer.bos_token_id).nonzero())
    end_inds = ((x == tokenizer.eos_token_id).nonzero())
    for i in range(x.size()[0]):
        # do not include the last cls token
        end_pos = i
        if g:
            end_pos = 2 * i + 1
        startind = start_inds[2*i+1][1].item() + 1
        endind = end_inds[end_pos][1].item() + 1
        # do not include second start
        label[i][0:startind] = torch.FloatTensor([-1 for _ in range(startind)])
        # include end token
        label[i][endind:] = torch.FloatTensor([-1 for _ in range(max_sen_len - endind)])
    return label

def parse_model_inputs_dr(local_labels):
    x = local_labels # b * s
    label = get_label_dr(tokenizer_dr, x)
    return x, label
