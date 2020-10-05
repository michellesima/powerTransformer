import pandas as pd
from nltk.stem import WordNetLemmatizer 
from transformers import *

def gen_cri(df):
    df['cri'] = df['sen'].str.replace(' ', '')
    df['cri'] = df['cri'].str.replace('[^\w\s]', '')
    df['cri'] = df['cri'].str.lower()
    return df

def dev_he(df):
    hedf = pd.read_csv(ROC_DEV_HE)
    df = gen_cri(df)
    hedf = gen_cri(hedf)
    df = df[df['cri'].isin(hedf['cri'])]
    df.drop(columns=['cri'], inplace=True)
    return df

def repeatN(list, n):
    ori = list
    for _ in range(n):
        list = list.append(ori, ignore_index=True)
    return list

def agen_verbs():
    '''
    for word in each category, get its infinitive form if it's in verb.txt
    for short phrases like 'apply to', only the first word is considered
    Note: 24 words not in verb.txt
    '''
    df = pd.read_csv('~/resources/lexica/CONNOTATION/agency_verb.csv')
    agen_v = {}
    total = 0
    cats = {'+': 'pos', '-':'neg', '=':'equal'}
    for k, v in cats.items():
        subdf = df[df['Agency{agent}_Label'] == k]
        ver_li = subdf['verb'].str.split()
        agen_v[v] = set(word_infinitive(li[0]) for li in ver_li if len(li) > 0)
        total += len(agen_v[v])
    return agen_v



def word_infinitive(word):
    #infi = lemmatizer.lemmatize(word)
    row = verb_form[verb_form.isin([word]).any(axis=1)]
    if row.empty:
        return word
    infi = row[0].iloc[0]
    return infi 

def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def add_pad(list, tokenizer):
    res = [__sen_pad(sen, tokenizer) for sen in list]
    return res

def __sen_pad(sen, tokenizer):
    # add padding for each sentence
    if len(sen) < max_sen_len:
        pad = [tokenizer.pad_token_id for i in range(max_sen_len - len(sen))]
        sen.extend(pad)
        return sen
    elif len(sen) > max_sen_len:
        orilen = len(sen)
        for i in range(orilen - max_sen_len):
            sen.pop(len(sen) - 2)
    return sen


max_sen_len = 64
#lemmatizer = WordNetLemmatizer() 
verb_form = pd.read_csv('verb.txt', usecols=[_ for _ in range(24)], header=None)
ps = [0.4, 0.6]
num_epoch = 10

agen_v = agen_verbs()
ROC_TRAIN = './con_rew_data/roc/train.csv'
ROC_TEST = './con_rew_data/roc/test.csv'
ROC_TEST_HE = './con_rew_data/roc/supplyVerb.csv'
ROC_DEV = './con_rew_data/roc/dev.csv'
ROC_DEV_HE = './con_rew_data/for_human_eval.csv'

MOVIE_DATA_G = './data/movie/forg/movie.pickle'
MOVIE_DATA_G_NONCAT = './data/movie/forg/movie_noncat.pickle'
MOVIE_DATA_G_NONCAT_M = './data/movie/forg/movie_noncat_missing.pickle'

MOVIE_DATA = './data/movie/narrs.split.pre.csv'
