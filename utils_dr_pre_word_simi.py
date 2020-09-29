from utils import *
from gensim.models.keyedvectors import KeyedVectors
import os
import sys

#fw = open('verb_outside_glove.txt', 'w')

def simi_each_word(verb, agenv, vcat, glove_model, agency_cats):
    res = []
    res.append(verb)
    res.append(vcat)
    if verb not in glove_model:
        res.append('none')
        res.append('none')
        res.append('none')
        return res
    for cat in agency_cats:
        verbset = agenv[cat]
        '''
        for v in verbset:
            if v not in glove_model:
                fw.write(v+'\n')
        '''
        verbset = list(filter(lambda v: v in glove_model, verbset)) 
        cp_vs = verbset.copy()
        if verb in cp_vs:
            cp_vs.remove(verb)
        verb_simi = glove_model.most_similar_to_given(verb, cp_vs)
        res.append(verb_simi)
    return res

def simi_verb_each_cat():
    '''
    for each word, get its most similar word in each cat from glove emb
    both word and its simi words are supposed to be in infinitive form
    '''
    # cat -> infi form of words
    df = pd.DataFrame()
    agency_cats = ['pos', 'neg', 'equal']
    glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
    vs_col = ['verb', 'oricat']
    vs_col.extend(agency_cats)
    for cat, verbset in agen_v.items():
        data = [simi_each_word(v, agen_v, cat, glove_model, agency_cats) for v in verbset]
        catdf = pd.DataFrame(data, columns=vs_col)
        df = df.append(catdf)
    df.to_csv('verb2simi.csv')
    return df

def load_word2simi():
    '''
    return the dataframe with column [verb, oricat, pos, neg equal]
    '''
    csvfile = './verb2simi.csv'
    if os.path.exists(csvfile):
        print('reading from csv')
        return pd.read_csv(csvfile)
    return simi_verb_each_cat()
