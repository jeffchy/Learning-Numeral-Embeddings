import logging
from six import iteritems
from web.datasets.similarity import fetch_MEN, fetch_SimLex999
from web.embeddings import fetch_GloVe
from web.evaluate import evaluate_similarity
import pandas as pd
from sklearn.datasets.base import Bunch
import pickle
from web.embedding import Embedding
from web.datasets.analogy import fetch_semeval_2012_2
from collections import defaultdict, OrderedDict
import glob
import os
from six import string_types, text_type
import numpy as np
import scipy
from web.evaluate import evaluate_analogy


# word similarity path
WS353 = 'dataset/WS353/EN-WS353.txt'
WSR353 = 'dataset/WS353/EN-WSR353.txt'
MEN = 'dataset/MEN/EN-MEN-LEM.txt'
SIM999 = 'dataset/SIM999/EN-SIM999.txt'
# SEMVAL2012 = 'dataset/EN-SEMVAL2012-2'
# GOOGLE = 'dataset/GOOGLE/EN-GOOGLE.txt'

# Semantic Part
def fetch_dataset_WS353(path):
    data = pd.read_csv(path, header=0, sep='\t')
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(np.float)
    data = Bunch(X=X.astype("object"), y=y)
    return data.X, data.y


def fetch_dataset_MEN(path):
    data = pd.read_csv(path, header=None, sep=' ')
    data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    X = data.values[:, 0:2].astype("object")
    y = data.values[:, 2:].astype(np.float) / 5.0
    data = Bunch(X=X, y=y)
    return data.X, data.y


def fetch_dataset_SIM999(path):
    data = pd.read_csv(path, sep='\t')
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values
    temp = Bunch(X=X.astype("object"), y=y, sd=sd, conc=conc, POS=POS, assoc=assoc)
    return temp.X, temp.y


# def fetch_google_analogy(path):
#     with open(path, "r") as f:
#         L = f.read().splitlines()
#
#     # Simple 4 word analogy questions with categories
#     questions = []
#     answers = []
#     category = []
#     cat = None
#     for l in L:
#         if l.startswith(":"):
#             cat = l.lower().split()[1]
#         else:
#             words = standardize_string(l).split()
#             questions.append(words[0:3])
#             answers.append(words[3])
#             category.append(cat)
#
#     assert set(category) == set(['gram3-comparative', 'gram8-plural', 'capital-common-countries',
#                                  'city-in-state', 'family', 'gram9-plural-verbs', 'gram2-opposite',
#                                  'currency', 'gram4-superlative', 'gram6-nationality-adjective',
#                                  'gram7-past-tense',
#                                  'gram5-present-participle', 'capital-world', 'gram1-adjective-to-adverb'])
#
#     syntactic = set([c for c in set(category) if c.startswith("gram")])
#     category_high_level = []
#     for cat in category:
#         category_high_level.append("syntactic" if cat in syntactic else "semantic")
#
#     # dtype=object for memory efficiency
#     return Bunch(X=np.vstack(questions).astype("object"),
#                  y=np.hstack(answers).astype("object"),
#                  category=np.hstack(category).astype("object"),
#                  category_high_level=np.hstack(category_high_level).astype("object"))

def evaluate(embeddings):
    X, y = fetch_dataset_WS353(WS353)
    print(embeddings.shape, X.shape, y.shape)
    print("Spearman correlation of scores on {} {}".format('WS353', evaluate_similarity(embeddings, X, y)))
    X, y = fetch_dataset_MEN(MEN)
    print("Spearman correlation of scores on {} {}".format('MEN', evaluate_similarity(embeddings, X, y)))
    X, y = fetch_dataset_SIM999(SIM999)
    print("Spearman correlation of scores on {} {}".format('SIM999', evaluate_similarity(embeddings, X, y)))


if __name__ == '__main__':
    preprocess_base_dir = '../data/wikipedia/preprocess1B/NumeralAsNumeral'
    vec_base_dir = '../data/wikipedia/save/1B/prototypes/'
    # postfix = ['LSTM', 'NumeralAsToken', 'NumeralAsTokenUnkNumeral','NumeralAsUnkNumeral']
    # postfix = ['50','100','200','300']
    # postfix = ['NumeralAsToken', 'NumeralAsToken3','NumeralAsToken8']
    # postfix = ['NumeralAsTokenUnkNumeral5_300']
    postfix = ['3','5']

    for p in postfix:

        vec = glob.glob(vec_base_dir + '/{}/idx2vec_i*.dat'.format(p))

        idx2word_path = preprocess_base_dir + '/idx2word.dat'
        idx2word = pickle.load(open(idx2word_path, 'rb'))
        for v in vec:
            print('evaluate vector file {}, in {}'.format(v,p))
            idx2vec = pickle.load(open(v, 'rb'))
            dicts = {idx2word[i]: idx2vec[i] for i in range(len(idx2vec))}
            embeddings = Embedding.from_dict(dicts)
            evaluate(embeddings)