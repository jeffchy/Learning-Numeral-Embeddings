import sys
sys.path.append('../../numeral_context/')
sys.path.append('../')
import numpy as np
from evaluate import f1_metrics
import pandas as pd

class Evaluator(object):

    def __init__(self, word2idx, dataset):

        self.word2idx = word2idx
        self.dataset = dataset

    def load_numeral_embeds(self, numeral_embed_i, numeral_embed_o):
        self.numeral_embed_i = numeral_embed_i  # N x D
        self.numeral_embed_o = numeral_embed_o

    def load_dataset(self, dataset):
        self.dataset = dataset
        # ('2010', ['100', 'most', 'anticipated', 'books', 'releasing', 'in'], 4, 6)

    def load_word2idx(self, word2idx):
        self.word2idx = word2idx

    def load_idx2vec(self, idx2vec_i, idx2vec_o):
        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o

    def compute_norm_factor_all(self, idx2vec_o_original):
        # N x H
        # normalization factor for p(c_j|n_i)
        mat = np.dot(idx2vec_o_original, np.array(self.numeral_embed_i).T)
        mat = np.exp(mat)
        norms = np.log(np.sum(mat, axis=0))
        self.norms = norms

    def _get_score(self, i):
        s0 = np.zeros((self.numeral_embed_i.shape[0],))
        s2 = np.zeros((self.numeral_embed_i.shape[0],))
        for c in i[1]:
            try:
                wordvec = self.idx2vec_o[self.word2idx[c]]
            except KeyError:
                wordvec = self.idx2vec_o[self.word2idx['<unk>']]
                # print('should not!')

            score = np.dot(wordvec, np.array(self.numeral_embed_i).T)
            s0 += score

            try:
                wordvec = self.idx2vec_i[self.word2idx[c]]
            except KeyError:
                wordvec = self.idx2vec_i[self.word2idx['<unk>']]
                # print('should not!')


            score = np.dot(wordvec, np.array(self.numeral_embed_o).T)
            s2 += score

        s1 = s0 - len(i[1]) * self.norms

        s3 = s1 + s2

        return [s0, s1, s2, s3] # \sum s(c_j|n) , \sum s(c_j|n) - Z(n), \sum s(n|c_j), \sum (s(c_j|n)+s(n|c_j))

    def evaluate_kbest(self, kbest):

        right = [0, 0, 0, 0]

        for i in self.dataset:

            scores = self._get_score(i)
            label = i[2]

            for s in range(len(scores)):

                nn = np.argsort(scores[s])[-kbest:]

                if (label-1) in nn:
                    right[s] += 1

        return [i / len(self.dataset) for i in right]

    def evaluate_f1(self):

        res = []
        train_TP = np.zeros((4, 8))
        train_FP = np.zeros((4, 8))
        train_FN = np.zeros((4, 8))

        for i in self.dataset:

            scores = self._get_score(i)
            label = i[2]

            for s in range(len(scores)):

                nn = np.argmax(scores[s])
                # if s == 1:
                    # print(nn)
                if nn == label-1:
                    train_TP[s][nn] += 1
                else:
                    train_FP[s][nn] += 1
                    train_FN[s][label-1] += 1

        for s in range(len(scores)):
            micro_f1, macro_f1 = f1_metrics(train_TP[s], train_FP[s], train_FN[s])
            # print(train_TP)
            res.append([micro_f1, macro_f1])

        return np.array(res).T # 2 x 3


    def evaluate_avg_rank(self):
        total_rank = [0, 0, 0, 0]

        for i in self.dataset:

            scores = self._get_score(i)
            label = i[2]

            for s in range(len(scores)):

                nn = np.argsort(scores[s])[::-1]

                # optimize
                rank = 0
                for k in nn:
                    rank += 1
                    if k == (label-1):
                        break

                total_rank[s] += rank

        avg_rank = [i / len(self.dataset) for i in total_rank]
        return np.array(avg_rank)

    def show_scores(self, result_dict, score=0):
        for k, v in result_dict.items():
            result_dict[k] = np.array(v)
        score_dict = {k: v[:,score] for k, v in result_dict.items()}
        return pd.DataFrame(score_dict)
