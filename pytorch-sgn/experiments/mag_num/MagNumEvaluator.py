import pickle
import numpy as np
import pandas as pd
import sys
import torch as t
from tqdm import tqdm_notebook
sys.path.append('../numeral_context/')
sys.path.append('../../')
from som.intopolate import weighted_log
from utils.number_handler import to_numeral

class MagNumEvaluator():

    def __init__(self, numerals_dict, idx2word=None, word2idx=None, type='MAG'):
        """
            numerals_dict: sorted numeral dicts when tyoe=NUM, else an array
            idx2word: can be None except NumeralAsToken
            word2idx: can be None except NumeralAsToken
        """
        assert type in ['MAG', 'NUM']
        if type == 'NUM':
            self.numerals = np.array(list(numerals_dict.keys()))
            self.numerals_dict = numerals_dict
        else:
            self.numerals = np.array(numerals_dict)
            self.numerals.sort()


        self.idx2word = idx2word
        self.word2idx = word2idx
        self.type = type
        self.varbose = False

    def reload(self, numerals_dict=[], idx2word=None, word2idx=None, type=None):
        assert type in ['MAG', 'NUM', None]
        if numerals_dict != []:
            if type == 'NUM':
                self.numerals = np.array(list(numerals_dict.keys()))
                self.numerals_dict = numerals_dict
            else:
                self.numerals = np.array(numerals_dict)
                self.numerals.sort()
        if idx2word != None:
            self.idx2word = idx2word

        if word2idx != None:
            self.word2idx = word2idx

        if type != None:
            self.type = type

    def set_varbose(self, flag):
        self.varbose = flag

    def load_prototype(self,
                       trained_prototypes,
                       idx2vec_i=None,
                       idx2vec_o=None,
                       alpha=1.0,
                       log_space=False
                       ):

        def get_numeral_embed_weights_batch(numerals, transformed_protp, alpha=1.0, fn=weighted_log):
            """
            :param numerals: tensor of numerals
            :return: weights matrix
            """
            numerals.apply_(fn)
            l_numerals = numerals.size()[0]
            l_prototypes = transformed_protp.size()[0]

            min_margin = t.tensor(0.0001, dtype=t.float32)
            transformed_prototypes_batch = transformed_protp.expand(l_prototypes, l_numerals)
            prototype_weights = t.pow(1 / t.max(t.abs(transformed_prototypes_batch - numerals), min_margin), alpha)
            prototype_weights /= t.sum(prototype_weights, 0)

            return prototype_weights  # [prototype_size x num_of_numerals]

        prototypes = trained_prototypes['prototypes']
        prototypes2vec_i = trained_prototypes['i_embedding']
        prototypes2vec_o = trained_prototypes['o_embedding']
        if log_space:
            # if log_space, the prototype is already transformed
            transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).view(-1, 1)
        else:
            transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).apply_(weighted_log).view(-1, 1)

        prototype_weights = get_numeral_embed_weights_batch(t.tensor(t.from_numpy(self.numerals), dtype=t.float),
                                                            transformed_protp, alpha=alpha)
        self.numeral_embed_i = t.matmul(prototype_weights.transpose(0, 1), t.from_numpy(prototypes2vec_i))
        # self.numeral_embed_o = t.matmul(prototype_weights.transpose(0, 1), t.from_numpy(prototypes2vec_o))

        if self.type == 'NUM':
            self.numeral_vec_embed_i = t.tensor(
                [idx2vec_i[v] for k, v in self.numerals_dict.items()]
            )
            #
            # self.numeral_vec_embed_o = t.tensor(
            #     [idx2vec_o[v] for k, v in self.numerals_dict.items()]
            # )

    def load_fixed(self,
                idx2vec_i=None,
                idx2vec_o=None,):

        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o

        def get_numeral_embed_batch(numerals):
            """
            :param numerals: tensor of numerals!
            :return: embedding matrix of numerals
            """
            embedding_size = self.idx2vec_i.shape[1]
            numerals.apply_(weighted_log)
            embed = t.ones((embedding_size, len(numerals)))
            embed[0] = numerals
            embed = embed.transpose(0, 1)
            embed /= (embedding_size * 2)

            return embed  # [num_of_numerals x embedding_size]

        self.numeral_embed_i = get_numeral_embed_batch(t.tensor(t.from_numpy(self.numerals),dtype=t.float)) # (93, 150)
        # self.numeral_embed_o = get_numeral_embed_batch(t.tensor(t.from_numpy(self.numerals),dtype=t.float))   # (93, 150)
        if self.type == 'NUM':
            self.numeral_vec_embed_i = t.tensor(
                [idx2vec_i[v] for k, v in self.numerals_dict.items()]
            )


    def load_TOKEN(self,
                   idx2vec_i,
                   idx2vec_o,
                   ):
        # CAUTION, need to reload the word2idx, nc ...

        numerals_str = [str(i) for i in self.numerals]
        self.numeral_embed_i = np.zeros((len(self.numerals), idx2vec_i.shape[1]))
        self.numeral_embed_o = np.zeros((len(self.numerals), idx2vec_i.shape[1]))
        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o

        UNK_idxs = []

        num_oov = 0
        for i in range(len(numerals_str)):
            try:
                self.numeral_embed_i[i] = idx2vec_i[self.word2idx[numerals_str[i]]]
            except:
                print('oov:', numerals_str[i])
                UNK_idxs.append(i)
                num_oov += 1
                self.numeral_embed_i[i] = idx2vec_i[self.word2idx['<UNK_N>']]
        print(num_oov, num_oov / len(numerals_str))

        for i in range(len(numerals_str)):
            try:
                self.numeral_embed_o[i] = idx2vec_o[self.word2idx[numerals_str[i]]]

            except:
                print('oov:', numerals_str[i])
                self.numeral_embed_o[i] = idx2vec_o[self.word2idx['<UNK_N>']]

        if self.type == 'NUM':
            self.numeral_vec_embed_i = t.tensor(
                [idx2vec_i[v] for k, v in self.numerals_dict.items()]
            )

            # self.numeral_vec_embed_o = t.tensor(
            #     [idx2vec_o[v] for k, v in self.numerals_dict.items()]
            # )


    def load_GMM(self,
                 trained_prototypes,
                 gmm,
                 idx2vec_i=None,
                 idx2vec_o=None,
                 log_space=False):

        if log_space:
            prototype_weights = gmm.predict_proba(np.array([weighted_log(x) for x in self.numerals]).reshape(-1, 1))
        else:
            prototype_weights = gmm.predict_proba(self.numerals.reshape(-1, 1))

        prototypes2vec_i = trained_prototypes['i_embedding']
        prototypes2vec_o = trained_prototypes['o_embedding']
        self.numeral_embed_i = t.from_numpy(np.matmul(prototype_weights, prototypes2vec_i))  # (93, 150)
        self.numeral_embed_o = t.from_numpy(np.matmul(prototype_weights, prototypes2vec_o))  # (93, 150)
        self.numeral_embed_i = np.array(self.numeral_embed_i)
        self.numeral_embed_o = np.array(self.numeral_embed_o)

        if self.type == 'NUM':
            self.numeral_vec_embed_i = np.array(
                [idx2vec_i[v] for k, v in self.numerals_dict.items()]
            )


    def load_LSTM(self,
                  idx2vec_i,
                  idx2vec_o,
                  LSTM_model_path):

        LSTM_params = t.load(LSTM_model_path, map_location='cpu')

        digital_RNN_i = t.nn.LSTM(14, idx2vec_i.shape[1], 1, batch_first=True)
        # digital_RNN_o = t.nn.LSTM(14, idx2vec_i.shape[1], 1, batch_first=True)

        digital_RNN_i.bias_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.bias_hh_l0'])
        digital_RNN_i.bias_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.bias_ih_l0'])
        digital_RNN_i.weight_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.weight_hh_l0'])
        digital_RNN_i.weight_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.weight_ih_l0'])
        # digital_RNN_o.bias_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.bias_hh_l0'])
        # digital_RNN_o.bias_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.bias_ih_l0'])
        # digital_RNN_o.weight_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.weight_hh_l0'])
        # digital_RNN_o.weight_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.weight_ih_l0'])

        idx2digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '+', 'e']
        digit2idx = {idx2digit[i]: i for i in range(len(idx2digit))}
        max_token_len = 20  # should be equal to

        def convert_digit_to_tensor(numeral_str):
            represent = t.zeros(max_token_len, len(idx2digit))
            assert len(numeral_str) <= max_token_len
            for i in range(len(numeral_str)):
                digit = numeral_str[i]
                idx = digit2idx[digit]
                represent[i][idx] = 1

            return represent

        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o
        self.numeral_embed_i = np.zeros((len(self.numerals), idx2vec_i.shape[1]))
        self.numeral_embed_o = np.zeros((len(self.numerals), idx2vec_i.shape[1]))

        counter = 0
        for i in self.numerals:
            temp = convert_digit_to_tensor(str(i)).view(1, 20, 14)
            _, (hn, cn) = digital_RNN_i(temp)
            e = hn.squeeze().detach().numpy()
            self.numeral_embed_i[counter] = e
            counter += 1

        # counter = 0
        # for i in self.numerals:
        #     temp = convert_digit_to_tensor(str(i)).view(1, 20, 14)
        #     _, (hn, cn) = digital_RNN_o(temp)
        #     e = hn.squeeze().detach().numpy()
        #     self.numeral_embed_o[counter] = e
        #     counter += 1

        self.numeral_embed_i = np.array(self.numeral_embed_i)
        # self.numeral_embed_o = np.array(self.numeral_embed_o)

        if self.type == 'NUM':
            self.numeral_vec_embed_i = np.array(
                [idx2vec_i[v] for k, v in self.numerals_dict.items()],
            )

            # self.numeral_vec_embed_o = np.array(
            #     [idx2vec_o[v] for k, v in self.numerals_dict.items()]
            # )


    def find_first_and_second_closest_idx(self, i):
        if i == 0:
            return 1, 2
        elif i == len(self.numerals) - 1:
            return len(self.numerals) - 2, len(self.numerals) - 3
        else:
            _l = self.numerals[i] - self.numerals[i - 1]
            _r = self.numerals[i + 1] - self.numerals[i]

            if (i == 1) or (i == len(self.numerals) - 2):
                return (i - 1, i + 1) if (_l < _r) else (i + 1, i - 1)
            elif _l < _r:
                return (i - 1, i + 1) if (_r < self.numerals[i] - self.numerals[i - 2]) else (i - 1, i - 2)
            else:
                return (i + 1, i - 1) if (_l < self.numerals[i + 2] - self.numerals[i]) else (i + 1, i + 2)


    def find_furthest_idx(self, i):
        if self.numerals[-1] - self.numerals[i] < self.numerals[i] - self.numerals[0]:
            return 0
        else:
            return len(self.numerals) - 1

    def evaluate_ova(self):

        right = 0
        for i in range(len(self.numerals)):
            first_idx, second_idx = self.find_first_and_second_closest_idx(i)
            if self.type == 'NUM':
                second_idx = first_idx
                first_idx = i


            target_embed = self.numeral_vec_embed_i if self.type == 'NUM' else self.numeral_embed_i
            embeds_i = self.numeral_embed_i[i] - target_embed
            res_norm = np.linalg.norm(embeds_i, axis=1)
            res = np.argsort(res_norm)[1]
            if res == first_idx:
                right += 1

        if self.varbose:
            print("{}-OVA: {}".format(self.type, right / len(self.numerals)))

        res = right / len(self.numerals)
        return res

    def evaluate_ova_r(self):

        right = 0
        for i in range(len(self.numerals)):
            first_idx, second_idx = self.find_first_and_second_closest_idx(i)
            if self.type == 'NUM':
                second_idx = first_idx
                first_idx = i

            target_embed = self.numeral_vec_embed_i if self.type == 'NUM' else self.numeral_embed_i
            embeds_i = self.numeral_embed_i[i] - target_embed
            res_norm = np.linalg.norm(embeds_i, axis=1)
            res = np.argsort(res_norm)[1:4]
            if first_idx in res:
                right += 1

        if self.varbose:
            print("{}-OVA-R: {}".format(self.type, right / len(self.numerals)))

        res = right / len(self.numerals)
        return res

    def evaluate_sc(self):

        right = 0
        for i in range(len(self.numerals)):
            first_idx, second_idx = self.find_first_and_second_closest_idx(i)
            if self.type == 'NUM':
                second_idx = first_idx
                first_idx = i

            target_embed = self.numeral_vec_embed_i if self.type == 'NUM' else self.numeral_embed_i
            norm_first = np.linalg.norm(self.numeral_embed_i[i] - target_embed[first_idx])
            norm_second = np.linalg.norm(self.numeral_embed_i[i] - target_embed[second_idx])
            if norm_first < norm_second:
                right += 1
            # else:
            #     print(self.numerals[first_idx], self.numerals[second_idx], self.numerals[i], )

        if self.varbose:
            print("{}-SC: {}".format(self.type, right / len(self.numerals)))

        res = right / len(self.numerals)
        return res

    def evaluate_bc(self):
        right = 0
        for i in range(len(self.numerals)):
            first_idx, second_idx = self.find_first_and_second_closest_idx(i)
            if self.type == 'NUM':
                second_idx = first_idx
                first_idx = i

            furthest_idx = self.find_furthest_idx(i)
            target_embed = self.numeral_vec_embed_i if self.type == 'NUM' else self.numeral_embed_i
            norm_first = np.linalg.norm(self.numeral_embed_i[i] - target_embed[first_idx])
            norm_furthest = np.linalg.norm(self.numeral_embed_i[i] - target_embed[furthest_idx])
            # print('{} , {}:{}, {}:{}'.format(self.numerals[i], self.numerals[first_idx], norm_first, self.numerals[furthest_idx], norm_furthest))
            # print(self.numeral_embed_i[i])
            # print(target_embed[first_idx])
            # print(target_embed[furthest_idx])
            if norm_first < norm_furthest:
                right += 1

        if self.varbose:
            print("{}-BC: {}".format(self.type, right / len(self.numerals)))
        res = right / len(self.numerals)
        return res

    def evaluate_avg_rank(self):
        rank = 0
        for i in range(len(self.numerals)):
            first_idx, second_idx = self.find_first_and_second_closest_idx(i)
            if self.type == 'NUM':
                second_idx = first_idx
                first_idx = i

            target_embed = self.numeral_vec_embed_i if self.type == 'NUM' else self.numeral_embed_i
            embeds_i = self.numeral_embed_i[i] - target_embed
            res_norm = np.linalg.norm(embeds_i, axis=1)
            res = np.argsort(res_norm)
            # print(res_norm)
            # print(res)
            rank_i = np.where(res == first_idx)[0]
            # print(first_idx)
            rank += rank_i

        rank = rank / len(self.numerals)
        return rank[0]


    def evaluate_all(self):
        OVA, OVAR, SC, BC, AVGR = self.evaluate_ova(), self.evaluate_ova_r(), self.evaluate_sc(), self.evaluate_bc(), self.evaluate_avg_rank()

        # if self.varbose:
        #     print('{}-OVA: {}, {}-OVAR: {}, {}-SC: {}, {}-BC: {}'.format(
        #         self.type, OVA, self.type, OVAR, self.type, SC, self.type, BC
        #     ))

        return OVA, OVAR, SC, BC, AVGR