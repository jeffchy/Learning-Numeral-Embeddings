import pickle
import numpy as np
import pandas as pd
import sys
import torch as t
from tqdm import tqdm_notebook
sys.path.append('../../numeral_context/')
sys.path.append('../../../')
from som.intopolate import weighted_log
from utils.number_handler import to_numeral

def load_prototype(numerals,
                   trained_prototypes,
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

    prototype_weights = get_numeral_embed_weights_batch(t.tensor(t.from_numpy(numerals), dtype=t.float),
                                                        transformed_protp, alpha=alpha)
    numeral_embed_i = t.matmul(prototype_weights.transpose(0, 1), t.from_numpy(prototypes2vec_i))
    numeral_embed_o = t.matmul(prototype_weights.transpose(0, 1), t.from_numpy(prototypes2vec_o))

    return numeral_embed_i, numeral_embed_o


def load_fixed(numerals, embed_size):

    def get_numeral_embed_batch(numerals):
        """
        :param numerals: tensor of numerals!
        :return: embedding matrix of numerals
        """
        embedding_size = embed_size
        numerals.apply_(weighted_log)
        embed = t.ones((embedding_size, len(numerals)))
        embed[0] = numerals
        embed = embed.transpose(0, 1)
        embed /= (embedding_size * 2)

        return embed  # [num_of_numerals x embedding_size]

    numeral_embed_i = get_numeral_embed_batch(t.tensor(t.from_numpy(numerals),dtype=t.float)) # (93, 150)
    numeral_embed_o = get_numeral_embed_batch(t.tensor(t.from_numpy(numerals),dtype=t.float))   # (93, 150)

    return numeral_embed_i, numeral_embed_o



def load_TOKEN(numerals,
               idx2vec_i,
               idx2vec_o,
               word2idx,
               ):
    # CAUTION, need to reload the word2idx, nc ...

    numerals_str = [str(i) for i in numerals]
    numeral_embed_i = np.zeros((len(numerals), idx2vec_i.shape[1]))
    numeral_embed_o = np.zeros((len(numerals), idx2vec_i.shape[1]))


    UNK_idxs = []

    for i in range(len(numerals_str)):
        try:
            numeral_embed_i[i] = idx2vec_i[word2idx[numerals_str[i]]]
        except:
            # print('oov:', numerals_str[i])
            UNK_idxs.append(i)
            numeral_embed_i[i] = idx2vec_i[word2idx['<UNK_N>']]

    for i in range(len(numerals_str)):
        try:
            numeral_embed_o[i] = idx2vec_o[word2idx[numerals_str[i]]]

        except:
            # print('oov:', numerals_str[i])
            numeral_embed_o[i] = idx2vec_o[word2idx['<UNK_N>']]

    return numeral_embed_i, numeral_embed_o



def load_GMM(numerals,
             trained_prototypes,
             gmm,
             log_space=False):

    if log_space:
        prototype_weights = gmm.predict_proba(np.array([weighted_log(x) for x in numerals]).reshape(-1, 1))
    else:
        prototype_weights = gmm.predict_proba(numerals.reshape(-1, 1))

    prototypes2vec_i = trained_prototypes['i_embedding']
    prototypes2vec_o = trained_prototypes['o_embedding']
    numeral_embed_i = t.from_numpy(np.matmul(prototype_weights, prototypes2vec_i))  # (93, 150)
    numeral_embed_o = t.from_numpy(np.matmul(prototype_weights, prototypes2vec_o))  # (93, 150)
    numeral_embed_i = np.array(numeral_embed_i)
    numeral_embed_o = np.array(numeral_embed_o)

    return numeral_embed_i, numeral_embed_o


def load_LSTM(numerals,
              LSTM_model_path, embed_size):
    model_dict = t.load(LSTM_model_path, map_location='cpu')

    digital_RNN_i = t.nn.LSTM(14, embed_size, 1, batch_first=True)
    digital_RNN_o = t.nn.LSTM(14, embed_size, 1, batch_first=True)

    new_dict = digital_RNN_i.state_dict()
    new_dict['bias_hh_l0'] = model_dict['embedding.digital_RNN_i.bias_hh_l0']
    new_dict['bias_ih_l0'] = model_dict['embedding.digital_RNN_i.bias_ih_l0']
    new_dict['weight_hh_l0'] = model_dict['embedding.digital_RNN_i.weight_hh_l0']
    new_dict['weight_ih_l0'] = model_dict['embedding.digital_RNN_i.weight_ih_l0']
    digital_RNN_i.load_state_dict(new_dict)


    new_dict = digital_RNN_o.state_dict()
    new_dict['bias_hh_l0'] = model_dict['embedding.digital_RNN_o.bias_hh_l0']
    new_dict['bias_ih_l0'] = model_dict['embedding.digital_RNN_o.bias_ih_l0']
    new_dict['weight_hh_l0'] = model_dict['embedding.digital_RNN_o.weight_hh_l0']
    new_dict['weight_ih_l0'] = model_dict['embedding.digital_RNN_o.weight_ih_l0']
    digital_RNN_o.load_state_dict(new_dict)


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

    numeral_embed_i = np.zeros((len(numerals), embed_size))
    numeral_embed_o = np.zeros((len(numerals), embed_size))

    counter = 0
    for i in numerals:
        temp = convert_digit_to_tensor(str(i)).view(1, 20, 14)
        _, (hn, cn) = digital_RNN_i(temp)
        e = hn.squeeze().detach().numpy()
        numeral_embed_i[counter] = e
        counter += 1

    counter = 0
    for i in numerals:
        temp = convert_digit_to_tensor(str(i)).view(1, 20, 14)
        _, (hn, cn) = digital_RNN_o(temp)
        e = hn.squeeze().detach().numpy()
        numeral_embed_o[counter] = e
        counter += 1

    numeral_embed_i = np.array(numeral_embed_i)
    numeral_embed_o = np.array(numeral_embed_o)

    return numeral_embed_i, numeral_embed_o


