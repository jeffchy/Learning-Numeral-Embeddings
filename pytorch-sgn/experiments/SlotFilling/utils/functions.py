# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-14 11:08:45
from __future__ import print_function
from __future__ import absolute_import
import sys
import numpy as np
import pickle
import os
import math
import torch as t
import glob
import random
import codecs

random.seed(0)

# sys.path.append('../../../')
def weighted_log(x):
    """
    :param values: values (prototypes)
    :return:  weights for each embeddings

    Linear interpolation
    """
    if x > 1:
        x = np.log(x) + 1
    elif x < -1:
        x = -1 * (np.log(np.abs(x)) + 1)

    return x

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, feature_alphabets, label_alphabet, number_normalized, max_sent_length, sentence_classification=False, split_token='\t', char_padding_size=-1, char_padding_symbol = '</pad>', portion = 1.0):
    feature_num = len(feature_alphabets)
    in_lines = open(input_file,'r', encoding="utf8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    features = []
    chars = []
    labels = []
    word_Ids = []
    feature_Ids = []
    char_Ids = []
    label_Ids = []

    ## if sentence classification data format, splited by \t
    if sentence_classification:
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split(split_token)
                sent = pairs[0]
                if sys.version_info[0] < 3:
                    sent = sent.decode('utf-8')
                original_words = sent.split()
                for word in original_words:
                    # word = word.encode('utf-8').decode('utf-8')
                    words.append(word)
                    if number_normalized:
                        word = normalize_word(word)
                    word_Ids.append(word_alphabet.get_index(word))
                    ## get char
                    char_list = []
                    char_Id = []
                    for char in word:
                        char_list.append(char)
                    if char_padding_size > 0:
                        char_number = len(char_list)
                        if char_number < char_padding_size:
                            char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                        assert(len(char_list) == char_padding_size)
                    for char in char_list:
                        char_Id.append(char_alphabet.get_index(char))
                    chars.append(char_list)
                    char_Ids.append(char_Id)

                label = pairs[-1]
                label_Id = label_alphabet.get_index(label)
                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                ## combine together and return, notice the feature/label as different format with sequence labeling task
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)):
                    instence_texts.append([words, feat_list, chars, label])
                    instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
                words = []
                features = []
                chars = []
                char_Ids = []
                word_Ids = []
                feature_Ids = []
                label_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            instence_texts.append([words, feat_list, chars, label])
            instence_Ids.append([word_Ids, feat_Id, char_Ids,label_Id])
            words = []
            features = []
            chars = []
            char_Ids = []
            word_Ids = []
            feature_Ids = []
            label_Ids = []

    else:
    ### for sequence labeling data format i.e. CoNLL 2003
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if sys.version_info[0] < 3:
                    word = word.decode('utf-8')
                words.append(word)
                if number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                labels.append(label)

                word_Ids.append(word_alphabet.get_index(word))
                label_Ids.append(label_alphabet.get_index(label))

                ## get features
                feat_list = []
                feat_Id = []
                for idx in range(feature_num):
                    feat_idx = pairs[idx+1].split(']',1)[-1]
                    feat_list.append(feat_idx)
                    feat_Id.append(feature_alphabets[idx].get_index(feat_idx))
                features.append(feat_list)
                feature_Ids.append(feat_Id)
                ## get char
                char_list = []
                char_Id = []
                for char in word:
                    char_list.append(char)
                if char_padding_size > 0:
                    char_number = len(char_list)
                    if char_number < char_padding_size:
                        char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                    assert(len(char_list) == char_padding_size)
                else:
                    ### not padding
                    pass
                for char in char_list:
                    char_Id.append(char_alphabet.get_index(char))
                chars.append(char_list)
                char_Ids.append(char_Id)
            else:
                if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
                    instence_texts.append([words, features, chars, labels])
                    instence_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
                words = []
                features = []
                chars = []
                labels = []
                word_Ids = []
                feature_Ids = []
                char_Ids = []
                label_Ids = []
        if (len(words) > 0) and ((max_sent_length < 0) or (len(words) < max_sent_length)) :
            instence_texts.append([words, features, chars, labels])
            instence_Ids.append([word_Ids, feature_Ids, char_Ids,label_Ids])
            words = []
            features = []
            chars = []
            labels = []
            word_Ids = []
            feature_Ids = []
            char_Ids = []
            label_Ids = []

    assert len(instence_Ids) == len(instence_texts)
    if portion < 1.0:
        temp = list(zip(instence_Ids, instence_texts))
        sampled_temp = random.sample(temp, int(len(instence_texts) *portion ))
        instence_Ids, instence_texts = zip(*sampled_temp)
        instence_Ids, instence_texts = list(instence_Ids), list(instence_texts)
    return instence_texts, instence_Ids

def to_numeral(token):
    try:
        num = np.float32(token)
        if num == np.float32('inf') or num == np.float32('-inf') or math.isnan(num):
            return False, None
        return True, num

    except ValueError:
        return False, None

def get_prototype_embedding(num, trained_prototypes, alpha, log_space=False):


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

    prototypes, prototypes_embeddings = trained_prototypes['prototypes'], trained_prototypes['i_embedding']
    prototypes2vec_i = trained_prototypes['i_embedding']
    if log_space:
        # if prototype already in log space, no need to transform
        transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).view(-1, 1)
    else:
        transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).apply_(weighted_log).view(-1, 1)

    prototype_weights = get_numeral_embed_weights_batch(t.tensor(t.from_numpy(np.array([num])),dtype=t.float),transformed_protp, alpha=alpha)
    vec = t.matmul(prototype_weights.transpose(0,1),t.from_numpy(prototypes2vec_i))

    return vec


def build_pretrain_embedding_prorotype(embedding_path, preprocess_dir, epoch, alpha, word_alphabet, embedd_dim=100, norm=True, log_space=False):
    embedd_dict = dict()

    assert embedding_path != None
    assert preprocess_dir != None

    file_paths = glob.glob(embedding_path +'/trained_prototypes_epoch{}_*.dat'.format(epoch))
    assert len(file_paths) == 1

    trained_prototypes = pickle.load(open(file_paths[0], 'rb'))
    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_i_epoch{}.dat'.format(epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]

    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_represent = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        flag, num = to_numeral(word)
        if word in word2idx:
            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif flag:
            vec = get_prototype_embedding(num, trained_prototypes, alpha, log_space=log_space)
            numeral_represent += 1

        elif word.lower() in word2idx:

            vec = idx2vec[word2idx[word.lower()]]
            case_match += 1
        else:
            vec = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

        if norm:
            pretrain_emb[index, :] = norm2one(vec)
        else:
            pretrain_emb[index, :] = vec

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_represent:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_represent ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_GMM(embedding_path, preprocess_dir, epoch, word_alphabet, embedd_dim=100, norm=True, log_space=False):
    embedd_dict = dict()

    assert embedding_path != None
    assert preprocess_dir != None

    file_paths = glob.glob(embedding_path +'/trained_gmms_epoch{}_*.dat'.format(epoch))
    gmms_path = embedding_path.split('/')[-1]

    assert len(file_paths) == 1

    gmm_parent_path = 'gmm' if not log_space else 'gmm_log'

    trained_prototypes = pickle.load(open(file_paths[0], 'rb'))
    try:
        gmm = pickle.load(open(os.path.join(os.path.join(preprocess_dir, gmm_parent_path), gmms_path + '.dat'), 'rb'))
    except FileNotFoundError:
        gmm = pickle.load(open(os.path.join(os.path.join(preprocess_dir, gmm_parent_path), gmms_path[:-2] + '.dat'), 'rb'))


    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_i_epoch{}.dat'.format(epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))

    prototypes2vec_i = trained_prototypes['i_embedding']

    embedd_dim = idx2vec.shape[1]

    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_represent = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        flag, num = to_numeral(word)

        if word in word2idx:
            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif flag:
            if log_space:
                num = weighted_log(num)

            prototype_weights = gmm.predict_proba([[num]])
            vec = np.matmul(prototype_weights, prototypes2vec_i)
            numeral_represent += 1

        elif word.lower() in embedd_dict:

            vec = idx2vec[word2idx[word.lower()]]
            case_match += 1
        else:
            vec = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

        if norm:
            pretrain_emb[index, :] = norm2one(vec)
        else:
            pretrain_emb[index, :] = vec

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_represent:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_represent ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


def build_pretrain_embedding_token(embedding_path, preprocess_dir, epoch, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()

    assert embedding_path != None
    assert preprocess_dir != None

    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_i_epoch{}.dat'.format(epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]

    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_perfect_match = 0
    numeral_not_match = 0

    not_match = 0
    for word, index in word_alphabet.iteritems():
        flag, num = to_numeral(word)

        if word in word2idx:
            if flag:
                numeral_perfect_match += 1

            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif word.lower() in embedd_dict:

            vec = idx2vec[word2idx[word.lower()]]
            case_match += 1

        else:
            if flag:
                numeral_not_match += 1
                vec = idx2vec[word2idx['<UNK_N>']]
            else:
                vec = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

        if norm:
            pretrain_emb[index, :] = norm2one(vec)
        else:
            pretrain_emb[index, :] = vec

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_perfect_match:%s, numeral_not_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_perfect_match, numeral_not_match ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_fixed(embedding_path, preprocess_dir, epoch, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()

    assert embedding_path != None
    assert preprocess_dir != None

    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_i_epoch{}.dat'.format(epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]

    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_perfect_match = 0
    numeral_not_match = 0

    not_match = 0
    for word, index in word_alphabet.iteritems():
        flag, num = to_numeral(word)

        if word in word2idx:
            if flag:
                numeral_perfect_match += 1

            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif word.lower() in embedd_dict:

            vec = idx2vec[word2idx[word.lower()]]
            case_match += 1

        else:
            if flag:
                numeral_not_match += 1
                num = weighted_log(num)
                vec = np.ones([1, embedd_dim])
                vec[0][0] = num
                vec /= (2 * embedd_dim)

            else:
                vec = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

        if norm:
            pretrain_emb[index, :] = norm2one(vec)
        else:
            pretrain_emb[index, :] = vec

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_perfect_match:%s, numeral_not_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_perfect_match, numeral_not_match ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


def build_pretrain_embedding_LSTM(embedding_path, preprocess_dir, epoch, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()

    assert embedding_path != None
    assert preprocess_dir != None

    def convert_digit_to_tensor(numeral_str):
        represent = t.zeros(max_token_len, len(idx2digit))
        assert len(numeral_str) <= max_token_len
        for i in range(len(numeral_str)):
            digit = numeral_str[i]
            idx = digit2idx[digit]
            represent[i][idx] = 1

        return represent


    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_i_epoch{}.dat'.format(epoch)), 'rb'))
    LSTM_model_path = os.path.join(embedding_path, 'sgns_epoch{}.pt'.format(epoch))
    LSTM_params = t.load(LSTM_model_path, map_location='cpu')
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]


    digital_RNN_i = t.nn.LSTM(14, idx2vec.shape[1], 1, batch_first=True)

    digital_RNN_i.bias_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.bias_hh_l0'])
    digital_RNN_i.bias_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.bias_ih_l0'])
    digital_RNN_i.weight_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.weight_hh_l0'])
    digital_RNN_i.weight_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.weight_ih_l0'])

    idx2digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '+', 'e']
    digit2idx = {idx2digit[i]: i for i in range(len(idx2digit))}
    max_token_len = 20  # should be equal to

    def get_LSTM_embedding(num, digital_RNN_i):
        temp = convert_digit_to_tensor(str(num)).view(1, 20, 14)
        _, (hn, cn) = digital_RNN_i(temp)
        e = hn.squeeze().detach().numpy()

        return e


    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_represent = 0

    not_match = 0
    for word, index in word_alphabet.iteritems():
        flag, num = to_numeral(word)
        if word in word2idx:
            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif flag:
            vec = get_LSTM_embedding(num, digital_RNN_i)
            numeral_represent += 1

        elif word.lower() in embedd_dict:

            vec = idx2vec[word2idx[word.lower()]]
            case_match += 1
        else:
            vec = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

        if norm:
            pretrain_emb[index, :] = norm2one(vec)
        else:
            pretrain_emb[index, :] = vec

    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_represent:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_represent ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    alphabet_size = word_alphabet.size()
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            if sys.version_info[0] < 3:
                first_col = tokens[0].decode('utf-8')
            else:
                first_col = tokens[0]
            embedd_dict[first_col] = embedd
    return embedd_dict, embedd_dim

if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
