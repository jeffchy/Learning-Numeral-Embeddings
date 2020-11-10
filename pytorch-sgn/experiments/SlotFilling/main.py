# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-01-14 16:09:16

from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
# sys.path.append('..')
# sys.path.append('utils')
# sys.path.append('model')

from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data
from copy import deepcopy

try:
    import cPickle as pickle
except ImportError:
    import pickle




def data_initialization(data):
    # data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    # TODO: why? defualt adding it in to the alphabet?
    # when we use GMM
    # if data.mode != 'none':
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.build_alphabet(data.test_augment_dir)

    data.fix_alphabet()



def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    if sentence_classification:
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0] ## =batch_size
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred)==len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer



def evaluate(data, model, name, nbest=None):
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'test_augment':
        instances = data.test_augment_Ids
    elif name == 'test_harder':
        instances = data.test_harder_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, False, data.sentence_classification)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        # print("tag:",tag_seq)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores
    return speed, acc, p, r, f, pred_results, pred_scores


def batchify_with_label(input_batch_list, gpu, if_train=True, sentence_classification=False):
    if sentence_classification:
        return batchify_sentence_classification_with_label(input_batch_list, gpu, if_train)
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train)


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def batchify_sentence_classification_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, feature_num), each sentence has one set of feature
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size,), each sentence has one set of feature

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size,), ... ] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, )
            mask: (batch_size, max_sent_len)
    """

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]    
    feature_num = len(features[0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, ), requires_grad =  if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    label_seq_tensor = torch.LongTensor(labels)
    # exit(0)
    for idx, (seq,  seqlen) in enumerate(zip(words,  word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def evaluate_and_print(data, model, name):
    test_start = time.time()
    speed, acc, p, r, f, _, _ = evaluate(data, model, name)
    test_finish = time.time()
    test_cost = test_finish - test_start
    if data.seg:
        print("%s: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (name,
        test_cost, speed, acc, p, r, f))
    else:
        print("%s: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (name, test_cost, speed, acc))

def evaluate_and_print_return(data, model, name):
    dev_finish = time.time()
    speed, acc, p, r, f, _, _ = evaluate(data, model, name)
    test_finish = time.time()
    test_cost = test_finish - dev_finish
    if data.seg:
        print("%s: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (name, test_cost, speed, acc, p, r, f))
    else:
        print("%s: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (name, test_cost, speed, acc))
    return acc, p, r, f

def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    # loss_function = nn.NLLLoss()
    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)
    best_dev = -10
    best_model = None

    dev_info = {'acc':[],'p':[],'f':[],'r':[]}
    test_info = {'acc':[],'p':[],'f':[],'r':[]}

    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        print("Shuffle: first input word list:", data.train_Ids[0][0])
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True, data.sentence_classification)
            instance_count += 1
            loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask, data.sentence_classification)
            right_token += right
            whole_token += whole
            # print("loss:",loss.item())
            sample_loss += loss.item()
            total_loss += loss.item()
            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _,_ = evaluate(data, model, "dev")
        dev_info['acc'].append(acc)
        dev_info['p'].append(p)
        dev_info['r'].append(r)
        dev_info['f'].append(f)

        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = data.model_dir +'.'+ str(idx) + ".model"

            best_model_dict = deepcopy(model.state_dict())
            best_dev = current_score

        # ## decode test
        if idx % 10 == 0:
            evaluate_and_print(data, model, 'test')
            evaluate_and_print(data, model, 'test_augment')
            evaluate_and_print(data, model, 'test_harder')

        gc.collect()

    # add test for best model
    print("======BEST MODEL TEST======")
    print("Save current best model in file:", model_name)
    torch.save(best_model_dict, model_name)
    model.load_state_dict(best_model_dict)

    acc, p, r, f = evaluate_and_print_return(data, model, 'test')
    acc_a, p_a, r_a, f_a = evaluate_and_print_return(data, model, 'test_augment')
    acc_h, p_h, r_h, f_h = evaluate_and_print_return(data, model, 'test_harder')
    print("======BEST DEV=======: {}".format(best_dev))

    return best_dev, acc, p, r, f, acc_a, p_a, r_a, f_a, acc_h, p_h, r_h, f_h, dev_info

def load_model_decode(data, name):
    print("Load Model from file: ", data.model_dir)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)

    # model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores


def parse_args_to_data(args, data):


    data.train_embedding = args.train_embedding
    data.train_dir = args.train
    data.dev_dir = args.dev
    data.test_dir = args.test
    data.test_augment_dir = args.test_augment
    data.test_harder_dir = args.test_harder
    data.model_dir = args.savemodel
    data.mode = args.mode
    data.use_crf = args.use_crf
    data.embed_epoch = args.embed_epoch
    data.alpha = args.alpha
    data.log_space = args.log_space
    data.train_portion = args.train_portion

    data.seg = args.seg
    data.status = args.status.lower()
    data.save_model_dir = args.savemodel
    data.HP_gpu = torch.cuda.is_available()

    data.number_normalized = False
    data.norm_word_emb = False
    data.norm_char_emb = False
    data.use_char = False
    data.word_emb_dim = 50
    data.char_emb_dim = 30
    data.word_seq_feature = 'LSTM'
    data.char_seq_feature = 'CNN'
    data.word_emb_dir = args.word_emb_dir
    data.preprocess_dir = args.preprocess_dir
    data.optimizer = 'SGD'
    data.HP_iteration = args.iteration
    data.batch_size = 10
    data.ave_batch_loss = False
    data.cnn_layer = 4
    data.char_hidden_dim = 50
    data.hidden_dim = 200
    data.dropout = 0.5
    data.lstm_layer = 1
    data.bilstm = True
    data.learning_rate = 0.015
    data.lr_decay = 0.03

    #data.lr_decay = 0.05
    data.momentum = 0
    data.l2 = 1e-8

    # args = parser.parse_args()
    # data = Data()
    # data.HP_gpu = torch.cuda.is_available()
    # data.read_config(args.config)
    data.show_data_summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--word_emb_dir', help='Embedding for words', default=None)
    parser.add_argument('--preprocess_dir', help='preprocess_dir', default=None)
    parser.add_argument('--train_embedding', type=bool, default=False)
    parser.add_argument('--run_name', type=str, default='run0')
    parser.add_argument('--mode', default='none')
    parser.add_argument('--use_crf', type=bool, default=True)
    parser.add_argument('--log_space', action='store_true', help='add if train SOM and GMM in log space')

    parser.add_argument('--charemb', help='Embedding for chars', default='None')
    parser.add_argument('--savemodel', default="leyan_data/model/leyanbug/saved_model")
    parser.add_argument('--embed_epoch', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.0)

    # parser.add_argument('--savedset', help='Dir of saved data setting')

    parser.add_argument('--iteration', type=int, default=100)
    parser.add_argument('--train_portion', type=float, default=0.1)

    # old !!!
    # parser.add_argument('--dev', default="leyan_data/test.txt")# reverse train, dev
    # parser.add_argument('--test', default="leyan_data/dev.txt")

    # new dataset, harder
    # parser.add_argument('--train', default="leyan_data/leyan_data_5fold/train0.txt")
    # parser.add_argument('--dev', default="leyan_data/leyan_data_5fold/dev0.txt") # not reverse train, dev
    # parser.add_argument('--test', default="leyan_data/leyan_data_5fold/test.txt")
    # parser.add_argument('--test_augment', default="leyan_data/leyan_data_5fold/test_augment.txt")

    # all domain dataset
    # parser.add_argument('--train', default="leyan_data/leyan_all_domain/train_domain.txt")
    # parser.add_argument('--dev', default="leyan_data/leyan_all_domain/dev_domain.txt") # not reverse train, dev
    # parser.add_argument('--test', default="leyan_data/leyan_all_domain/test_domain.txt")

    # all leyan bug dataset
    parser.add_argument('--train', default="leyan_data/leyan_bug/train.txt")
    parser.add_argument('--dev', default="leyan_data/leyan_bug/dev.txt") # not reverse train, dev
    parser.add_argument('--test', default="leyan_data/leyan_bug/test.txt")
    parser.add_argument('--test_augment', default="leyan_data/leyan_bug/test_augment_5x.txt")
    parser.add_argument('--test_harder', default="leyan_data/leyan_bug/test_harder.txt")

    parser.add_argument('--seg', type=bool, default=True)
    # parser.add_argument('--raw')
    # parser.add_argument('--loadmodel')
    # parser.add_argument('--output')
    args = parser.parse_args()
    data = Data()

    best_dev_all = -100
    best_res = None
    best_epoch = -1

    if not os.path.exists('leyan_data/model/{}'.format(args.run_name)):
        os.mkdir('leyan_data/model/{}'.format(args.run_name))

    for i in range(3):
    # for i in range(1):

        # 7 8 9 10
        best_dev_list = []
        dev_info_list = []

        acc_list, p_list, r_list, f_list = [], [], [], []
        acc_a_list, p_a_list, r_a_list, f_a_list = [], [], [], []
        acc_h_list, p_h_list, r_h_list, f_h_list = [], [], [], []

        data = Data()
        embed_epoch = int(i+8)
        # embed_epoch = int(5)

        args.embed_epoch = embed_epoch
        parse_args_to_data(args, data)

        name = args.word_emb_dir.split('/')[-1] if args.word_emb_dir else 'no_embed_train_{}'.format(args.train_embedding)

        # for s in range(3):
        for s in range(3):

            seed_num = s
            random.seed(seed_num)
            torch.manual_seed(seed_num)
            np.random.seed(seed_num)

            data.model_dir = "leyan_data/model/{}/{}-s{}-a{}-e{}".format(args.run_name, name, s, args.alpha, args.embed_epoch)
            status = data.status.lower()

            print("Seed num:",seed_num)

            if status == 'train':
                print("MODEL: train")
                data_initialization(data)
                data.generate_instance('train')
                data.generate_instance('dev')
                data.generate_instance('test')
                data.generate_instance('test_augment')
                data.generate_instance('test_harder')
                data.build_pretrain_emb()
                best_dev, acc, p, r, f, acc_a, p_a, r_a, f_a, acc_h, p_h, r_h, f_h, dev_info = train(data)
                dev_info_list.append(dev_info)
                best_dev_list.append(best_dev)

                acc_list.append(acc)
                p_list.append(p)
                r_list.append(r)
                f_list.append(f)

                acc_a_list.append(acc_a)
                p_a_list.append(p_a)
                r_a_list.append(r_a)
                f_a_list.append(f_a)

                acc_h_list.append(acc_h)
                p_h_list.append(p_h)
                r_h_list.append(r_h)
                f_h_list.append(f_h)

            elif status == 'decode':
                print("MODEL: decode")
                data.load(data.dset_dir)
                data.read_config(args.config)
                print(data.raw_dir)
                # exit(0)
                data.show_data_summary()
                data.generate_instance('raw')
                print("nbest: %s"%(data.nbest))
                decode_results, pred_scores = load_model_decode(data, 'raw')
                if data.nbest and not data.sentence_classification:
                    data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
                else:
                    data.write_decoded_results(decode_results, 'raw')
            else:
                print("Invalid argument! Please use valid arguments! (train/test/decode)")


        res_json = {
            'dev_info_list': dev_info_list,
            'best_dev_list': best_dev_list,
            'acc_list': acc_list,
            'p_list': p_list,
            'f_list': f_list,
            'r_list': r_list,
            'acc_a_list': acc_a_list,
            'p_a_list': p_a_list,
            'f_a_list': f_a_list,
            'r_a_list': r_a_list,
            'acc_h_list': acc_h_list,
            'p_h_list': p_h_list,
            'f_h_list': f_h_list,
            'r_h_list': r_h_list,
        }

        print("mean best dev: {}, mean acc: {}, mean p: {}, mean r: {}, mean f: {} for Epoch: {}".format(
            np.mean(best_dev_list),
            np.mean(acc_list),
            np.mean(p_list),
            np.mean(r_list),
            np.mean(f_list),
            embed_epoch,
        ))

        print("mean best dev: {}, augment mean acc: {}, mean p: {}, mean r: {}, mean f: {} for Epoch: {}".format(
            np.mean(best_dev_list),
            np.mean(acc_a_list),
            np.mean(p_a_list),
            np.mean(r_a_list),
            np.mean(f_a_list),
            embed_epoch,
        ))

        print("mean best dev: {}, harder mean acc: {}, mean p: {}, mean r: {}, mean f: {} for Epoch: {}".format(
            np.mean(best_dev_list),
            np.mean(acc_h_list),
            np.mean(p_h_list),
            np.mean(r_h_list),
            np.mean(f_h_list),
            embed_epoch,
        ))


        mean_best_dev = np.mean(best_dev_list)
        if mean_best_dev > best_dev_all:
            best_epoch = embed_epoch
            best_res = deepcopy(res_json)
            best_dev_all = mean_best_dev

    print("res *${}$*".format(best_res))
    print("best epoch: {}".format(best_epoch))
    print("mean best dev: {}, mean acc: {}, mean p: {}, mean r: {}, mean f: {}".format(
        np.mean(best_res['best_dev_list']),
        np.mean(best_res['acc_list']),
        np.mean(best_res['p_list']),
        np.mean(best_res['r_list']),
        np.mean(best_res['f_list']),
    ))
