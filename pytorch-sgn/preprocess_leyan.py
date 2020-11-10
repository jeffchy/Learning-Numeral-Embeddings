import os
import codecs
import pickle
import argparse
from utils.number_handler import number_handler, is_numeral, to_numeral, to_numeral_if_possible
import random
from som.som import MiniSom as SOM
from copy import copy
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
mpl.use('Agg') # save fig on server
import matplotlib.pyplot as plt
import numpy as np


from preprocess import parse_args
from preprocess import Preprocess, PreprocessNumeralAsNumeralLSTM, PreprocessNumeralAsToken, PreprocessNumeralAsTokenUnkNumeral, PreprocessNumeralAsUnkNumeral

class Preprocess_Leyan(Preprocess):
    def __init__(self, window=5, unk='<UNK_W>', save_dir='./preprocess/'):
        Preprocess.__init__(self, window, unk, save_dir)

    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import re
        import glob

        step = 0

        # support for directory mode, saving into 1 single file dump
        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:
                s = file.read()
                sent = s.strip().split()
                sent_filtered = []
                for token in sent:

                    if is_numeral(token):
                        # ['-32.000'] to ['-32']
                        # prevent '-haha' like token, double check
                        number = str(to_numeral(token))
                        self.nc[number] = self.nc.get(number, 0) + 1
                        sent_filtered.append(number)

                    else:
                        self.wc[token] = self.wc.get(token, 0) + 1
                        sent_filtered.append(token)

                output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")

class Preprocess_Leyan_LSTM(PreprocessNumeralAsNumeralLSTM):
    def __init__(self, window=5, unk='<UNK_W>', save_dir='./preprocess/'):
        PreprocessNumeralAsNumeralLSTM.__init__(self, window, unk, save_dir)

    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import glob

        # support for directory mode, saving into 1 single file dump
        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:
                s = file.read()
                sent = s.strip().split()
                sent_filtered = []
                for token in sent:

                    if is_numeral(token):
                        # ['-32.000'] to ['-32']
                        # prevent '-haha' like token, double check
                        number = str(to_numeral(token))
                        self.nc[number] = self.nc.get(number, 0) + 1
                        sent_filtered.append(token) # for LSTM change to original token

                    else:
                        self.wc[token] = self.wc.get(token, 0) + 1
                        sent_filtered.append(token)

                output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")

class PreprocessNumeralAsTokenUnkNumeral_Leyan(PreprocessNumeralAsTokenUnkNumeral):
    def __init__(self, window=5, unk='<UNK_W>', unk_n='<UNK_N>', save_dir='./preprocess/'):
        PreprocessNumeralAsTokenUnkNumeral.__init__(self, window, unk, unk_n ,save_dir)

    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import glob

        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:
                s = file.read()
                sent = s.strip().split()
                sent_filtered = []
                for token in sent:

                    if is_numeral(token):
                        # ['-32.000'] to ['-32']
                        # prevent '-haha' like token, double check
                        self.wc[token] = self.wc.get(token, 0) + 1
                        sent_filtered.append(token)

                    else:
                        self.wc[token] = self.wc.get(token, 0) + 1
                        sent_filtered.append(token)

                output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")



if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    assert args.scheme in ['numeral_as_numeral', 'numeral_as_numeral_lstm','numeral_as_token', 'numeral_as_unk_numeral', 'numeral_as_token_unk_numeral']
    assert args.mode in ['train_som', 'train_gmm', 'build', 'convert', 'all']

    if args.scheme == 'numeral_as_numeral':

        preprocess = Preprocess_Leyan(
            window=args.window,
            unk=args.unk_w,
            save_dir=os.path.join(args.save_dir, 'NumeralAsNumeral')
        )

    if args.scheme == 'numeral_as_numeral_lstm':

        preprocess = Preprocess_Leyan_LSTM(
            window=args.window,
            unk=args.unk_w,
            save_dir=os.path.join(args.save_dir, 'NumeralAsNumeralLSTM')
        )

    if args.scheme == 'numeral_as_token_unk_numeral':

        preprocess = PreprocessNumeralAsTokenUnkNumeral_Leyan(
            window=args.window,
            unk=args.unk_w,
            unk_n=args.unk_n,
            save_dir=os.path.join(args.save_dir, 'NumeralAsTokenUnkNumeral')
        )

    if args.mode == 'build':
        preprocess.filter_and_count(args.corpus, os.path.join(preprocess.save_dir, args.filtered))
        preprocess.build_vocab(max_vocab=args.max_vocab)
        preprocess.dump_built_files()

    elif args.mode == 'train_som':
        preprocess.train_som(prototypes=args.num_prototypes, sigma=args.sigma, lr=args.lr, iters=args.num_iters, log_space=args.log_space)

    elif args.mode == 'train_gmm':
        preprocess.train_gmm(components=args.num_components, iters=args.gmm_iters, gmm_init_mode=args.gmm_init_mode, gmm_type=args.gmm_type, prototype_path=args.prototype_path, log_space=args.log_space)

    elif args.mode == 'convert':
        preprocess.convert(os.path.join(preprocess.save_dir, args.filtered), MAXDATA=args.MAXDATA)

    elif args.mode == 'all':

        preprocess.filter_and_count(args.corpus, os.path.join(preprocess.save_dir, args.filtered))
        preprocess.build_vocab(max_vocab=args.max_vocab)
        preprocess.dump_built_files()
        # if args.scheme == 'numeral_as_numeral':
        #     preprocess.train_som(prototypes=args.num_prototypes, sigma=args.sigma, lr=args.lr, iters=args.num_iters)
        #     preprocess.train_gmm(components=args.num_component, iters=args.num_iters)
        preprocess.convert(os.path.join(preprocess.save_dir, args.filtered), MAXDATA=args.MAXDATA)

    else:
        print("Invalid preprocess mode")




