# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
from utils.number_handler import number_handler, is_numeral, to_numeral, to_numeral_if_possible
import random
from som.som import MiniSom as SOM
from copy import copy
from sklearn.mixture import GaussianMixture
from utils.HardEMGMM import HardEMGaussianMixture
import matplotlib as mpl
mpl.use('Agg') # save fig on server
import matplotlib.pyplot as plt
import numpy as np
from som.intopolate import weighted_log

random_seed = 100
random.seed(random_seed)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./preprocess/', help="saving data directory path")

    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path / directory for the corpus need preprocess")
    parser.add_argument('--filtered', type=str, default='filtered.txt', help="filename for saving filtered corpus")
    parser.add_argument('--unk_w', type=str, default='<UNK_W>', help="UNK token word")
    parser.add_argument('--unk_n', type=str, default='<UNK_N>', help="UNK numeral")

    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab for tokens (not numerals)")
    parser.add_argument('--mode', type=str, default='build', help='build, train_gmm, train_som, convert, all')
    parser.add_argument('--MAXDATA', type=int, default=5000000, help='Max datapair for batch, 16GB ram set to 5000000, HPC can set larger')

    parser.add_argument('--scheme', type=str, default='numeral_as_numeral', help="scheme should be in ['numeral_as_numeral','numeral_as_unk_numeral', 'numeral_as_token, numeral_as_token_unk_numeral']")

    # train som part
    parser.add_argument('--num_prototypes', type=int, default=100, help='number of prototypes')
    parser.add_argument('--lr', type=float, default=0.3, help='learning rate of som')
    parser.add_argument('--sigma', type=float, default=3, help='sigma of gaussian neighbouring function of som')
    parser.add_argument('--num_iters', type=int, default=10000, help='number of iterations')

    # train gmm part
    parser.add_argument('--num_components', type=int, default=30, help='number of gmm components')
    parser.add_argument('--gmm_iters', type=int, default=1000, help='number of gmm iterations')
    parser.add_argument('--prototype_path', type=str, default=None, help='if given, not none, initialize it from prototype')
    parser.add_argument('--gmm_init_mode', type=str, default='rd', help='init mode of gmm prototypes, should be one of [rd, km, fp]')
    parser.add_argument('--gmm_type', type=str, default='soft', help='gmm type, soft EM GMM and hard EM GMM, should be in [soft, hard]')

    # saving dir name
    parser.add_argument('--saved_dir_name', type=str, default=None, help='saved dir name, eg. NumeralAsNumeral')

    # if log space?
    parser.add_argument('--log_space', action='store_true', help='if do SOM and GMM in log space')

    return parser.parse_args()

class   Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', save_dir='./preprocess/'):
        self.window = window
        self.unk = unk
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.nc = {}
        self.wc = {self.unk: 1}
        self.prototypes = None
        # self.data_for_SOM = {}

    def skipgram(self, sentence, i):
        iword = sentence[i]
        left = sentence[i - self.window: i]
        right = sentence[i + 1: i + 1 + self.window]
        return iword, left + right

    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import re
        import glob

        step = 0

        # the re for all possible token number
        RE_NUM = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))",
                            re.UNICODE)

        # support for directory mode, saving into 1 single file dump
        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:

                for line in file:
                    step += 1
                    if not step % 1000:
                        print("\n working on {}kth line in file {}".format(step // 1000, fpath))

                    line = line.strip()
                    if not line:
                        continue
                    sent = line.split()
                    sent_filtered = []
                    for token in sent:
                        # we treat word and numerals differently
                        # match numerals
                        res = re.findall(RE_NUM, token)
                        if res != []:
                            target = number_handler(token)
                            # we do not want nc to record ''
                            if target == '':
                                continue

                            if type(target) is list:
                                # ['u-32'] to ['u','-'.'32']
                                # [1997/07] to ['1997','/','7']

                                for i in target:
                                    if is_numeral(i):
                                        number = str(to_numeral(i))
                                        self.nc[number] = self.nc.get(number, 0) + 1
                                        sent_filtered.append(number)
                                    else:
                                        self.wc[i] = self.wc.get(i, 0) + 1
                                        sent_filtered.append(i)

                            elif is_numeral(target):
                                # ['-32.000'] to ['-32']
                                # prevent '-haha' like token, double check
                                number = str(to_numeral(target))
                                self.nc[number] = self.nc.get(number, 0) + 1
                                sent_filtered.append(number)

                        else:
                            self.wc[token] = self.wc.get(token, 0) + 1
                            sent_filtered.append(token)

                    output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")


    def build_vocab(self, max_vocab=20000):
        print("start building vocab")
        wc_nounk = copy(self.wc)
        wc_nounk.__delitem__(self.unk)
        self.idx2word = [self.unk]  + sorted(wc_nounk, key=wc_nounk.get, reverse=True)[:max_vocab - 1]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        print("building vocab done")


    def dump_built_files(self):
        pickle.dump(self.wc, open(os.path.join(self.save_dir, 'wc.dat'), 'wb'))
        pickle.dump(self.nc, open(os.path.join(self.save_dir, 'nc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.save_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.save_dir, 'idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.save_dir, 'word2idx.dat'), 'wb'))
        print("Dump done")


    def train_som(self, prototypes=10, sigma=0.03, lr=0.3, iters=10000, log_space=False):
        """
        :param nc_path: path under the save_directory
        :param prototypes: number of SOM neurons
        :param sigma:  sigma of SOM
        :param lr:  learning rate of SOM
        :return: None

        Train a simple SOM, and save it's neuron weights as prototypes, given numeral counts

        """
        nc = pickle.load(open(os.path.join(self.save_dir,'nc.dat'), 'rb'))

        # unfold and shuffle nc data
        data = []
        for k, v in nc.items():
            if to_numeral(k) == None:
                print('invalid numeral {}'.format(k))
            else:
                data += [[to_numeral(k)]]*v

        print('total number of different numerals: ', len(nc))
        print('total number of numeral samples: ', len(data))

        random.shuffle(data)
        if log_space:
            data = [[weighted_log(x[0])] for x in data]

        som = SOM(prototypes, 1, 1, sigma=sigma, learning_rate=lr, random_seed=random_seed)  # initialization

        print("Training SOMs...")
        # som.random_weights_init(data)
        som.train_random(data, iters)  # trains the SOM with 1000 iterations
        print("...Ready!")
        # win_map = som.win_map(data)
        self.prototypes = som.get_weights().reshape(prototypes) # nd array
        if log_space:
            som_save_dir = os.path.join(self.save_dir, 'som_log')
        else:
            som_save_dir = os.path.join(self.save_dir, 'som')

        if not os.path.exists(som_save_dir):
            os.makedirs(som_save_dir)
        print('prototypes: \n{}'.format(self.prototypes))
        pickle.dump(self.prototypes, open(os.path.join(som_save_dir, 'prototypes-{}-{}-{}.dat'.format(prototypes, sigma, lr)), 'wb'))
        print('...Saving Prototypes')

    def train_gmm(self, components=20, iters=100, gmm_init_mode='rd', gmm_type='soft', prototype_path=None, log_space=False):

        # print('<<<<<<<<<<INITIALIZING>>>>>>>>>> \n means: {} \n sigma: {}\n, weights: {}'.format(gmm.means_, gmm.covariances_, gmm.weights_))
        assert gmm_init_mode in ['rd', 'fp', 'km']
        assert gmm_type in ['soft', 'hard']
        nc = pickle.load(open(os.path.join(self.save_dir, 'nc.dat'), 'rb'))

        # we use fix random seed
        # random.seed(100)
        # unfold and shuffle nc data
        data = []
        for k, v in nc.items():
            if to_numeral(k) == None:
                print('invalid numeral {}'.format(k))
            else:
                data += [[to_numeral(k)]]*v

        print('total number of different numerals: ', len(nc))
        print('total number of numeral samples: ', len(data))

        # shuffle and subsample for MEM problem
        random.shuffle(data)

        if len(data) > 2000000:
            data = data[:2000000]

        if log_space:
            data = [weighted_log(x[0]) for x in data]

        print('subsampled to {}'.format(len(data)))

        data = np.array(data).reshape(-1,1)
        # getting initialization parameters
        if gmm_init_mode == 'km':

            if gmm_type == 'soft':
                gmm = GaussianMixture(components, max_iter=iters, n_init=1, verbose=10, init_params='kmeans')
            else:
                gmm = HardEMGaussianMixture(components, max_iter=iters, n_init=1, verbose=10, init_params='kmeans')


        else:
            # random select means
            if gmm_init_mode == 'rd' :
                prototypes = np.random.choice(data.reshape(-1), components)
            else:
                assert prototype_path is not None
                if log_space:
                    path = os.path.join(self.save_dir, 'som_log')
                else:
                    path = os.path.join(self.save_dir, 'som')

                path = os.path.join(path, prototype_path)
                prototypes = pickle.load(open(path, 'rb'))

                assert len(prototypes) == components

            mus = prototypes
            min_sigma = 1e-6

            diff = np.abs(data.reshape(len(data)) - mus[:, np.newaxis])

            amin = np.argmin(diff, axis=0)

            K = len(prototypes)
            clusters = [[0] for i in range(K)]
            for i in range(len(data)):
                clusters[amin[i]].append(data[i])

            means = np.array([np.mean(i) for i in clusters]).reshape(-1, 1)

            covs = np.array([np.std(i) if len(i) > 1 else min_sigma for i in clusters]).reshape(-1, 1, 1)
            precision = np.linalg.inv(covs)

            weights = np.array([len(c) for c in clusters])
            weights = weights / np.sum(weights)

            if gmm_type == 'soft':
                gmm = GaussianMixture(components, max_iter=iters, n_init=1, verbose=10, means_init=means,
                                  precisions_init=precision, weights_init=weights)
            else:
                gmm = HardEMGaussianMixture(components, max_iter=iters, n_init=1, verbose=10, means_init=means,
                                  precisions_init=precision, weights_init=weights)

        gmm.fit(data)
        if log_space:
            gmm_save_dir = os.path.join(self.save_dir, 'gmm_log')
        else:
            gmm_save_dir = os.path.join(self.save_dir, 'gmm')

        if not os.path.exists(gmm_save_dir):
            os.makedirs(gmm_save_dir)

        def single_variable_gaussian(x, mu, sigma):
            return 1. / (np.sqrt(2. * np.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.) / 2)

        def draw(gmm, X):
            x_min, x_max = min(X), max(X)
            # x = np.linspace(x_min, x_max, 10000)
            # x = np.array([])
            # for i in range(len(gmm.means_)):
            #     range_min, range_max = gmm.means_[i][0]-2 * gmm.covariances_[i][0], gmm.means_[i][0] + 2 * gmm.covariances_[i][0]
            #     x = np.append(x, np.linspace(range_min, range_max, 20))
            # x.sort()
            # print(x)
            print(gmm.means_)
            print(gmm.covariances_)
            print(gmm.weights_)

            X.sort()
            sum_y = np.zeros_like(X)
            plt.figure(0)
            plt.title('components')
            for i in range(len(gmm.means_)):
                y = single_variable_gaussian(X, gmm.means_[i][0], gmm.covariances_[i][0])
                y[y > 1] = 0 # set to 0 for better plot!
                sum_y += y * gmm.weights_[i]
                # yp = single_variable_gaussian(X, gmm.means_[i][0], gmm.covariances_[i][0])
                # yp[yp > 1] = 0
                # sum_yp += yp
                plt.plot(X, y)
            plt.savefig(os.path.join(gmm_save_dir, 'components-{}.png'.format(components)))

            plt.figure(1)
            plt.title('mixtures')

            plt.plot(X, sum_y, 'g-')
            plt.savefig(os.path.join(gmm_save_dir, 'mixture-{}.png'.format(components)))


        # 'rd' indicates for random initialization, 'fp' for 'from prototypes'

        pickle.dump(gmm, open(os.path.join(gmm_save_dir, 'gmm-{}-{}-{}.dat'.format(components, gmm_init_mode, gmm_type)), 'wb'))
        print('means: {} \n sigma: {}\n, weights: {}'.format(gmm.means_, gmm.covariances_, gmm.weights_))

        if log_space:
            data_points = np.array([weighted_log(x) for x in np.array(list(nc.keys()), dtype=np.float32)]).reshape(-1,1)
        else:
            data_points = np.array(list(nc.keys()), dtype=np.float32).reshape(-1,1)

        posterior = gmm.predict_proba(data_points)
        path = os.path.join(gmm_save_dir, 'gmm_posterior-{}-{}-{}.dat'.format(components, gmm_init_mode, gmm_type))
        pickle.dump(posterior, open(path, 'wb'))
        print('...Saving trained GMMs objects to {}'.format(path))


    def get_sent(self, line):
        sent = []
        for word in line.split():
            if word in self.vocab or is_numeral(word): # keep common words and all numerals
                sent.append(word)
            else:
                sent.append(self.unk)
        return sent

    def get_item(self, iword, owords):
        """
        form a proper data structure
        :param iword:
        :param owords:
        :return:
        """
        item = [None, [], 0, None, [0] * 2 * self.window, []]
        # [
        #   iword,
        #   [list of owords],
        #   0 or 1, indicator of iwords,
        #   None if iword is a token, numeral float if iword is a numeral,
        #   [one-hot indicator of owords],
        #   [list of numerals]
        # ]
        #
        # For example: if She is the center word and the window size is 2
        # oh , (She) is 1.67 m
        # [12, [99, 4, 5, 0], 0, None, [0,0,0,1], [1.67]]

        if is_numeral(iword):
            item[0] = self.word2idx[self.unk]
            item[2] = 1
            item[3] = to_numeral(iword)

        else:
            item[0] = self.word2idx[iword]

        for j in range(len(owords)):
            flag, oword = to_numeral_if_possible(owords[j])

            if flag:
                item[1].append(self.word2idx[self.unk])
                item[4][j] = 1
                item[5].append(oword)
            else:
                item[1].append(self.word2idx[oword])

        return item


    def convert(self, filepath, MAXDATA):
        print("loading built information")
        self.wc = pickle.load(open(os.path.join(self.save_dir, 'wc.dat'), 'rb'))
        self.nc = pickle.load(open(os.path.join(self.save_dir, 'nc.dat'), 'rb'))
        self.vocab = pickle.load(open(os.path.join(self.save_dir, 'vocab.dat'), 'rb'))
        self.idx2word = pickle.load(open(os.path.join(self.save_dir, 'idx2word.dat'), 'rb'))
        self.word2idx = pickle.load(open(os.path.join(self.save_dir, 'word2idx.dat'), 'rb'))

        print("converting corpus...")
        step = 0
        data = []
        batches = 0
        # Very Important Arguments
        MAXDATA = MAXDATA
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000))
                line = line.strip()
                if not line:
                    continue

                sent = self.get_sent(line)

                for i in range(self.window, len(sent) - self.window):
                    iword, owords = self.skipgram(sent, i)
                    item = self.get_item(iword, owords)
                    data.append(item)

                if len(data) > MAXDATA:
                    batches += 1

                    # change to joblib saving strategy
                    # joblib.dump(data, os.path.join(self.save_dir, 'train-batch-{}.jldat'.format(batches)))
                    pickle.dump(data, open(os.path.join(self.save_dir, 'train-batch-{}.dat'.format(batches)), 'wb'))
                    print('Saving train-batch-{}.dat'.format(batches))
                    data = [] # reset the data

            batches += 1
            # joblib.dump(data, os.path.join(self.save_dir, 'train-batch-{}.jldat'.format(batches)))
            pickle.dump(data, open(os.path.join(self.save_dir, 'train-batch-{}.dat'.format(batches)), 'wb'))

        print("")
        print("conversion done")

class PreprocessNumeralAsNumeralLSTM(Preprocess):
    def __init__(self, window=5, unk='<UNK_W>', save_dir='./preprocess/'):
        Preprocess.__init__(self, window, unk, save_dir)


    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import re
        import glob

        step = 0

        # the re for all possible token number
        RE_NUM = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))",
                            re.UNICODE)

        # support for directory mode, saving into 1 single file dump
        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:

                for line in file:
                    step += 1
                    if not step % 1000:
                        print("\n working on {}kth line in file {}".format(step // 1000, fpath))

                    line = line.strip()
                    if not line:
                        continue
                    sent = line.split()
                    sent_filtered = []
                    for token in sent:
                        # we treat word and numerals differently
                        # match numerals
                        res = re.findall(RE_NUM, token)
                        if res != []:
                            target = number_handler(token)
                            # we do not want nc to record ''
                            if target == '':
                                continue

                            if type(target) is list:
                                # ['u-32'] to ['u','-'.'32']
                                # [1997/07] to ['1997','/','7']

                                for i in target:
                                    if is_numeral(i):
                                        number = str(to_numeral(i))
                                        self.nc[number] = self.nc.get(number, 0) + 1
                                        # not change number to float in LSTM baseline
                                        sent_filtered.append(i)
                                    else:
                                        self.wc[i] = self.wc.get(i, 0) + 1
                                        sent_filtered.append(i)

                            elif is_numeral(target):
                                # ['-32.000'] to ['-32']
                                # prevent '-haha' like token, double check
                                number = str(to_numeral(target))
                                self.nc[number] = self.nc.get(number, 0) + 1
                                # not change number to float in LSTM baseline
                                sent_filtered.append(target)

                        else:
                            self.wc[token] = self.wc.get(token, 0) + 1
                            sent_filtered.append(token)

                    output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')

        output.close()
        print("filtering corpus done")

    def get_item(self, iword, owords):
        """
        form a proper data structure
        :param iword:
        :param owords:
        :return:
        """
        item = [None, [], 0, None, [0] * 2 * self.window, []]
        # [
        #   iword,
        #   [list of owords],
        #   0 or 1, indicator of iwords,
        #   None if iword is a token, numeral float if iword is a numeral,
        #   [one-hot indicator of owords],
        #   [list of numerals]
        # ]
        #
        # For example: if She is the center word and the window size is 2
        # oh , (She) is 1.67 m
        # [12, [99, 4, 5, 0], 0, None, [0,0,0,1], [1.67]]

        if is_numeral(iword):
            item[0] = self.word2idx[self.unk]
            item[2] = 1
            item[3] = iword # not convert to number

        else:
            item[0] = self.word2idx[iword]

        for j in range(len(owords)):
            flag, oword = to_numeral_if_possible(owords[j])

            if flag:
                item[1].append(self.word2idx[self.unk])
                item[4][j] = 1
                item[5].append(owords[j]) # not convert to number
            else:
                item[1].append(self.word2idx[oword])

        return item

# Unk_w: idx: 0, Unk_n, idx: 1
class PreprocessNumeralAsUnkNumeral(Preprocess):

    def __init__(self, window=5, unk='<UNK_W>', unk_n='<UNK_N>', save_dir='./preprocess/'):
        Preprocess.__init__(self, window, unk, save_dir)
        self.unk_n = unk_n
        self.wc = {self.unk: 1, self.unk_n: 1}

    def build_vocab(self, max_vocab=20000):
        """
        Overloading the vocabulary building
        :return:
        """
        print("rebuilding vocab")
        wc_nounk = copy(self.wc)
        wc_nounk.__delitem__(self.unk)
        wc_nounk.__delitem__(self.unk_n)
        self.idx2word = [self.unk, self.unk_n] + sorted(wc_nounk, key=wc_nounk.get, reverse=True)[:max_vocab - 2]
        self.word2idx = {self.idx2word[idx]: idx for idx, _ in enumerate(self.idx2word)}
        self.vocab = set([word for word in self.word2idx])
        print("build done")

    def get_sent(self, line):
        """
        Overload the get_sent method, we need to append unk_k when we get a numeral
        :param line:
        :return:
        """
        sent = []
        for word in line.split():
            if word in self.vocab:
                sent.append(word)
            elif is_numeral(word):
                sent.append(self.unk_n)
            else:
                sent.append(self.unk)
        return sent

    def get_item(self, iword, owords):
        item = [None, [], 0, None, [0] * 2 * self.window, []]

        # [
        #   iword,
        #   [list of owords],
        #   0 or 1, indicator of iwords numerals,
        #   None if iword is a token, numeral float if iword is a numeral,
        #   [one-hot indicator of owords],
        #   [list of numerals]
        # ]
        #
        # For example: if She is the center word and the window size is 2
        # oh , (She) is UNK_N m
        # [12, [99, 4, 5, 0], 0, None, [0,0,0,0], []]

        item[0] = self.word2idx[iword]
        for j in range(len(owords)):
            item[1].append(self.word2idx[owords[j]])

        return item


# Unk_w: idx: 0
class PreprocessNumeralAsToken(Preprocess):

    def __init__(self, window=5, unk='<UNK_W>', save_dir='./preprocess/'):
        Preprocess.__init__(self, window, unk, save_dir)

    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import re
        import glob

        step = 0

        # the re for all possible token number
        RE_NUM = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))",
                            re.UNICODE)

        # support for directory mode, saving into 1 single file dump
        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:

                for line in file:
                    step += 1
                    if not step % 1000:
                        print("\n working on {}kth line in file {}".format(step // 1000, fpath))

                    line = line.strip()
                    if not line:
                        continue
                    sent = line.split()
                    sent_filtered = []
                    for token in sent:
                        # we treat word and numerals differently
                        # match numerals
                        res = re.findall(RE_NUM, token)
                        if res != []:
                            target = number_handler(token)
                            # we do not want nc to record ''
                            if target == '':
                                continue

                            if type(target) is list:
                                # ['u-32'] to ['u','-'.'32']
                                # [1997/07] to ['1997','/','7']
                                for i in target:
                                    ww = str(to_numeral(i)) if is_numeral(i) else i
                                    self.wc[ww] = self.wc.get(ww, 0) + 1 # change to wc
                                    sent_filtered.append(ww)


                            elif is_numeral(target):
                                # ['-32.000'] to ['-32']
                                # prevent '-haha' like token, double check
                                number = str(to_numeral(target))
                                self.wc[number] = self.wc.get(number, 0) + 1 # change to wc
                                sent_filtered.append(number)

                        else:
                            self.wc[token] = self.wc.get(token, 0) + 1
                            sent_filtered.append(token)

                    output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')
        output.close()
        print("filtering and counting done")

    def get_sent(self, line):
        """
        Overload the get_sent method, we need to append unk_k when we get a numeral
        :param line:
        :return:
        """
        sent = []
        for word in line.split():
            if word in self.vocab:
                sent.append(word)
            else:
                sent.append(self.unk)
        return sent

    def get_item(self, iword, owords):
        item = [None, [], 0, None, [0] * 2 * self.window, []]

        item[0] = self.word2idx[iword]
        for j in range(len(owords)):
            item[1].append(self.word2idx[owords[j]])

        return item

class PreprocessNumeralAsTokenUnkNumeral(PreprocessNumeralAsUnkNumeral):

    def __init__(self, window=5, unk='<UNK_W>', unk_n='<UNK_N>', save_dir='./preprocess/'):
        PreprocessNumeralAsUnkNumeral.__init__(self, window, unk, unk_n, save_dir)

    def filter_and_count(self, filepath_in, filepath_out):
        print("Filtering numbers ...")
        import re
        import glob

        step = 0

        # the re for all possible token number
        RE_NUM = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))",
                            re.UNICODE)

        # support for directory mode, saving into 1 single file dump
        if os.path.isdir(filepath_in):
            # directory mode
            files = glob.glob(filepath_in + '/*.txt')
        else:
            files = [filepath_in]

        output = open(filepath_out, 'w', encoding='utf-8')

        for fpath in files:

            with codecs.open(fpath, 'r', encoding='utf-8') as file:

                for line in file:
                    step += 1
                    if not step % 1000:
                        print("\n working on {}kth line in file {}".format(step // 1000, fpath))

                    line = line.strip()
                    if not line:
                        continue
                    sent = line.split()
                    sent_filtered = []
                    for token in sent:
                        # we treat word and numerals differently
                        # match numerals
                        res = re.findall(RE_NUM, token)
                        if res != []:
                            target = number_handler(token)
                            # we do not want nc to record ''
                            if target == '':
                                continue

                            if type(target) is list:
                                # ['u-32'] to ['u','-'.'32']
                                # [1997/07] to ['1997','/','7']
                                for i in target:
                                    ww = str(to_numeral(i)) if is_numeral(i) else i
                                    self.wc[ww] = self.wc.get(ww, 0) + 1 # change to wc
                                    sent_filtered.append(ww)


                            elif is_numeral(target):
                                # ['-32.000'] to ['-32']
                                # prevent '-haha' like token, double check
                                number = str(to_numeral(target))
                                self.wc[number] = self.wc.get(number, 0) + 1 # change to wc
                                sent_filtered.append(number)

                        else:
                            self.wc[token] = self.wc.get(token, 0) + 1
                            sent_filtered.append(token)

                    output.write(bytes(' '.join(sent_filtered), 'utf-8').decode('utf-8') + '\n')
        output.close()
        print("filtering and counting done")


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    assert args.scheme in ['numeral_as_numeral', 'numeral_as_numeral_lstm',  'numeral_as_numeral', 'numeral_as_token', 'numeral_as_unk_numeral', 'numeral_as_token_unk_numeral']
    assert args.mode in ['train_som', 'train_gmm', 'build', 'convert', 'all']
    assert args.gmm_init_mode in ['rd', 'fp', 'km']

    if args.scheme == 'numeral_as_numeral':
        dir_name = args.saved_dir_name if args.saved_dir_name is not None else 'NumeralAsNumeral'
        preprocess = Preprocess(
            window=args.window,
            unk=args.unk_w,
            save_dir=os.path.join(args.save_dir, dir_name)
        )

    if args.scheme == 'numeral_as_numeral_lstm':
        dir_name = args.saved_dir_name if args.saved_dir_name is not None else 'NumeralAsNumeralLSTM'
        preprocess = PreprocessNumeralAsNumeralLSTM(
            window=args.window,
            unk=args.unk_w,
            save_dir=os.path.join(args.save_dir, dir_name)
        )


    elif args.scheme == 'numeral_as_token':
        dir_name = args.saved_dir_name if args.saved_dir_name is not None else 'NumeralAsToken'

        preprocess = PreprocessNumeralAsToken(
            window=args.window,
            unk=args.unk_w,
            save_dir=os.path.join(args.save_dir, dir_name)
        )

    elif args.scheme == 'numeral_as_unk_numeral':
        dir_name = args.saved_dir_name if args.saved_dir_name is not None else 'NumeralAsUnkNumeral'

        preprocess = PreprocessNumeralAsUnkNumeral(
            window=args.window,
            unk=args.unk_w,
            unk_n=args.unk_n,
            save_dir=os.path.join(args.save_dir, dir_name)
        )

    else:

        dir_name = args.saved_dir_name if args.saved_dir_name is not None else 'NumeralAsTokenUnkNumeral'

        preprocess = PreprocessNumeralAsTokenUnkNumeral(
            window=args.window,
            unk=args.unk_w,
            unk_n=args.unk_n,
            save_dir=os.path.join(args.save_dir, dir_name)
        )



    # TODO: add multiprocessing, only use 1 cpu core, slow
    # assert args.scheme in ['numeral_as_numeral','numeral_as_unk_numeral', 'numeral_as_token', 'all']

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
        preprocess.convert(os.path.join(preprocess.save_dir, args.filtered), MAXDATA=args.MAXDATA)

    else:
        print("Invalid preprocess mode")
