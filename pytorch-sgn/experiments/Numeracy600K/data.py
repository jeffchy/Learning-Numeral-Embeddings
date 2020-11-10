import pickle
from torch.utils.data import Dataset, DataLoader
from data_utils import read_json
import os
import numpy as np
import torch as t
import glob
import math
from collections import Counter
from tqdm import tqdm

class Numeracy600kDataset(Dataset):

    def __init__(self, fname, exp_root):
        self.data = read_json(fname)
        self.exp_root = exp_root
        self.vocab = pickle.load(open(os.path.join(exp_root, 'vocab.pkl'), 'rb'))
        self.word2idx = pickle.load(open(os.path.join(exp_root, 'word2idx.pkl'), 'rb'))
        self.idx2word = pickle.load(open(os.path.join(exp_root, 'idx2word.pkl'), 'rb'))
        self.get_longest()


    def __len__(self):
        return len(self.data)

    def get_longest(self):
        longest = 0
        for instance in self.data:
            length = len(instance['title_new'].strip().split())
            if length > longest:
                longest = length
        self.longest = longest

    def __getitem__(self, idx):
        # create data sample
        instance = self.data[idx]
        sent = instance['title_new'].strip().split()

        input = []
        for token in sent:
            token = token.lower()
            if token in self.word2idx:
                input.append(self.word2idx[token])
            else:
                input.append(self.word2idx['<unk>'])

        # pad sequence
        length = len(input)
        input = input + [self.word2idx['<pad>']]*(self.longest - len(input))
        label = instance['magnitude']
        return {
            'input': np.array(input),
            'label': label,
            'length': length
        }

def to_numeral(token):
    try:
        num = np.float32(token)
        if num == np.float32('inf') or num == np.float32('-inf') or math.isnan(num):
            return False, None
        return True, num

    except ValueError:
        return False, None

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

def get_prototype_embedding(num, trained_prototypes, alpha, log_space=False, mode='i'):
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

    prototypes, prototypes_embeddings = trained_prototypes['prototypes'], trained_prototypes['{}_embedding'.format(mode)]
    prototypes2vec_i = trained_prototypes['{}_embedding'.format(mode)]
    if log_space:
        # if prototype already in log space, no need to transform
        transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).view(-1, 1)
    else:
        transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).apply_(weighted_log).view(-1, 1)

    prototype_weights = get_numeral_embed_weights_batch(t.tensor(t.from_numpy(np.array([num])),dtype=t.float),transformed_protp, alpha=alpha)
    vec = t.matmul(prototype_weights.transpose(0,1),t.from_numpy(prototypes2vec_i))

    return vec

def build_pretrain_embedding_prorotype(embedding_path, preprocess_dir, epoch, alpha, word_alphabet, log_space=False, mode='i'):

    assert embedding_path != None
    assert preprocess_dir != None

    file_paths = glob.glob(embedding_path +'/trained_prototypes_epoch{}_*.dat'.format(epoch))
    assert len(file_paths) == 1

    trained_prototypes = pickle.load(open(file_paths[0], 'rb'))
    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_{}_epoch{}.dat'.format(mode, epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]

    alphabet_size = len(word_alphabet)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_alphabet), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_represent = 0
    not_match = 0
    for word, index in tqdm(word_alphabet.items()):
        flag, num = to_numeral(word)
        if word in word2idx:
            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif flag:
            vec = get_prototype_embedding(num, trained_prototypes, alpha, log_space=log_space, mode=mode)
            numeral_represent += 1

        elif word.capitalize() in word2idx:

            vec = idx2vec[word2idx[word.capitalize()]]
            case_match += 1
        else:
            vec = idx2vec[word2idx['<UNK_W>']]
            not_match += 1

        pretrain_emb[index, :] = vec

    pretrained_size = len(word2idx)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_represent:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_represent ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_GMM(embedding_path, preprocess_dir, epoch, word_alphabet, log_space=False, mode='i'):

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


    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_{}_epoch{}.dat'.format(mode, epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))

    prototypes2vec_i = trained_prototypes['{}_embedding'.format(mode)]

    embedd_dim = idx2vec.shape[1]

    alphabet_size = len(word_alphabet)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_alphabet), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_represent = 0
    not_match = 0
    for word, index in tqdm(word_alphabet.items()):
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

        elif word.capitalize() in word2idx:

            vec = idx2vec[word2idx[word.capitalize()]]
            case_match += 1
        else:
            vec = idx2vec[word2idx['<UNK_W>']]
            not_match += 1

        pretrain_emb[index, :] = vec

    pretrained_size = len(word2idx)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_represent:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_represent ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_token(embedding_path, preprocess_dir, epoch, word_alphabet, mode='i'):

    assert embedding_path != None
    assert preprocess_dir != None

    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_{}_epoch{}.dat'.format(mode, epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]

    alphabet_size = len(word_alphabet)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_alphabet), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_perfect_match = 0
    numeral_not_match = 0

    not_match = 0
    for word, index in tqdm(word_alphabet.items()):
        flag, num = to_numeral(word)

        if word in word2idx:
            if flag:
                numeral_perfect_match += 1

            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif word.capitalize() in word2idx:

            vec = idx2vec[word2idx[word.capitalize()]]
            case_match += 1

        else:
            if flag:
                numeral_not_match += 1
                vec = idx2vec[word2idx['<UNK_N>']]
            else:
                vec = idx2vec[word2idx['<UNK_W>']]

            not_match += 1

        pretrain_emb[index, :] = vec

    pretrained_size = len(word2idx)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_perfect_match:%s, numeral_not_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_perfect_match, numeral_not_match ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_fixed(embedding_path, preprocess_dir, epoch, word_alphabet, mode='i'):

    assert embedding_path != None
    assert preprocess_dir != None

    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_{}_epoch{}.dat'.format(mode, epoch)), 'rb'))
    idx2word = pickle.load(open(os.path.join(preprocess_dir, 'idx2word.dat'), 'rb'))
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]

    alphabet_size = len(word_alphabet)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_alphabet), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_perfect_match = 0
    numeral_not_match = 0

    not_match = 0
    for word, index in tqdm(word_alphabet.items()):
        flag, num = to_numeral(word)

        if word in word2idx:
            if flag:
                numeral_perfect_match += 1

            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif word.capitalize() in word2idx:

            vec = idx2vec[word2idx[word.capitalize()]]
            case_match += 1

        else:
            if flag:
                numeral_not_match += 1
                num = weighted_log(num)
                vec = np.ones([1, embedd_dim])
                vec[0][0] = num
                vec /= (2 * embedd_dim)

            else:
                vec = idx2vec[word2idx['<UNK_W>']]

            not_match += 1

        pretrain_emb[index, :] = vec

    pretrained_size = len(word2idx)

    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_perfect_match:%s, numeral_not_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_perfect_match, numeral_not_match ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

def build_pretrain_embedding_LSTM(embedding_path, preprocess_dir, epoch, word_alphabet, mode='i'):

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

    idx2vec = pickle.load(open(os.path.join(embedding_path, 'idx2vec_{}_epoch{}.dat'.format(mode, epoch)), 'rb'))
    LSTM_model_path = os.path.join(embedding_path, 'sgns_epoch{}.pt'.format(epoch))
    LSTM_params = t.load(LSTM_model_path, map_location='cpu')
    word2idx = pickle.load(open(os.path.join(preprocess_dir, 'word2idx.dat'), 'rb'))
    embedd_dim = idx2vec.shape[1]


    digital_RNN_i = t.nn.LSTM(14, idx2vec.shape[1], 1, batch_first=True)

    digital_RNN_i.bias_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_{}.bias_hh_l0'.format(mode)])
    digital_RNN_i.bias_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_{}.bias_ih_l0'.format(mode)])
    digital_RNN_i.weight_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_{}.weight_hh_l0'.format(mode)])
    digital_RNN_i.weight_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_{}.weight_ih_l0'.format(mode)])

    idx2digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', '+', 'e']
    digit2idx = {idx2digit[i]: i for i in range(len(idx2digit))}
    max_token_len = 20  # should be equal to

    def get_LSTM_embedding(num, digital_RNN_i):
        temp = convert_digit_to_tensor(str(num)).view(1, 20, 14)
        _, (hn, cn) = digital_RNN_i(temp)
        e = hn.squeeze().detach().numpy()

        return e


    alphabet_size = len(word_alphabet)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_alphabet), embedd_dim])
    perfect_match = 0
    case_match = 0
    numeral_represent = 0

    not_match = 0
    for word, index in tqdm(word_alphabet.items()):
        flag, num = to_numeral(word)
        if word in word2idx:
            vec = idx2vec[word2idx[word]]
            perfect_match += 1

        elif flag:
            vec = get_LSTM_embedding(num, digital_RNN_i)
            numeral_represent += 1

        elif word.capitalize() in word2idx:

            vec = idx2vec[word2idx[word.capitalize()]]
            case_match += 1
        else:
            vec = idx2vec[word2idx['<UNK_W>']]
            not_match += 1

        pretrain_emb[index, :] = vec

    pretrained_size = len(word2idx)

    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, numeral_represent:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, numeral_represent ,not_match, (not_match+0.)/alphabet_size))
    return pretrain_emb, embedd_dim

class EmbedLoader(object):
    def __init__(self, config, word2idx):
        self.config = config
        self.word2idx = word2idx

    def build_pretrained_embed(self,
                               embed_name,
                               embedding_path,
                               preprocess_dir,
                               embedding_epoch = 1,
                               alpha = 1.0,
                               log_space = False,
                               mode='i'):

        if embed_name == 'Rand':
            pretrained_embed = None
            embed_dim = self.config.embedding_dim

        elif embed_name == 'Prototype':
            pretrained_embed, embed_dim = build_pretrain_embedding_prorotype(
                embedding_path = embedding_path,
                preprocess_dir = preprocess_dir,
                epoch = embedding_epoch,
                alpha = alpha,
                word_alphabet = self.word2idx,
                log_space = log_space,
                mode = mode
            )

        elif embed_name == 'GMM':
            pretrained_embed, embed_dim = build_pretrain_embedding_GMM(
                embedding_path = embedding_path,
                preprocess_dir = preprocess_dir,
                epoch = embedding_epoch,
                word_alphabet=self.word2idx,
                log_space = log_space,
                mode=mode
            )

        else:
            if embed_name == 'Token':
                fn = build_pretrain_embedding_token
            elif embed_name == 'LSTM':
                fn = build_pretrain_embedding_LSTM
            else:
                fn = build_pretrain_embedding_fixed

            pretrained_embed, embed_dim = fn(
                embedding_path = embedding_path,
                preprocess_dir = preprocess_dir,
                epoch = embedding_epoch,
                word_alphabet=self.word2idx,
                mode=mode
            )

        assert embed_dim == self.config.embedding_dim

        # some post processing to distinguish <trt> and <unk_w>
        np.random.seed(0)
        pretrained_embed[self.word2idx['<trt>'], :] = np.random.uniform(-0.15, 0.15, [1, embed_dim])
        print(pretrained_embed.shape)

        pickle.dump(pretrained_embed, open('exps/{}/{}/{}_{}.emb'.format(self.config.exp_dir, 'embed', self.config.pretrained_embed_name, mode), 'wb'))



