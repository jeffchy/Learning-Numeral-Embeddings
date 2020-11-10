# -*- coding: utf-8 -*-
import pickle
import random
import argparse
import torch as t
import numpy as np

from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS, Word2VecRNN, Word2VecGMM, Word2VecFixed
import os.path
from glob import glob
from tensorboardX import SummaryWriter
from sklearn.externals import joblib
from utils.number_handler import number_handler, is_numeral, to_numeral, to_numeral_if_possible


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='sgns', help="model name")
    parser.add_argument('--preprocess_dir', type=str, default='./preprocess/', help="data directory path")
    parser.add_argument('--save_dir', type=str, default='./pts/', help="model directory path")
    parser.add_argument('--e_dim', type=int, default=100, help="embedding dimension")
    parser.add_argument('--n_negs', type=int, default=10, help="number of negative samples")
    parser.add_argument('--numeral_pow', type=float, default=0.75, help="power of numerals")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=2048, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--conti', action='store_true', help="continue learning")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--no_subsample', action='store_true', help="do not use subsample for tokens")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    parser.add_argument('--log_dir', type=str, default='./logs/', help="log dir")
    parser.add_argument('--scheme', type=str, default='prototype', help="scheme for handling numbers, options [prototype, none, RNN, LSTM, fixed]")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument('--prototypes_path', type=str, default='prototypes.dat', help="filename of prototypes in the preprocessed path")
    parser.add_argument('--gmms_path', type=str, default='gmm.dat', help="filename of gmm instance in the preprocessed path")
    parser.add_argument('--alpha', type=float, default=1.0, help='exponential power factor of the prototype interpolation')
    parser.add_argument('--clip', type=float, default=0.02, help="clipping value for gradient")
    parser.add_argument('--log_space', action='store_true', help='log space mode for GMM and SOM')

    return parser.parse_args()

def custom_collate_prototype(batch):

    iword = []
    owords = []
    iword_id = []
    iword_numerals = []
    owords_id = []
    owords_numerals = []
    for i in batch:
        iword.append(i[0])
        owords.append(i[1])
        iword_id.append(i[2])
        if i[3] != None:
            iword_numerals.append(i[3])

        owords_id.append(i[4])

        if i[5] != []:
            owords_numerals += i[5]

    return ( t.tensor(iword),
             t.tensor(owords),
             t.tensor(iword_id, dtype=t.uint8),
             t.tensor(iword_numerals, dtype=t.float),
             t.tensor(owords_id, dtype=t.uint8),
             t.tensor(owords_numerals, dtype=t.float) )

numeral2idx = None

def custom_collate_gmm(batch):

    iword = []
    owords = []
    iword_id = []
    iword_numerals = []
    owords_id = []
    owords_numerals = []
    for i in batch:
        iword.append(i[0])
        owords.append(i[1])
        iword_id.append(i[2])
        if i[3] != None:
            iword_numerals.append(numeral2idx[i[3]])

        owords_id.append(i[4])

        if i[5] != []:
            owords_numerals += [numeral2idx[j] for j in i[5]]

    return ( t.tensor(iword),
             t.tensor(owords),
             t.tensor(iword_id, dtype=t.uint8),
             t.tensor(iword_numerals, dtype=t.int64),
             t.tensor(owords_id, dtype=t.uint8),
             t.tensor(owords_numerals, dtype=t.int64) )


idx2digit = ['0','1','2','3','4','5','6','7','8','9','.','-','+','e']
digit2idx = {idx2digit[i]:i for i in range(len(idx2digit))}
max_token_len = 20 # should be equal to

def convert_digit_to_tensor(numeral_str):
    represent = t.zeros(max_token_len, len(idx2digit))
    assert len(numeral_str) <= max_token_len
    for i in range(len(numeral_str)):
        digit = numeral_str[i]
        try:
            idx = digit2idx[digit]
        except KeyError:
            idx = 0 # very small number of corrupt cases, simple fix

        represent[i][idx] = 1

    return represent


def custom_collate_RNN(batch):
    # TODO change to the RNN supported format
    iword = []
    owords = []
    iword_id = []
    iword_numerals = []
    owords_id = []
    owords_numerals = []
    for i in batch:
        iword.append(i[0])
        owords.append(i[1])
        iword_id.append(i[2])
        if i[3] != None:
            iword_numerals.append(i[3])

        owords_id.append(i[4])

        if i[5] != []:
            owords_numerals += i[5]
    # c = iword_numerals
    iword_numerals_length = [len(str(i)) for i in iword_numerals]
    owords_numerals_length = [len(str(i)) for i in owords_numerals]
    iword_numerals = [t.unsqueeze(convert_digit_to_tensor(str(i)), 0) for i in iword_numerals]
    owords_numerals = [t.unsqueeze(convert_digit_to_tensor(str(i)), 0) for i in owords_numerals]

    if iword_numerals:
        iword_numerals = t.cat(iword_numerals)
        iword_numerals_length = t.tensor(iword_numerals_length, dtype=t.uint8)
    else:
        iword_numerals = t.tensor([])
        iword_numerals_length = t.tensor([])

    if owords_numerals:
        owords_numerals = t.cat(owords_numerals)
        owords_numerals_length = t.tensor(owords_numerals_length, dtype=t.uint8)
    else:
        owords_numerals = t.tensor([])
        owords_numerals_length = t.tensor([])

    return ( t.tensor(iword),
             t.tensor(owords),
             t.tensor(iword_id, dtype=t.uint8),
             iword_numerals, # B x seq_max_len x 14 (digit vocab)
             t.tensor(owords_id, dtype=t.uint8),
             owords_numerals,
             iword_numerals_length,
             owords_numerals_length)


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        # https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
        # bytes_in = bytearray(0)
        # max_bytes = 2 ** 31 - 1
        # input_size = os.path.getsize(datapath)
        # with open(datapath, 'rb') as f_in:
        #     for _ in range(0, input_size, max_bytes):
        #         bytes_in += f_in.read(max_bytes)
        # data = pickle.loads(bytes_in)
        # data = joblib.load(datapath)
        # joblib is insanely slow for py object

        # on the cluster we can directly load the file! about 2xfaster than loading bytes
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None: # sub-sampling
            self.data = []
            for i in data:
                iword, owords, iword_indicator, iword_numerals, owords_indicator, owords_numerals = i
                if iword_indicator != 0: # iword is a numeral, we keep all numerals
                    self.data.append(i)
                elif random.random() > ws[iword]: # iword is a token, and satisfy subsample rate
                    self.data.append(i)

        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NanLossError(Exception):
    def __init__(self):
        Exception.__init__(self)  # 重写父类方法

    def __str__(self):
        return "NanLoss Appear During Training"

if __name__ == '__main__':

    args = parse_args()
    # initialize

    writer = SummaryWriter(args.log_dir)

    idx2word = pickle.load(open(os.path.join(args.preprocess_dir, 'idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(args.preprocess_dir, 'wc.dat'), 'rb'))
    nc = pickle.load(open(os.path.join(args.preprocess_dir, 'nc.dat'), 'rb'))

    # filter nc
    for k, v in nc.copy().items():
        f = np.float32(k) # caution need to be float32 cause we use float32 in further caculation
        if np.isnan(f) or np.isinf(f):
            nc.pop(k)
            print(f)

    numeral2idx = {to_numeral(numeral):idx for idx, numeral in enumerate(list(nc.keys()))}

    wf = np.array([wc[word] for word in idx2word])
    w_sum = wf.sum()
    wf = wf / w_sum
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    token_weights = wf if args.weights else None

    nf = np.array(list(nc.values()))
    n_sum = nf.sum()
    nf = nf / n_sum
    numerals = np.array(list(nc.keys()))

    numeral_weights = nf

    n_rate = n_sum / (n_sum + w_sum)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # check for ags.scheme, config the custom_collate_fn
    assert args.scheme in ['prototype', 'none', 'RNN', 'LSTM','GMM', 'fixed']
    if args.scheme in ['prototype']:
        custom_collate_fn = custom_collate_prototype
        if args.log_space:
            prototypes = pickle.load(open(os.path.join(args.preprocess_dir, os.path.join('som_log', args.prototypes_path)), 'rb'))

        else:
            prototypes = pickle.load(open(os.path.join(args.preprocess_dir, os.path.join('som', args.prototypes_path)), 'rb'))

        model = Word2Vec(prototypes=prototypes, alpha=args.alpha, vocab_size=vocab_size, embedding_size=args.e_dim,
                         is_cuda=args.cuda, log_space=args.log_space)

    elif args.scheme in ['GMM']:
        # we can still use custom collate prototype for dataloader
        custom_collate_fn = custom_collate_gmm
        # gmm instance path!
        if args.log_space:
            gmm_posterior = pickle.load(open(os.path.join(args.preprocess_dir, os.path.join('gmm_log', args.gmms_path)), 'rb'))
        else:
            gmm_posterior = pickle.load(open(os.path.join(args.preprocess_dir, os.path.join('gmm', args.gmms_path)), 'rb'))

        # data_points = np.array(list(nc.keys()), dtype=np.float).reshape(-1,1)
        # posterior = gmm.predict_proba(data_points)

        model = Word2VecGMM(gmm_posterior=gmm_posterior, vocab_size=vocab_size, embedding_size=args.e_dim,
                         is_cuda=args.cuda, log_space=args.log_space)

    elif args.scheme in ['none']:
        custom_collate_fn = custom_collate_prototype
        model = Word2Vec(prototypes=None, vocab_size=vocab_size, embedding_size=args.e_dim,
                         is_cuda=args.cuda)

    elif args.scheme in ['fixed']:
        custom_collate_fn = custom_collate_prototype
        model = Word2VecFixed(vocab_size=vocab_size, embedding_size=args.e_dim,
                         is_cuda=args.cuda)


    else:
        custom_collate_fn = custom_collate_RNN

        numeral_to_onehot = t.zeros(len(numerals), max_token_len, len(digit2idx))
        numeral_to_length = t.tensor(t.zeros(len(numerals)), dtype=t.uint8)

        print('start building numeral to onehot')
        for i in range(len(numerals)):
            numeral_to_onehot[i] = convert_digit_to_tensor(numerals[i])
            numeral_to_length[i] = len(numerals[i])
        print('building onehot done')

        model = Word2VecRNN(numeral_to_onehot=numeral_to_onehot, numeral_to_length=numeral_to_length, vocab_size=vocab_size, embedding_size=args.e_dim,
                         is_cuda=args.cuda, scheme=args.scheme)


    if args.cuda:
        model = model.cuda()

    modelpath = os.path.join(args.save_dir, '{}.pt'.format(args.name))

    sgns = SGNS(token_weights=token_weights, numeral_weights=numeral_weights, embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, n_rate=n_rate, numerals=numerals, scheme=args.scheme, numeral_pow=args.numeral_pow)

    if os.path.isfile(modelpath) and args.conti:
        sgns.load_state_dict(t.load(modelpath))

    optim = Adam(sgns.parameters(), lr=args.lr)
    optimpath = os.path.join(args.save_dir, '{}.optim.pt'.format(args.name))
    if os.path.isfile(optimpath) and args.conti:
        optim.load_state_dict(t.load(optimpath))

    # Serialized Training
    for epoch in range(1, args.epoch + 1):
        total_loss = 0

        path_list = glob(args.preprocess_dir+'/train-batch-*.dat')
        for train_file in path_list:

            dataset = PermutedSubsampledCorpus(train_file, None) if args.no_subsample else PermutedSubsampledCorpus(train_file, ws)

            dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True, collate_fn=custom_collate_fn)
            total_batches = int(np.ceil(len(dataset) / args.mb))

            pbar = tqdm(dataloader)
            pbar.set_description("[Epoch {}, File {}]".format(epoch, train_file))

            if args.scheme in ['none', 'prototype', 'GMM', 'fixed']:

                for iword, owords, iword_indicator, iword_numerals, owords_indicator, owords_numerals in pbar:

                    loss = sgns(iword,
                                owords,
                                iword_indicator,
                                iword_numerals,
                                owords_indicator,
                                owords_numerals)




                    if t.isnan(loss):
                        pickle.dump([iword, owords, iword_numerals, owords_numerals],
                                    open(os.path.join(args.save_dir, 'nan_info.dat'), 'wb'))

                        raise NanLossError()

                    optim.zero_grad()
                    loss.backward()
                    
                    norms = []
                    for p in sgns.parameters():
                        if p.grad is not None:
                            norm = p.grad.data.norm(2)
                            norms.append(norm)
                    max_norm = np.max(norms)

                    if np.isnan(max_norm):
                        continue
 
                    t.nn.utils.clip_grad_norm_(sgns.parameters(), max_norm=args.clip, norm_type=2)

                    optim.step()
                    pbar.set_postfix(loss=loss.item(), total_loss=total_loss, max_norm=max_norm)
                    total_loss += loss.item()


            else:

                for iword, owords, iword_indicator, iword_numerals, owords_indicator, owords_numerals, iword_numerals_length, owords_numerals_length in pbar:

                    loss = sgns.forward_RNN(iword,
                                            owords,
                                            iword_indicator,
                                            iword_numerals,
                                            owords_indicator,
                                            owords_numerals,
                                            iword_numerals_length,
                                            owords_numerals_length)
                    if t.isnan(loss):
                        pickle.dump([iword, owords, iword_numerals, owords_numerals],
                                    open(os.path.join(args.save_dir, 'nan_info.dat'), 'wb'))

                        raise NanLossError()

                    optim.zero_grad()
                    loss.backward()

                    norms = []
                    for p in sgns.parameters():
                        if p.grad is not None:
                            norm = p.grad.data.norm(2)
                            norms.append(norm)
                    max_norm = np.max(norms)

                    if np.isnan(max_norm):
                        continue

                    t.nn.utils.clip_grad_norm_(sgns.parameters(), max_norm=args.clip, norm_type=2)

                    optim.step()
                    pbar.set_postfix(loss=loss.item(), total_loss=total_loss, max_norm=max_norm)
                    total_loss += loss.item()

        writer.add_scalar('data/loss', total_loss, epoch)
        for name, param in model.named_parameters():
            try:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            except:
                print("Error when adding histogram for param {}".format(name))

        idx2vec_i = model.ivectors.weight.data.clone().cpu().numpy()
        idx2vec_o = model.ovectors.weight.data.clone().cpu().numpy()

        # save the trained embeddings
        if args.scheme == 'prototype':
            iprototype_embeddings = model.iprototypes_embeddings.clone().cpu().detach().numpy()
            oprototype_embeddings = model.oprototypes_embeddings.clone().cpu().detach().numpy()
            pickle.dump({
                'i_embedding': iprototype_embeddings,
                'o_embedding': oprototype_embeddings,
                'prototypes': prototypes,
            }, open(os.path.join(args.save_dir, 'trained_prototypes_epoch{}_{}_{}.dat'.format(epoch, len(prototypes), args.alpha)), 'wb'))

        if args.scheme == 'GMM':
            iprototype_embeddings = model.iprototypes_embeddings.clone().cpu().detach().numpy()
            oprototype_embeddings = model.oprototypes_embeddings.clone().cpu().detach().numpy()
            pickle.dump({
                'i_embedding': iprototype_embeddings,
                'o_embedding': oprototype_embeddings,
                'gmm_posterior': gmm_posterior,
            }, open(os.path.join(args.save_dir, 'trained_gmms_epoch{}_{}.dat'.format(epoch, gmm_posterior.shape[1])), 'wb'))


        # save the word vectors and the model/optims
        pickle.dump(idx2vec_i, open(os.path.join(args.save_dir, 'idx2vec_i_epoch{}.dat'.format(epoch)), 'wb'))
        pickle.dump(idx2vec_o, open(os.path.join(args.save_dir, 'idx2vec_o_epoch{}.dat'.format(epoch)), 'wb'))
        t.save(sgns.state_dict(), os.path.join(args.save_dir, '{}_epoch{}.pt'.format(args.name, epoch)))
        t.save(optim.state_dict(), os.path.join(args.save_dir, '{}_epoch{}.optim.pt'.format(args.name, epoch)))

    # save the arguments
    with open(os.path.join(args.save_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

