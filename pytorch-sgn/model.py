# -*- coding: utf-8 -*-
import numpy as np
import torch as t
import torch.nn as nn

from torch import LongTensor as LT
from torch import FloatTensor as FT
from som.intopolate import weighted_log, weighted_average
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Word2Vec(nn.Module):

    def __init__(self, prototypes, alpha=1.0, is_cuda=True, vocab_size=20000, embedding_size=300,  padding_idx=0, numeral_weighted_fn=weighted_log, log_space=False):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)

        # initialize weights
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))

        self.prototypes = prototypes # numpy array
        self.alpha = alpha # the power factor of prototype interpolation
        self.is_cuda = is_cuda
        self.log_space = log_space

        if prototypes is not None:
            self.iprototypes_embeddings = nn.Parameter(FT(len(self.prototypes), self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
            self.oprototypes_embeddings = nn.Parameter(FT(len(self.prototypes), self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
            self.iprototypes_embeddings.requires_grad = True
            self.oprototypes_embeddings.requires_grad = True

        self.numeral_weighted_fn = numeral_weighted_fn

        self.transformed_prototypes = None
        if prototypes is not None and len(prototypes.shape) == 1:
            if self.log_space:
                # if prototype already in log space, no need to transform
                self.transformed_prototypes = t.tensor(t.from_numpy(self.prototypes), dtype=t.float32).view(-1,1)
            else:
                self.transformed_prototypes = t.tensor(t.from_numpy(self.prototypes), dtype=t.float32).apply_(self.numeral_weighted_fn).view(-1,1)
            self.transformed_prototypes = self.transformed_prototypes.cuda() if self.is_cuda else self.transformed_prototypes

        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True


    def get_numeral_embed_weights_batch(self, numerals):
        """
        :param numerals: tensor of numerals
        :return: weights matrix
        """

        # TODO: can we get a function that can easily be gpu_accelerated?
        numerals.apply_(self.numeral_weighted_fn)
        numerals = numerals.cuda() if self.is_cuda else numerals
        l_numerals = numerals.size()[0]
        l_prototypes = self.transformed_prototypes.size()[0]

        min_margin =  t.tensor(0.0001, dtype=t.float32)
        min_margin = min_margin.cuda() if self.is_cuda else min_margin
        transformed_prototypes_batch = self.transformed_prototypes.expand(l_prototypes, l_numerals)

        prototype_weights = t.pow(1 / t.max(t.abs(transformed_prototypes_batch - numerals), min_margin), self.alpha)
        prototype_weights /= t.sum(prototype_weights, 0)
        return prototype_weights # [prototype_size x num_of_numerals]


    def forward_i(self, data, iword_indicator, iword_numerals):
        v = LT(data)
        v = v.cuda(self.ivectors.weight.device) if self.is_cuda else v
        embed = self.ivectors(v)

        if iword_numerals.size()[0] == 0 or self.prototypes is None:
            return embed

        prototype_weights = self.get_numeral_embed_weights_batch(iword_numerals) # [prototype_size x num_of_numerals]
        numeral_embed = t.matmul(prototype_weights.transpose(0,1), self.iprototypes_embeddings)
        # [num_of_numerals x prototype_size ]  x [prototype_size x embedding_size] => [num_of_numeral x embedding_size]
        embed[iword_indicator] = numeral_embed

        return embed

    def forward_o(self, data, owords_indicator, owords_numerals):
        v = LT(data)
        v = v.cuda(self.ivectors.weight.device) if self.is_cuda else v
        embed = self.ovectors(v)
        if owords_numerals.size()[0] == 0 or self.prototypes is None:
            return embed

        prototype_weights = self.get_numeral_embed_weights_batch(owords_numerals) # [prototype_size x num_of_numerals]
        numeral_embed = t.matmul(prototype_weights.transpose(0,1), self.oprototypes_embeddings)
        # [num_of_numerals x prototype_size ]  x [prototype_size x embedding_size] => [num_of_numeral x embedding_size]
        embed[owords_indicator] = numeral_embed

        return embed


class Word2VecRNN(nn.Module):

    def __init__(self, numeral_to_onehot, numeral_to_length, is_cuda=True, vocab_size=20000, embedding_size=300,  padding_idx=0, scheme='LSTM'):
        super(Word2VecRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)

        if scheme == 'LSTM':
            self.digital_RNN_i = nn.LSTM(14, self.embedding_size, 1, batch_first=True)
            self.digital_RNN_o = nn.LSTM(14, self.embedding_size, 1, batch_first=True)

        else:
            self.digital_RNN_i = nn.RNN(14, self.embedding_size, 1, batch_first=True)
            self.digital_RNN_o = nn.RNN(14, self.embedding_size, 1, batch_first=True)


        # initialize weights
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))

        self.is_cuda = is_cuda
        self.scheme = scheme
        self.numeral_to_onehot = numeral_to_onehot
        self.numeral_to_length = numeral_to_length

    def invert_permutation(self, permutation):
        if permutation is None:
            return None
        output = t.empty_like(permutation)
        s_ = t.arange(0, permutation.numel())
        s_ = s_.cuda() if self.is_cuda else s_
        output.scatter_(0, permutation, s_)
        return output

    def forward_i(self, data, iword_indicator, iword_numerals, iword_numeral_length):
        v = LT(data)
        v = v.cuda() if self.is_cuda else v
        embed = self.ivectors(v)

        # B x T x F
        if iword_numerals.size()[0] == 0:
            return embed

        iword_numerals = iword_numerals.cuda() if self.is_cuda else iword_numerals
        iword_numeral_length = iword_numeral_length.cuda() if self.is_cuda else iword_numeral_length
        iword_numeral_length_permuted, perm_idx = iword_numeral_length.sort(0, descending=True)
        iword_numerals_permuted = iword_numerals[perm_idx]
        packed_input = pack_padded_sequence(iword_numerals_permuted, iword_numeral_length_permuted, batch_first=True)
        invert_perm_idx = self.invert_permutation(perm_idx)

        # assert t.equal(iword_numerals_permuted[invert_perm_idx], iword_numerals)
        # assert iword_indicator.sum() == iword_numerals.size()[0]

        if self.scheme == 'LSTM':
            _, (hn, cn) = self.digital_RNN_i(packed_input)

        else:
            _, hn = self.digital_RNN_i(packed_input)

        # TODO: how to check?
        embed[iword_indicator] = hn.squeeze(0)[invert_perm_idx]

        return embed

    def forward_o(self, data, owords_indicator, owords_numerals, owords_numeral_length):
        v = LT(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        embed = self.ovectors(v)
        if owords_numerals.size()[0] == 0:
            return embed

        owords_numerals = owords_numerals.cuda() if self.is_cuda else owords_numerals
        owords_numeral_length = owords_numeral_length.cuda() if self.is_cuda else owords_numeral_length
        owords_numeral_length_permuted, perm_idx = owords_numeral_length.sort(0, descending=True)
        owords_numerals_permuted = owords_numerals[perm_idx]
        packed_input = pack_padded_sequence(owords_numerals_permuted, owords_numeral_length_permuted, batch_first=True)
        invert_perm_idx = self.invert_permutation(perm_idx)

        assert t.equal(owords_numerals_permuted[invert_perm_idx], owords_numerals)
        assert owords_indicator.sum() == owords_numerals.size()[0]

        if self.scheme == 'LSTM':
            _, (hn, cn) = self.digital_RNN_o(packed_input)
        else:
            _, hn = self.digital_RNN_o(packed_input)

        embed[owords_indicator] = hn.squeeze(0)[invert_perm_idx]

        return embed

class Word2VecGMM(nn.Module):

    def __init__(self, gmm_posterior, is_cuda=True, vocab_size=20000, embedding_size=300,  padding_idx=0, log_space=False, numeral_weighted_fn=weighted_log):
        super(Word2VecGMM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)

        # initialize weights
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))

        self.is_cuda = is_cuda

        self.gmm_posterior = t.tensor(gmm_posterior, dtype=t.float32) # gmm_posterior [numeral x prototype]
        # self.gmm_posterior = self.gmm_posterior.cuda() if self.is_cuda else self.gmm_posterior # to cuda

        self.iprototypes_embeddings = nn.Parameter(FT(self.gmm_posterior.size()[1], self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.oprototypes_embeddings = nn.Parameter(FT(self.gmm_posterior.size()[1], self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size))
        self.iprototypes_embeddings.requires_grad = True
        self.oprototypes_embeddings.requires_grad = True

        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

        self.log_space = log_space
        self.numeral_weighted_fn = numeral_weighted_fn


    def get_numeral_embed_weights_batch(self, numerals):
        """
        :param numerals: tensor of numerals idxs!
        :return: weights matrix
        """
        # if self.log_space:
        #     numerals.apply_(self.numeral_weighted_fn)

        posterior = self.gmm_posterior[numerals]
        posterior = posterior.cuda() if self.is_cuda else posterior

        return posterior # [num_of_numerals x prototype_size]


    def forward_i(self, data, iword_indicator, iword_numerals):
        v = LT(data)
        v = v.cuda(self.ivectors.weight.device) if self.is_cuda else v
        embed = self.ivectors(v)

        if iword_numerals.size()[0] == 0 or self.gmm_posterior is None:
            return embed

        prototype_weights = self.get_numeral_embed_weights_batch(iword_numerals) # [ num_of_numerals x prototype_size]
        numeral_embed = t.matmul(prototype_weights, self.iprototypes_embeddings)
        # [num_of_numerals x prototype_size ]  x [prototype_size x embedding_size] => [num_of_numeral x embedding_size]
        embed[iword_indicator] = numeral_embed

        return embed

    def forward_o(self, data, owords_indicator, owords_numerals):
        v = LT(data)
        v = v.cuda(self.ivectors.weight.device) if self.is_cuda else v
        embed = self.ovectors(v)
        if owords_numerals.size()[0] == 0 or self.gmm_posterior is None:
            return embed

        prototype_weights = self.get_numeral_embed_weights_batch(owords_numerals) # [prototype_size x num_of_numerals]
        numeral_embed = t.matmul(prototype_weights, self.oprototypes_embeddings)
        # [num_of_numerals x prototype_size ]  x [prototype_size x embedding_size] => [num_of_numeral x embedding_size]
        embed[owords_indicator] = numeral_embed

        return embed

class Word2VecFixed(nn.Module):

    def __init__(self, is_cuda=True, vocab_size=20000, embedding_size=300,  padding_idx=0, numeral_weighted_fn=weighted_log):
        super(Word2VecFixed, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.ivectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)
        self.ovectors = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=padding_idx)

        # initialize weights
        self.ivectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))
        self.ovectors.weight = nn.Parameter(t.cat([t.zeros(1, self.embedding_size), FT(self.vocab_size - 1, self.embedding_size).uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)]))

        self.is_cuda = is_cuda

        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

        self.numeral_weighted_fn = numeral_weighted_fn


    def get_numeral_embed_batch(self, numerals):
        """
        :param numerals: tensor of numerals!
        :return: embedding matrix of numerals
        """
        numerals.apply_(self.numeral_weighted_fn)
        embed = t.ones((self.embedding_size, len(numerals))).cuda()
        embed[0] = numerals
        embed = embed.transpose(0, 1)
        embed /= (self.embedding_size * 2)

        return embed # [num_of_numerals x embedding_size]


    def forward_i(self, data, iword_indicator, iword_numerals):
        v = LT(data)
        v = v.cuda(self.ivectors.weight.device) if self.is_cuda else v
        embed = self.ivectors(v)

        if iword_numerals.size()[0] == 0:
            return embed

        # prototype_weights = self.get_numeral_embed_weights_batch(iword_numerals) # [ num_of_numerals x prototype_size]
        # numeral_embed = t.matmul(prototype_weights, self.iprototypes_embeddings)
        # [num_of_numerals x prototype_size ]  x [prototype_size x embedding_size] => [num_of_numeral x embedding_size]
        numeral_embed = self.get_numeral_embed_batch(iword_numerals)

        embed[iword_indicator] = numeral_embed

        return embed

    def forward_o(self, data, owords_indicator, owords_numerals):
        v = LT(data)
        v = v.cuda(self.ivectors.weight.device) if self.is_cuda else v
        embed = self.ovectors(v)
        if owords_numerals.size()[0] == 0:
            return embed

        numeral_embed = self.get_numeral_embed_batch(owords_numerals)

        # [num_of_numerals x prototype_size ]  x [prototype_size x embedding_size] => [num_of_numeral x embedding_size]
        embed[owords_indicator] = numeral_embed

        return embed


class SGNS(nn.Module):

    def __init__(self, token_weights, numeral_weights, embedding, numerals, n_rate, scheme, vocab_size=20000, n_negs=20, numeral_pow=0.75):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.numerals = t.tensor(t.from_numpy(numerals.astype(np.float)), dtype=t.float32) # numeral counts numpy array
        self.numeral_size = len(self.numerals)
        self.n_negs = n_negs
        self.token_weights = None
        self.numeral_weights = None
        self.n_rate = n_rate # percentage of negative words sampled as numeral
        self.scheme = scheme
        self.numeral_pow = numeral_pow

        if token_weights is not None:
            wf = np.power(token_weights, 0.75)
            wf = wf / wf.sum()
            self.token_weights = FT(wf)

        if numeral_weights is not None:
            nf = np.power(numeral_weights, self.numeral_pow)
            nf = nf / nf.sum()
            self.numeral_weights = FT(nf)
            assert len(self.numeral_weights) == self.numeral_size

    def forward(self, iword, owords, iword_indicator, iword_numerals, owords_indicator, owords_numerals):

        assert self.scheme in ['none', 'prototype','GMM', 'fixed']

        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        # ivectors
        # assert t.sum(owords_indicator) == len(owords_numerals)
        # assert t.sum(iword_indicator) == len(iword_numerals)

        if self.scheme == 'none':
            iword_numerals = t.tensor([])
            owords_numerals = t.tensor([])

        ivectors = self.embedding.forward_i(iword, iword_indicator, iword_numerals).unsqueeze(2) # 4096 x 100 x 1
        ovectors = self.embedding.forward_o(owords, owords_indicator, owords_numerals) # 4096 x context_size x 100

        # Compute RNN input online is very inefficient
        # TODO implement negative sampling part for digital RNN

        all_neg_words = batch_size * context_size * self.n_negs
        if self.token_weights is not None:
	        # TODO add sampler for numbers
            nwords = t.multinomial(self.token_weights, all_neg_words, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()

        # for none scheme, suitable for NumeralAsToken and NumeralAsUnkNumeral
        if self.scheme == 'none':
            nvectors = self.embedding.forward_o(nwords, t.tensor([]), t.tensor([])).neg() # 4096 x (context_size * n_neg)x100

        else:
            masks = t.rand(batch_size, context_size * self.n_negs).le(self.n_rate)
            num_neg_numeral = masks.sum()

            if num_neg_numeral == 0: # sampled size equals to zero
                nvectors = self.embedding.forward_o(nwords, masks, t.tensor([])).neg() # 4096 x (context_size * n_neg)x100

            else:


                if self.numeral_weights is not None:
                    nnumerals_idxs = t.multinomial(self.numeral_weights, num_neg_numeral, replacement=True)
                else:
                    nnumerals_idxs = FT(num_neg_numeral).uniform_(0, self.numeral_size - 1).long()

                # ugly -_-, we cache all the gmm posterior, and use batch taking to directly get the weights
                # so we only need sample idxs and directly feed into the forward_i and o
                if self.scheme == 'GMM':
                    nnumerals = nnumerals_idxs
                else:
                    nnumerals = self.numerals[nnumerals_idxs]

                assert nnumerals.size()[0] == num_neg_numeral

                nvectors = self.embedding.forward_o(nwords, masks, nnumerals).neg() # 4096 x (context_size * n_neg)x100

        # cross entropy loss

        # Negative sampling loss
        # t.bmm(ovectors, ivectors) -> t.bmm(4096 x context_size x 100 , 4096 x 100 x 1) -> 4096 x context_size x 1
        # squeeze() - > 4096 x context_size, scores of each context word for each batch
        # .sigmoid().log() -> log probability of context word
        # .mean() -> mean over context words

        oloss = t.bmm(ovectors, ivectors).squeeze(2).sigmoid().log().mean(1)

        # t.bmm(nvectors, ivectors) -> t.bmm(4096 x (context_size * n_neg) x 100 , 4096 x 100 x 1) -> 4096 x (context_size * n_neg) x 1
        # squeeze() ->  4096 x (context_size * n_neg) , scores of each context word for each batch
        # .sigmoid().log() -> log probability of context word  4096 x (context_size * n_neg)
        # .view() ->  4096 x context_size x n_neg
        # .sum() -> 4096 x context_size -> sum over k negative words'
        # .mean() -> 4096 -> mean over context words

        # assert nvectors.size()[1] == context_size * self.n_negs
        nloss = t.bmm(nvectors, ivectors).squeeze(2).sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)

        scores = -(oloss + nloss).mean()

        # if t.isnan(scores):
        #
        #     print('oloss nan: ', t.isnan(oloss).sum())
        #     print('nloss nan: ', t.isnan(nloss).sum())
        #     print('nnumeral nan: ', t.isnan(nnumerals).sum())
        #     print('nnumeral inf: ', t.isinf(nnumerals).sum())
        #
        #
        #     exit(0)

        return scores


        # TODO: change to marginal ranking loss
        # oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid() #  4096 x context_size
        # oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid()
        # ovectors:  4096 x context_size x 100 -> （4096 x context_size ) x 100 x 1
        # nvectors:  4096 x (context_size * n_neg)x100 -> (4096 x context_size) x n_neg x 100
        # bmm() -> （4096 x context_size ) x n_neg x 1
        # dim_size = ivectors.size()[1]
        # nvectors = self.embedding.forward_o(nwords) # 4096 x (context_size * n_neg)x100, no .neg
        # temp_ovec = ovectors.view(batch_size*context_size, dim_size).unsqueeze(2)
        # temp_nvec = nvectors.view(batch_size*context_size, self.n_negs, dim_size)
        # context_neg_scores = t.bmm(temp_nvec, temp_ovec).squeeze().view(batch_size, context_size, self.n_negs) # # batch_size x context_size x n_neg
        # context_target_scores = t.bmm(ovectors, ivectors) # batch_size x context_size x 1
        # scores = 1 - context_target_scores.sigmoid() + context_neg_scores.sigmoid() # batch_size x context_size x n_neg
        # scores[scores < 0] = 0
        # _scores = t.sum(scores, dim=(2,1))

        # return _scores.mean()


    def forward_RNN(self, iword, owords, iword_indicator, iword_numerals, owords_indicator, owords_numerals, iword_numeral_length, owords_numeral_length):

        assert self.scheme in ['RNN', 'LSTM']

        batch_size = iword.size()[0]
        context_size = owords.size()[1]

        ivectors = self.embedding.forward_i(iword, iword_indicator, iword_numerals, iword_numeral_length).unsqueeze(2)  # 4096 x 100 x 1
        ovectors = self.embedding.forward_o(owords, owords_indicator, owords_numerals, owords_numeral_length)  # 4096 x context_size x 100

        all_neg_words = batch_size * context_size * self.n_negs
        if self.token_weights is not None:
            # TODO add sampler for numbers
            nwords = t.multinomial(self.token_weights, all_neg_words, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0, self.vocab_size - 1).long()


        masks = t.rand(batch_size, context_size * self.n_negs).le(self.n_rate)
        num_neg_numeral = masks.sum()

        if num_neg_numeral == 0:  # sampled size equals to zero
            nvectors = self.embedding.forward_o(nwords, masks,
                                                t.tensor([]), t.tensor([])).neg()  # 4096 x (context_size * n_neg)x100

        else:
            # TODO: Do we need sample in the large set? can we just sample in a small set or just the prototypes
            # may have a accelerate

            if self.numeral_weights is not None:
                nnumerals_idxs = t.multinomial(self.numeral_weights, num_neg_numeral, replacement=True)
            else:
                nnumerals_idxs = FT(num_neg_numeral).uniform_(0, self.numeral_size - 1).long()

            nnumerals = self.embedding.numeral_to_onehot[nnumerals_idxs]
            nword_numeral_length = self.embedding.numeral_to_length[nnumerals_idxs]
            assert nnumerals.size()[0] == num_neg_numeral

            nvectors = self.embedding.forward_o(nwords, masks, nnumerals, nword_numeral_length).neg()  # 4096 x (context_size * n_neg)x100

        oloss = t.bmm(ovectors, ivectors).squeeze(2).sigmoid().log().mean(1)

        nloss = t.bmm(nvectors, ivectors).squeeze(2).sigmoid().log().view(-1, context_size, self.n_negs).sum(2).mean(1)

        scores = -(oloss + nloss).mean()

        return scores
