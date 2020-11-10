import pickle, numpy as np
import sys
import torch as t
sys.path.append("../../")
import matplotlib.pyplot as plt
from som.intopolate import weighted_log
from utils.number_handler import to_numeral
import glob
import seaborn as sns
import pandas as pd
import json

sns.set(style='darkgrid')

class Evaluator(object):

    def __init__(self,
                 dataset,
                 wc,
                 nc,
                 idx2word,
                 word2idx):

        self.dataset = dataset
        self.wc = wc
        self.nc = nc
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.numerals = np.array([float(i[0]) for i in dataset])
        self.show_dataset_info()
        self.varbose = False
        self.random = False
        self.error_thres = 10000


        # loads or compute later
        self.idx2vec_i = None
        self.numeral_embed_i = None
        self.mean_vec_i = None
        self.idx2vec_o = None
        self.numeral_embed_o = None
        self.mean_vec_o = None
        self.norms = None



    def show_dataset_info(self):
        print('length of dataset: {} \ndataset examples: \n {} \nnumeral examples : {} \n '.format(
            len(self.dataset),
            self.dataset[:5],
            self.numerals[:10]
        ))


    def compute_mean_vec(self, idx2vec):
        vecs = []
        c_dict = {}
        for i in self.dataset:
            C = i[1]
            for c in C:
                if c not in c_dict:
                    if c in self.word2idx:
                        vecs.append(idx2vec[self.word2idx[c]])
                else:
                    c_dict[c] += 1

        mean_vec = np.array(vecs)
        print('Computing mean vectors, missing {} vectors'.format(
            len(self.dataset) * len(self.dataset[0][1]) - len(mean_vec)))

        return np.mean(mean_vec, axis=0)

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    def compute_norm_factor(self):
        # N x H
        # normalization factor for p(c_j|n_i)
        norms = np.zeros(self.numeral_embed_i.shape[0])
        c_dict = {}
        for i in self.dataset:
            C = i[1]
            for c in C:
                if c not in c_dict:
                    try:
                        wordvec = self.idx2vec_o[self.word2idx[c]]
                    except KeyError:
                        wordvec = self.mean_vec_o

                    score = np.dot(wordvec, np.array(self.numeral_embed_i).T)  # N x 1 # remove the replicated c?
                    norms += np.exp(score)
                    c_dict[c] = 0
                else:
                    c_dict[c] += 1

        return norms, c_dict

    def compute_norm_factor_all(self):
        # N x H
        # normalization factor for p(c_j|n_i)
        mat = np.dot(self.idx2vec_o, np.array(self.numeral_embed_i).T)
        mat = np.exp(mat)
        norms = np.sum(mat, axis=0)

        return norms


    def _get_score(self, i):

        s0 = np.zeros((len(self.numerals),))
        s3 = np.zeros((len(self.numerals),))
        for c in i[1]:
            try:
                wordvec = self.idx2vec_o[self.word2idx[c]]
            except KeyError:
                wordvec = self.mean_vec_o

            score = np.dot(wordvec, np.array(self.numeral_embed_i).T)
            s0 += score

            try:
                wordvec = self.idx2vec_i[self.word2idx[c]]
            except KeyError:
                wordvec = self.mean_vec_i

            score = np.dot(wordvec, np.array(self.numeral_embed_o).T)
            s3 += score



        s1 = s0 - len(i[1]) * np.log(self.norms)
        s2 = s3 + s1

        return [s0, s1, s2, s3]


    def predict(self, score):
        pred_idx = np.argsort(score)[-1]
        reduplic = np.where(score == score[pred_idx])

        if len(reduplic) > 1:
            pred_idx = np.random.choice(reduplic)
            print('predict UNK_W, randomly choose a word')

        # random:
        if self.random:
            pred_idx = np.random.choice(len(score))

        pred = self.numerals[pred_idx]

        return pred

    def evaluate_kbest(self, kbest):

        right = [0, 0, 0, 0]

        for i in self.dataset:

            scores = self._get_score(i)

            for s in range(len(scores)):

                nn = np.argsort(scores[s])[-kbest:]

                if float(i[0]) in self.numerals[nn]:
                    right[s] += 1

        return [i / len(self.dataset) for i in right]

    def evaluate_RMSE(self):

        total_square_err = [[],[],[],[]]

        for i in self.dataset:

            scores = self._get_score(i)

            for s in range(len(scores)):

                pred = self.predict(scores[s])

                total_square_err[s].append((float(i[0]) - pred) ** 2)

        RMSE = np.sqrt(np.mean(total_square_err, axis=1))
        return RMSE

    def evaluate_avg_rank(self):
        total_rank = [0, 0, 0, 0]

        for i in self.dataset:

            scores = self._get_score(i)

            for s in range(len(scores)):

                nn = np.argsort(scores[s])[::-1]

                rank = 0
                for k in self.numerals[nn]:
                    rank += 1
                    if k == float(i[0]):
                        break

                total_rank[s] += rank

        avg_rank = [i / len(self.dataset) for i in total_rank]
        return avg_rank

    def evaluate_avg_absolute_and_percent_err(self):  # emmmm
        total_mape = [[],[],[],[]]
        total_mae = [[],[],[],[]]

        for i in self.dataset:

            scores = self._get_score(i)

            for s in range(len(scores)):

                pred = self.predict(scores[s])

                mape = abs((float(i[0]) - pred) / max(abs(float(i[0])), 1e-4))
                mae = abs((float(i[0]) - pred))

                if mape > self.error_thres and self.varbose:
                    print('truth: {}, pred: {}, mape: {}, score: {}'.format(float(i[0]), pred, mape, s))
                    print('data sample: ',i[1][:5] + [i[0]] + i[1][-5:])

                if mae > self.error_thres and self.varbose:
                    print('truth: {}, pred: {}, mae: {}, score: {}'.format(float(i[0]), pred, mae, s))
                    print('data sample: ',i[1][:5] + [i[0]] + i[1][-5:])

                total_mape[s].append(mape)
                total_mae[s].append(mae)

        if self.varbose:
            temp = np.array(total_mape)/np.sum(total_mape, axis=1)[:, np.newaxis]
            print('mape outlier portion: ', np.sort(temp)[:,-10:])
            print('top 10 mape portion: ', np.sum(np.sort(temp)[:,-10:], axis=1))


            temp = np.array(total_mae)/np.sum(total_mae, axis=1)[:, np.newaxis]
            print('mae outlier portion: ', np.sort(temp)[:,-10:])
            print('top 10 mae portion: ', np.sum(np.sort(temp)[:,-10:], axis=1))




        avg_mape = np.mean(total_mape, axis=1)
        median_mape = np.median(total_mape, axis=1)
        avg_mae = np.mean(total_mae, axis=1)
        median_mae = np.median(total_mae, axis=1)

        return avg_mape, median_mape, avg_mae, median_mae


    def load_prototype(self,
                       idx2vec_i,
                       idx2vec_o,
                       trained_prototypes,
                       alpha=1.0,
                       log_space=False):

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

        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o
        self.mean_vec_i = self.compute_mean_vec(idx2vec_i)
        self.mean_vec_o = self.compute_mean_vec(idx2vec_o)

        prototypes = trained_prototypes['prototypes']
        prototypes2vec_i = trained_prototypes['i_embedding']
        prototypes2vec_o = trained_prototypes['o_embedding']
        if log_space:
            # if log_space, the prototype is already transformed
            transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).view(-1, 1)
        else:
            transformed_protp = t.tensor(t.from_numpy(prototypes), dtype=t.float32).apply_(weighted_log).view(-1, 1)

        prototype_weights = get_numeral_embed_weights_batch(t.tensor(t.from_numpy(self.numerals),dtype=t.float),transformed_protp, alpha=alpha)
        self.numeral_embed_i = t.matmul(prototype_weights.transpose(0,1),t.from_numpy(prototypes2vec_i))
        self.numeral_embed_o = t.matmul(prototype_weights.transpose(0,1),t.from_numpy(prototypes2vec_o))
        self.norms, _ = self.compute_norm_factor()


    def load_GMM(self,
                idx2vec_i,
                idx2vec_o,
                trained_prototypes,
                gmm,
                log_space=False):

        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o
        self.mean_vec_i = self.compute_mean_vec(idx2vec_i)
        self.mean_vec_o = self.compute_mean_vec(idx2vec_o)
        # print(self.numerals)
        if log_space:
            prototype_weights = gmm.predict_proba(np.array([weighted_log(x) for x in self.numerals]).reshape(-1, 1))
            # print(self.numerals[0], prototype_weights[0], np.argmax(prototype_weights[0]), gmm.means_[np.argmax(prototype_weights[0]),])

        else:
            prototype_weights = gmm.predict_proba(self.numerals.reshape(-1, 1))

        prototypes2vec_i = trained_prototypes['i_embedding']
        prototypes2vec_o = trained_prototypes['o_embedding']
        self.numeral_embed_i = t.from_numpy(np.matmul(prototype_weights, prototypes2vec_i))  # (93, 150)
        self.numeral_embed_o = t.from_numpy(np.matmul(prototype_weights, prototypes2vec_o))  # (93, 150)
        self.norms, _ = self.compute_norm_factor()


    def load_fixed(self,
                idx2vec_i,
                idx2vec_o,):

        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o
        self.mean_vec_i = self.compute_mean_vec(idx2vec_i)
        self.mean_vec_o = self.compute_mean_vec(idx2vec_o)


        def get_numeral_embed_batch(numerals):
            """
            :param numerals: tensor of numerals!
            :return: embedding matrix of numerals
            """
            embedding_size = self.mean_vec_i.shape[0]
            numerals.apply_(weighted_log)
            embed = t.ones((embedding_size, len(numerals)))
            embed[0] = numerals
            embed = embed.transpose(0, 1)
            embed /= (embedding_size * 2)

            return embed  # [num_of_numerals x embedding_size]

        self.numeral_embed_i = get_numeral_embed_batch(t.tensor(t.from_numpy(self.numerals),dtype=t.float)) # (93, 150)
        self.numeral_embed_o = get_numeral_embed_batch(t.tensor(t.from_numpy(self.numerals),dtype=t.float))   # (93, 150)
        self.norms, _ = self.compute_norm_factor()

    def load_LSTM(self,
                  idx2vec_i,
                  idx2vec_o,
                  LSTM_model_path):

        LSTM_params = t.load(LSTM_model_path, map_location='cpu')

        digital_RNN_i = t.nn.LSTM(14, idx2vec_i.shape[1], 1, batch_first=True)
        digital_RNN_o = t.nn.LSTM(14, idx2vec_i.shape[1], 1, batch_first=True)

        digital_RNN_i.bias_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.bias_hh_l0'])
        digital_RNN_i.bias_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.bias_ih_l0'])
        digital_RNN_i.weight_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.weight_hh_l0'])
        digital_RNN_i.weight_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_i.weight_ih_l0'])
        digital_RNN_o.bias_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.bias_hh_l0'])
        digital_RNN_o.bias_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.bias_ih_l0'])
        digital_RNN_o.weight_hh_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.weight_hh_l0'])
        digital_RNN_o.weight_ih_l0 = t.nn.Parameter(LSTM_params['embedding.digital_RNN_o.weight_ih_l0'])

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
        self.mean_vec_i = self.compute_mean_vec(idx2vec_i)
        self.mean_vec_o = self.compute_mean_vec(idx2vec_o)
        self.numeral_embed_i = np.zeros((len(self.dataset), idx2vec_i.shape[1]))
        self.numeral_embed_o = np.zeros((len(self.dataset), idx2vec_i.shape[1]))

        counter = 0
        for i in self.dataset:
            temp = convert_digit_to_tensor(i[0]).view(1, 20, 14)
            _, (hn, cn) = digital_RNN_i(temp)
            e = hn.squeeze().detach().numpy()
            self.numeral_embed_i[counter] = e
            counter += 1

        counter = 0
        for i in self.dataset:
            temp = convert_digit_to_tensor(i[0]).view(1, 20, 14)
            _, (hn, cn) = digital_RNN_o(temp)
            e = hn.squeeze().detach().numpy()
            self.numeral_embed_o[counter] = e
            counter += 1

        self.norms, _ = self.compute_norm_factor()

    def load_TOKEN(self,
                   idx2vec_i,
                   idx2vec_o,
                   ):
        # CAUTION, need to reload the word2idx, nc ...

        numerals_str = [i[0] for i in self.dataset]
        self.numeral_embed_i = np.zeros((len(self.dataset), idx2vec_i.shape[1]))
        self.numeral_embed_o = np.zeros((len(self.dataset), idx2vec_i.shape[1]))
        self.mean_vec_i = self.compute_mean_vec(idx2vec_i)
        self.mean_vec_o = self.compute_mean_vec(idx2vec_o)
        self.idx2vec_i = idx2vec_i
        self.idx2vec_o = idx2vec_o

        UNK_idxs = []

        for i in range(len(numerals_str)):
            try:
                self.numeral_embed_i[i] = idx2vec_i[self.word2idx[numerals_str[i]]]

            except:
                print('oov:', numerals_str[i])
                UNK_idxs.append(i)
                self.numeral_embed_i[i] = idx2vec_i[self.word2idx['<UNK_N>']]

        for i in range(len(numerals_str)):
            try:
                self.numeral_embed_o[i] = idx2vec_o[self.word2idx[numerals_str[i]]]

            except:
                print('oov:', numerals_str[i])
                self.numeral_embed_o[i] = idx2vec_o[self.word2idx['<UNK_N>']]

        self.norms, _ = self.compute_norm_factor()




    def eval_all(self, random=False, varbose=False):

        self.random = random
        self.varbose = varbose

        res = {}

        if len(self.dataset) == 100:
            a = [1, 3, 10, 15, 25, 40, 50, 60, 80, 100]
        elif len(self.dataset) == 400:
            a = [3, 10, 15, 25, 40, 100, 150, 200, 250, 320]
        else:
            print("Wrong dataset size")
            raise KeyboardInterrupt

        base_res = [i / len(self.dataset) for i in a]
        base_rank = 0.5 * (len(self.dataset) + 1)

        res['x'] = a
        res['topk'] = np.vstack((np.array([base_res]), np.array([self.evaluate_kbest(i) for i in a]).T))
        res['RMSE'] = self.evaluate_RMSE()
        avg_mape, median_mape, avg_mae, median_mae = self.evaluate_avg_absolute_and_percent_err()

        res['mape'] = avg_mape
        res['mae'] = avg_mae
        res['mdae'] = median_mae
        res['mdape'] = median_mape
        res['avg_rank'] = [base_rank] + self.evaluate_avg_rank()

        return res

    @staticmethod
    def draw(res):
        f = 1
        topk_label = ['base', 'score0', 'score1', 'score2', 'score3']
        fig = plt.figure(f)
        fig.set_figheight(7)
        fig.set_figwidth(14)
        count = 1
        for k in list(res.keys()):
            plt.subplot(1, len(list(res.keys())), count)
            R = res[k]
            x = R['x']
            topk = R['topk']
            for i in range(len(topk)):
                plt.plot(x, topk[i], label=topk_label[i])
            plt.title(k)
            plt.legend(loc='upper left')
            count += 1

        plt.show()
        f += 1

        fig = plt.figure(f)
        fig.set_figheight(7)
        fig.set_figwidth(14)
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            for k in list(res.keys()):
                R = res[k]
                x = R['x']
                topk = R['topk']
                plt.plot(x, topk[i + 1], label=k)
            plt.title(topk_label[i + 1])
            plt.legend(loc='upper left')
        plt.show()
        f += 1

        fig = plt.figure(f)
        fig.set_figheight(21)
        fig.set_figwidth(21)

        mape_data = []
        mdape_data = []
        mdae_data = []
        RMSE_data = []
        mae_data = []
        avg_rank_data = []
        for k in list(res.keys()):
            d = res[k]['mape']
            for j in range(len(d)):
                mape_data.append([float(d[j]), topk_label[1:][j], 'mape', k])

            d = res[k]['mdape']
            for j in range(len(d)):
                mdape_data.append([float(d[j]), topk_label[1:][j], 'mdape', k])

            d = res[k]['mdae']
            for j in range(len(d)):
                mdae_data.append([float(d[j]), topk_label[1:][j], 'mdae', k])

            d = res[k]['mae']
            for j in range(len(d)):
                mae_data.append([float(d[j]), topk_label[1:][j], 'mae', k])

            d = res[k]['RMSE']
            for j in range(len(d)):
                RMSE_data.append([float(d[j]), topk_label[1:][j], 'RMSE', k])

            d = res[k]['avg_rank']
            lab = ['r', 0, 1, 2, 3]
            for j in range(len(d)):
                avg_rank_data.append([d[j], lab[j], 'avg_rank', k])

        mae_data = pd.DataFrame(mae_data, columns=['val', 'score', 'scheme', 'type'])
        mdape_data = pd.DataFrame(mdape_data, columns=['val', 'score', 'scheme', 'type'])
        mdae_data = pd.DataFrame(mdae_data, columns=['val', 'score', 'scheme', 'type'])
        RMSE_data = pd.DataFrame(RMSE_data, columns=['val', 'score', 'scheme', 'type'])
        mape_data = pd.DataFrame(mape_data, columns=['val', 'score', 'scheme', 'type'])
        avg_rank_data = pd.DataFrame(avg_rank_data, columns=['val', 'score', 'scheme', 'type'])

        plt.subplot(3, 2, 1)
        sns.barplot(x='type', y='val', hue='score', data=mape_data)
        plt.title('MAPE')

        plt.subplot(3, 2, 2)
        sns.barplot(x='type', y='val', hue='score', data=mae_data)
        plt.title('MAE')

        plt.subplot(3, 2, 3)
        sns.barplot(x='type', y='val', hue='score', data=mdae_data)
        plt.title('MDAE')

        plt.subplot(3, 2, 4)
        sns.barplot(x='type', y='val', hue='score', data=mdape_data)
        plt.title('MDAPE')

        plt.subplot(3, 2, 5)
        sns.barplot(x='type', y='val', hue='score', data=RMSE_data)
        plt.title('RMSE')

        plt.subplot(3, 2, 6)
        sns.barplot(x='type', y='val', hue='score', data=avg_rank_data)
        plt.title('AVG_RANK')
        plt.show()

    def flod_score(self, res, score = 0):
        from copy import deepcopy
        new_res = deepcopy(res)

        for model, model_res in res.items():
            for attr, attr_res in new_res[model].items():

                if attr == 'avg_rank' or attr == 'topk':
                    new_res[model][attr] = new_res[model][attr][score + 1]
                elif attr != 'x':
                    new_res[model][attr] = new_res[model][attr][score]

        return pd.DataFrame(new_res).transpose()

    def flod_model(self, res, model = None):
        from copy import deepcopy
        new_res = {}

        for attr, attr_res in res[model].items():

            if attr == 'avg_rank':
                new_res[attr] = attr_res[1:]
            elif attr == 'x' or attr == 'topk':
                pass
            else:
                new_res[attr] = attr_res

        print(new_res)
        return pd.DataFrame.from_dict(new_res).transpose()


