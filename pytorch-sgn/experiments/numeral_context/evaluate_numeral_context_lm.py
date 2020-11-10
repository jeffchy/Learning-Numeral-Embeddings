import pickle, numpy as np
import sys
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
                 nll_score,
                 nll_adj_score):

        self.dataset = dataset # 1400
        self.numerals = np.array([float(i[0]) for i in dataset])
        self.nll_score = nll_score
        self.nll_adj_score = nll_adj_score # 1400 x 1400
        self.varbose = False
        self.random = False
        all_num = [i[0] for i in self.dataset]
        self.numstr2idx = {num: idx for idx, num in enumerate(all_num)}
        self.error_thres = 10000


    def show_dataset_info(self):
        print('length of dataset: {} \ndataset examples: \n {} \nnumeral examples : {} \n '.format(
            len(self.dataset),
            self.dataset[:5],
            self.numerals[:10]
        ))

    def _get_score(self, i):
        idx = self.numstr2idx[i[0]]
        return [self.nll_score[idx], self.nll_adj_score[idx]]


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

        right = [0, 0,]

        for i in self.dataset:

            scores = self._get_score(i)

            for s in range(len(scores)):

                nn = np.argsort(scores[s])[-kbest:]

                if float(i[0]) in self.numerals[nn]:
                    right[s] += 1

        return [i / len(self.dataset) for i in right]

    def evaluate_RMSE(self):

        total_square_err = [[],[]]

        for i in self.dataset:

            scores = self._get_score(i)

            for s in range(len(scores)):

                pred = self.predict(scores[s])

                total_square_err[s].append((float(i[0]) - pred) ** 2)

        RMSE = np.sqrt(np.mean(total_square_err, axis=1))
        return RMSE

    def evaluate_avg_rank(self):
        total_rank = [0, 0]

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
        total_mape = [[],[]]
        total_mae = [[],[]]

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

    def eval_all(self, random=False, varbose=False):

        self.random = random
        self.varbose = varbose

        res = {}

        portions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
        a = [int(p * len(self.dataset)) for p in portions]

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
    #
    # @staticmethod
    # def draw(res):
    #     f = 1
    #     topk_label = ['base', 'score0', 'score1']
    #     fig = plt.figure(f)
    #     fig.set_figheight(7)
    #     fig.set_figwidth(14)
    #     count = 1
    #     for k in list(res.keys()):
    #         plt.subplot(1, len(list(res.keys())), count)
    #         R = res[k]
    #         x = R['x']
    #         topk = R['topk']
    #         for i in range(len(topk)):
    #             plt.plot(x, topk[i], label=topk_label[i])
    #         plt.title(k)
    #         plt.legend(loc='upper left')
    #         count += 1
    #
    #     plt.show()
    #     f += 1
    #
    #     fig = plt.figure(f)
    #     fig.set_figheight(7)
    #     fig.set_figwidth(14)
    #     for i in range(4):
    #         plt.subplot(2, 2, i + 1)
    #         for k in list(res.keys()):
    #             R = res[k]
    #             x = R['x']
    #             topk = R['topk']
    #             plt.plot(x, topk[i + 1], label=k)
    #         plt.title(topk_label[i + 1])
    #         plt.legend(loc='upper left')
    #     plt.show()
    #     f += 1
    #
    #     fig = plt.figure(f)
    #     fig.set_figheight(21)
    #     fig.set_figwidth(21)
    #
    #     mape_data = []
    #     mdape_data = []
    #     mdae_data = []
    #     RMSE_data = []
    #     mae_data = []
    #     avg_rank_data = []
    #     for k in list(res.keys()):
    #         d = res[k]['mape']
    #         for j in range(len(d)):
    #             mape_data.append([float(d[j]), topk_label[1:][j], 'mape', k])
    #
    #         d = res[k]['mdape']
    #         for j in range(len(d)):
    #             mdape_data.append([float(d[j]), topk_label[1:][j], 'mdape', k])
    #
    #         d = res[k]['mdae']
    #         for j in range(len(d)):
    #             mdae_data.append([float(d[j]), topk_label[1:][j], 'mdae', k])
    #
    #         d = res[k]['mae']
    #         for j in range(len(d)):
    #             mae_data.append([float(d[j]), topk_label[1:][j], 'mae', k])
    #
    #         d = res[k]['RMSE']
    #         for j in range(len(d)):
    #             RMSE_data.append([float(d[j]), topk_label[1:][j], 'RMSE', k])
    #
    #         d = res[k]['avg_rank']
    #         lab = ['r', 0, 1, 2, 3]
    #         for j in range(len(d)):
    #             avg_rank_data.append([d[j], lab[j], 'avg_rank', k])
    #
    #     mae_data = pd.DataFrame(mae_data, columns=['val', 'score', 'scheme', 'type'])
    #     mdape_data = pd.DataFrame(mdape_data, columns=['val', 'score', 'scheme', 'type'])
    #     mdae_data = pd.DataFrame(mdae_data, columns=['val', 'score', 'scheme', 'type'])
    #     RMSE_data = pd.DataFrame(RMSE_data, columns=['val', 'score', 'scheme', 'type'])
    #     mape_data = pd.DataFrame(mape_data, columns=['val', 'score', 'scheme', 'type'])
    #     avg_rank_data = pd.DataFrame(avg_rank_data, columns=['val', 'score', 'scheme', 'type'])
    #
    #     plt.subplot(3, 2, 1)
    #     sns.barplot(x='type', y='val', hue='score', data=mape_data)
    #     plt.title('MAPE')
    #
    #     plt.subplot(3, 2, 2)
    #     sns.barplot(x='type', y='val', hue='score', data=mae_data)
    #     plt.title('MAE')
    #
    #     plt.subplot(3, 2, 3)
    #     sns.barplot(x='type', y='val', hue='score', data=mdae_data)
    #     plt.title('MDAE')
    #
    #     plt.subplot(3, 2, 4)
    #     sns.barplot(x='type', y='val', hue='score', data=mdape_data)
    #     plt.title('MDAPE')
    #
    #     plt.subplot(3, 2, 5)
    #     sns.barplot(x='type', y='val', hue='score', data=RMSE_data)
    #     plt.title('RMSE')
    #
    #     plt.subplot(3, 2, 6)
    #     sns.barplot(x='type', y='val', hue='score', data=avg_rank_data)
    #     plt.title('AVG_RANK')
    #     plt.show()

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


