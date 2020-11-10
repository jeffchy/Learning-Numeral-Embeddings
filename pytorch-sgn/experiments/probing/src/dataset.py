import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import random
import logging

def create_dataset(numerals):
    random.seed(0)
    numerals = np.array(numerals)
    _len = len(numerals)
    N = 50000
    all_instance = []

    for i in range(N):
        instance = random.sample(list(np.arange(_len)), 5)
        numbers = numerals[instance]
        label = np.argmax(numbers)
        all_instance.append([instance, label])

    pickle.dump(all_instance[:int(N*0.7)], open('../data/train.infre.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.7):int(N*0.8)], open('../data/dev.infre.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.8):], open('../data/test.infre.pkl', 'wb'))


def create_dataset_2(numerals):
    random.seed(0)
    numerals = np.array(numerals)
    _len = len(numerals)
    N = 50000
    all_instance = []

    for i in range(N):
        instance = random.sample(list(np.arange(_len)), 3)
        numbers = numerals[instance]

        target = numbers[-1]
        srcl = numbers[0]
        srcr = numbers[1]

        diff1 = np.abs(target - srcl)
        diff2 = np.abs(target - srcr)
        if diff1 < diff2:
            label = 0
        else:
            label = 1
        all_instance.append([instance, label])

    pickle.dump(all_instance[:int(N*0.7)], open('../data/train.closer.infre.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.7):int(N*0.8)], open('../data/dev.closer.infre.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.8):], open('../data/test.closer.infre.pkl', 'wb'))

def create_dataset_3(numerals):
    from som.intopolate import weighted_log
    random.seed(0)
    numerals = np.array(numerals)
    _len = len(numerals)
    N = 200000
    all_instance = []

    for i in range(N):
        instance = random.sample(list(np.arange(_len)), 2)
        numbers = numerals[instance]
        srcl = numbers[0]
        srcr = numbers[1]
        diff = srcl - srcr
        # log_diff = float(weighted_log(diff))
        all_instance.append([instance, diff])

    pickle.dump(all_instance[:int(N*0.6)], open('../data/train.diff.lownolog.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.6):int(N*0.8)], open('../data/dev.diff.lownolog.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.8):], open('../data/test.diff.lownolog.pkl', 'wb'))

def create_dataset_3_no_corr(numerals):
    from som.intopolate import weighted_log
    random.seed(0)
    numerals = np.array(numerals)
    _len = len(numerals)
    N = 200000
    all_instance = []
    k = int(N / len(numerals)) + 1

    for i in range(_len):
        srcl = numerals[i]
        srcr_idxs = np.random.choice(_len,k,replace=False)
        for srcr_idx in srcr_idxs:
            srcr = numerals[srcr_idx]
            diff = srcl - srcr
            all_instance.append([[i, srcr_idx], diff])

    random.shuffle(all_instance)
    all_instance = all_instance[:N]

    print("len 1: {}".format(len(all_instance)))
    print("len 2: {}".format(len(set([str(m) for m in all_instance]))))

    pickle.dump(all_instance[:int(N*0.6)], open('../data/train.diff.highnolognocorr.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.6):int(N*0.8)], open('../data/dev.diff.highnolognocorr.pkl', 'wb'))
    pickle.dump(all_instance[int(N*0.8):], open('../data/test.diff.highnolognocorr.pkl', 'wb'))


def create_dataset_4(numerals):
    from som.intopolate import weighted_log
    random.seed(0)
    numerals = np.array(numerals)
    _len = len(numerals)
    all_instance = []

    for i in range(_len):
        numbers = numerals[i]
        # log_diff = float(weighted_log(diff))
        all_instance.append([i, numbers])

    print(all_instance[-1])
    random.shuffle(all_instance)
    print(all_instance[-1])

    pickle.dump(all_instance[:int(_len*0.6)], open('../data/train.diff.lowregre.pkl', 'wb'))
    pickle.dump(all_instance[int(_len*0.6):int(_len*0.8)], open('../data/dev.diff.lowregre.pkl', 'wb'))
    pickle.dump(all_instance[int(_len*0.8):], open('../data/test.diff.lowregre.pkl', 'wb'))


class ProbingListMaxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input': self.dataset[idx][0],
            'label': self.dataset[idx][1]
        }


if __name__ == '__main__':
    numerals = pickle.load(open('../../mag_num/data/numerals_mag_1B.pkl', 'rb'))
    # numerals = pickle.load(open('../../mag_num/data/numerals_mag_infre.pkl', 'rb'))
    # numerals = pickle.load(open('../../mag_num/data/numerals_mag_infre.pkl', 'rb'))

    create_dataset_3_no_corr(numerals)


