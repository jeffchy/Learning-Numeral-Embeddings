import numpy as np
import pickle
import math

def to_numeral(token):
	try:
		num = np.float32(token)
		if num == np.float32('inf') or num == np.float32('-inf') or math.isnan(num):
			return None

		return num
	except ValueError:
		return None

def build_eval_dataset(dataset,nc,wc,idx2word,word2idx,vocab):
    assert len(vocab) == len(idx2word)
    assert len(word2idx) == len(idx2word)
    vocab_c = {}
    for key in vocab:
        if key in wc.keys():
           vocab_c[key] = wc[key]
    # vocab_c={key:wc[key] for key in vocab}
    processed_dataset = []
    for pair in dataset:# get rid of pair where unknown or numeral at
        flag = 1
        for item in pair[1]:
            if item not in vocab:
                flag = 0
        if flag == 1:
            processed_dataset.append(pair)
    context_weight = []
    for pair in processed_dataset:# calculate context word freq
        w = 0
        for item in pair[1]:
            w += vocab_c[item]
        context_weight.append(w)
    context_weight = np.array(context_weight)
    filtered_dataset = []
    # numeral_freq_lower_bound = np.percentile(np.array(list(nc.values())), 50)
    numeral_freq_lower_bound = 0

    idx = context_weight > np.percentile(context_weight, 75) # choose context  with context and in numeral freq of 50%-100%
    for counter, value in enumerate(idx):
        number = processed_dataset[counter - 1][0]
        if number not in nc:
            continue

        if value and  nc[number] > numeral_freq_lower_bound:
            # print(processed_dataset[counter-1][0])
            filtered_dataset.append(processed_dataset[counter - 1])

    # remove repeat numbers
    nnn = {}
    final = []
    for i in filtered_dataset:
        if i[0] in nnn:
            continue
        else:
            nnn[i[0]] = 0
            final.append(i)


    return final


def build_eval_naive(dataset):

    filtered_dataset = []
    for pair in dataset:

        flag = 1
        for item in pair[1]:
            if to_numeral(item) is not None:
                flag = 0

        if flag == 0: # no other numerals
            continue

        filtered_dataset.append(pair)
        # num = to_numeral(pair[0])
        # if abs(num) < 3000 and abs(num) > 0.1 and flag:
        #     filtered_dataset.append(pair)

    # remove repeat numbers
    # nnn = {}
    # final = []
    # for i in filtered_dataset:
    #     if i[0] in nnn:
    #         continue
    #     else:
    #         nnn[i[0]] = 0
    #         final.append(i)

    # return final
    return filtered_dataset




if __name__ == '__main__':

    dataset = pickle.load(open('build/data_1Bnotable.pkl','rb'))
    print(len(dataset))
    # nc = pickle.load(open('../wikipedia/preprocess0.05Bnotable/test/NumeralAsNumeral/nc.dat','rb'))
    # wc = pickle.load(open('../wikipedia/preprocess0.05Bnotable/test/NumeralAsNumeral/wc.dat','rb'))
    # idx2word = pickle.load(open('../wikipedia/preprocess0.05Bnotable/test/NumeralAsNumeral/idx2word.dat','rb'))
    # word2idx = pickle.load(open('../wikipedia/preprocess0.05Bnotable/test/NumeralAsNumeral/word2idx.dat','rb'))
    # vocab = set(idx2word)
    # filtered_dataset = build_eval_dataset(dataset, nc, wc, idx2word, word2idx, vocab)
    filtered_dataset = build_eval_naive(dataset)
    print(len(filtered_dataset))
    import random

    random.shuffle(filtered_dataset)
    filtered_dataset = filtered_dataset[:2000]
    pickle.dump(filtered_dataset[:int(len(filtered_dataset)*0.7)], open('build/data_filtered_1Bnotable.test.uniform.pkl','wb'))
    pickle.dump(filtered_dataset[int(len(filtered_dataset)*0.7):], open('build/data_filtered_1Bnotable.val.uniform.pkl','wb'))
