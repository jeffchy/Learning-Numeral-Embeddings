import pickle
import codecs
import random
import math

def skipgram(sentence, i, window):
    iword = sentence[i]
    left = sentence[i - window: i]
    right = sentence[i + 1: i + 1 + window]
    return iword, left + right

def is_numeral(token):
	try:
		num = float(token)
		if num == float('inf') or num == float('-inf') or math.isnan(num):
			return False

		return True
	except ValueError:
		return False


if __name__ == '__main__':

    filtered_file_path = '../wikipedia/preprocess0.05Bnotable/test/NumeralAsNumeral/filtered.txt'

    save_path = 'build/data_0.05notable.pkl'
    step = 0
    window = 5
    data = []
    keep_rate = 0.8
    sample = 0
    random.seed(0)

    with codecs.open(filtered_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            step += 1
            if not step % 1000:
                print("working on {}kth line".format(step // 1000))
            line = line.strip()
            if not line:
                continue

            sent = line.split()

            for i in range(window, len(sent) - window):
                iword, owords = skipgram(sent, i, window)
                if is_numeral(iword) and random.random() < keep_rate:
                    data.append((iword, owords))
                    sample += 1

                    if sample % 1000 == 0:
                        print('...... extracting {} samples already'.format(sample))

    pickle.dump(data, open(save_path, 'wb'))
    print('Built done, saving data')



