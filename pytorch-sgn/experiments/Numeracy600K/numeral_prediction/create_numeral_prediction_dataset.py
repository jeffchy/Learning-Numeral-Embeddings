import sys
sys.path.append('../')

from data_utils import read_json
from preprocess600k import replace_symbols
from tqdm import tqdm
import pickle

def to_numeral_prediction(json):

    print('Creating Numeral Prediciton Dataset')
    samples = []
    avg_l = 0
    avg_r = 0
    for js in tqdm(json):
        offset = js['offset']
        length = js['length']
        title = js['title']
        number_str = js['number']
        magnitude = js['magnitude']

        left_part = title[:offset]
        right_part = title[offset+length:]
        left_part = [i.lower() for i in replace_symbols(left_part).split()]
        right_part = [i.lower() for i in replace_symbols(right_part).split()]

        avg_l += len(left_part)
        avg_r += len(right_part)

        # 筛选吗？先不吧
        # number, context, magnitude, idx for number
        instance = (number_str, left_part + right_part, magnitude, len(left_part))

        if float(number_str) < 0:
            print(magnitude)

        samples.append(instance)

    return samples


if __name__ == '__main__':

    filename = 'data/Numeracy_600K_article_title.json'
    exp_dir = 'title_all'
    data = read_json(filename)
    numeral_prediction_data = to_numeral_prediction(data)
    pickle.dump(numeral_prediction_data, open('numeral_prediction/data.pkl', 'wb'))
