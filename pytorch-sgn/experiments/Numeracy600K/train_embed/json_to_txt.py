import sys
sys.path.append('../')

from data_utils import read_json
from tqdm import tqdm
import pickle

if __name__ == '__main__':
    jsons = read_json('exps/title_all_correct/preproc.train.json')
    with open('train_embed/corpus.txt', 'w', encoding='utf-8') as fout:
        for js in tqdm(jsons):
            sent = js['title']
            fout.writelines(sent)


