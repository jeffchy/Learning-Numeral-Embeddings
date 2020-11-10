import json
import random
random.seed(0)


def train_dev_test_split(json_in):
    random.shuffle(json_in)
    train_rate, dev_rate, test_rate = 4.5/6, 0.5/6, 1/6
    length = len(json_in)
    train_json = json_in[:int(train_rate*length)]
    dev_json = json_in[int(train_rate*length): int(train_rate*length) + int(dev_rate*length)]
    test_json = json_in[int(train_rate*length) + int(dev_rate*length):]

    return train_json, dev_json, test_json

def create_sample_dataset(json_in, n):
    return random.sample(json_in, n)

def read_json(fname):
    with open(fname, 'r', encoding='utf-8') as fin:
        return json.load(fin)

def write_json(fname, obj):
    with open(fname, 'w', encoding='utf-8') as fout:
        json.dump(obj, fout)


if __name__ == '__main__':

    with open('./data/Numeracy_600K_article_title.json', 'r', encoding='utf-8') as fin:
        json_in_title = json.load(fin)

    with open('./data/Numeracy_600K_market_comment.json', 'r', encoding='utf-8') as fin:
        json_in_market = json.load(fin)

    json_sample_title = create_sample_dataset(json_in_title, 100)
    json_sample_market = create_sample_dataset(json_in_market, 100)

    with open('./data/Numeracy_600K_article_title.sample.json', 'w', encoding='utf-8') as fout:
        json.dump(json_sample_title, fout)

    with open('./data/Numeracy_600K_market_comment.sample.json', 'w', encoding='utf-8') as fout:
        json.dump(json_sample_market, fout)
