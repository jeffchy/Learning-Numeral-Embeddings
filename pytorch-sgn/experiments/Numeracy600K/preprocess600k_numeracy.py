from data_utils import read_json, write_json, train_dev_test_split
from tqdm import tqdm
from collections import Counter
import pickle
from config import Config
from data import EmbedLoader
import numpy as np
import random

random.seed(0)
np.random.seed(0)

def replace_symbols(string):
    string = string.replace('-', ' - ')
    string = string.replace('+', ' + ')
    string = string.replace('$', ' $ ')
    string = string.replace(':', ' : ')
    string = string.replace('/', ' / ')
    string = string.replace(',', ' , ')
    string = string.replace('#', ' # ')
    string = string.replace("'", "")
    string = string.replace('%', ' % ')

    # remove
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace("?", '')
    string = string.replace('!', '')
    string = string.replace(';', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('...', '')

    return string

def preprocess_title_dataset(json):
    stats = Counter()

    print('Preprocessing Dataset')
    for js in tqdm(json):
        offset = js['offset']
        length = js['length']
        title = js['title']
        number_str = js['number']

        # fill target number with TRT, also create
        title_new = title[:offset] + ' trt ' + title[offset+length:]
        title_origin = title[:offset] + ' ' + number_str + ' ' + title[offset+length:]
        title_new = replace_symbols(title_new)
        title_origin = replace_symbols(title_origin)

        title_new = ' '.join(([i.lower() for i in title_new.split()]))
        title_origin = ' '.join(([i.lower() for i in title_origin.split()]))

        del js['length']
        del js['offset']
        del js['publish_date']
        js['title'] = title_origin
        js['title_new'] = title_new

        stats[js['magnitude']] += 1
    return stats

def build_vocab(data):
    """
    :param data:
    :return the sorted vocabulary dict:
    """
    print('Building Vocabulary')
    vocab = Counter()
    for js in data:
        sent = js['title']
        vocab.update(sent.split())

    vocab['<trt>'] = 0
    vocab['<unk>'] = 0
    vocab['<pad>'] = 0

    # sort by value
    new_vocab = {}
    for k in sorted(vocab, key=vocab.get):
        new_vocab[k] = vocab[k]

    return new_vocab

if __name__ == '__main__':

    # filename = 'data/Numeracy_600K_article_title.sample.json'
    # exp_dir = 'title_sample'

    filename = 'data/Numeracy_600K_article_title.json'
    config = Config(None)

    exp_dir = config.exp_dir
    data = read_json(filename)
    stats = preprocess_title_dataset(data)

    print(stats)
    filename = 'exps/{}/preproc.json'.format(exp_dir)

    write_json(filename, data)

    vocab = build_vocab(data)
    word2idx = {word:idx for idx, word in enumerate(vocab)}
    idx2word = list(vocab.keys())

    print('Saving vocab and word-idxs')
    pickle.dump(word2idx, open('exps/{}/word2idx.pkl'.format(exp_dir), 'wb'))
    pickle.dump(idx2word, open('exps/{}/idx2word.pkl'.format(exp_dir), 'wb'))
    pickle.dump(vocab, open('exps/{}/vocab.pkl'.format(exp_dir), 'wb'))

    print("Train Dev Test splitting and saving")
    train_json, dev_json, test_json = train_dev_test_split(data)
    write_json('exps/{}/preproc.train.json'.format(exp_dir), train_json)
    write_json('exps/{}/preproc.dev.json'.format(exp_dir), dev_json)
    write_json('exps/{}/preproc.test.json'.format(exp_dir), test_json)

    print("Making Pretrain Embed Files")
    embed_loader = EmbedLoader(config, word2idx)

    config.logger.info("Start Creating Pretrained Embed Files")
    config.logger.info("Creating Protypes ... ")
    # # Prototype
    embed_name = 'Prototype'
    embedding_dim = 300
    embedding_epoch = 10
    preprocess_dir = '../../data/numeracy600k/NumeralAsNumeral/'
    alpha = 1.0
    log_space = False
    for i in [50, 100, 150, 200, 300]:
        embedding_path = '../../data/numeracy600k/save/som-{}-0/'.format(i)
        config.pretrained_embed_name = 'p-{}'.format(i)
        embed_loader.build_pretrained_embed(
            embed_name,
            embedding_path,
            preprocess_dir,
            embedding_epoch=embedding_epoch,
            log_space=log_space,
            mode='i'
        )

    config.logger.info("Creating Protypes Logs ... ")
    # Prototype Log
    log_space = True
    for i in [50, 100, 150, 200, 300]:
        embedding_path = '../../data/numeracy600k/save-log/som-{}-0/'.format(i)
        config.pretrained_embed_name = 'p-{}-log'.format(i)
        embed_loader.build_pretrained_embed(
            embed_name,
            embedding_path,
            preprocess_dir,
            embedding_epoch=embedding_epoch,
            log_space=log_space,
            mode='i'
        )

    config.logger.info("Creating GMM ... ")
    # GMM
    embed_name = 'GMM'
    log_space = False
    for i in [50, 100, 150, 200, 300] :
    # for i in [100, 300]:
        for j in ['soft', 'hard']:
            embedding_path = '../../data/numeracy600k/save/gmm-{}-rd-{}-0'.format(i, j)
            config.pretrained_embed_name = 'g-{}-{}'.format(i, j)
            embed_loader.build_pretrained_embed(
                embed_name,
                embedding_path,
                preprocess_dir,
                embedding_epoch=embedding_epoch,
                log_space=log_space,
                mode='i'
            )

    config.logger.info("Creating GMM Logs ... ")
    # GMM Log
    log_space = True
    for i in [50, 100, 150, 200, 300]:
        for j in ['soft', 'hard']:
            embedding_path = '../../data/numeracy600k/save-log/gmm-{}-rd-{}-0'.format(i, j)
            config.pretrained_embed_name = 'g-{}-{}-log'.format(i, j)
            embed_loader.build_pretrained_embed(
                embed_name,
                embedding_path,
                preprocess_dir,
                embedding_epoch=embedding_epoch,
                log_space=log_space,
                mode='i'
            )

    config.logger.info("Creating Fixed ... ")
    # Fixed
    embed_name = 'Fixed'
    embedding_path = '../../data/numeracy600k/save/fixed-0'
    config.pretrained_embed_name = 'Fixed'
    embed_loader.build_pretrained_embed(
        embed_name,
        embedding_path,
        preprocess_dir,
        embedding_epoch=embedding_epoch,
        mode='i'
    )

    config.logger.info("Creating Token ... ")
    # Fixed
    embed_name = 'Token'
    preprocess_dir = '../../data/numeracy600k/NumeralAsTokenUnkNumeral/'
    embedding_path = '../../data/numeracy600k/save/NumeralAsTokenUnkNumeral-0-550'
    config.pretrained_embed_name = 'token'
    embed_loader.build_pretrained_embed(
        embed_name,
        embedding_path,
        preprocess_dir,
        embedding_epoch=embedding_epoch,
        mode='i'
    )

    config.logger.info("Creating LSTM ... ")
    # LSTM
    embed_name = 'LSTM'
    preprocess_dir = '../../data/numeracy600k/NumeralAsNumeral/'
    embedding_path = '../../data/numeracy600k/save/LSTM-0'
    config.pretrained_embed_name = 'LSTM'
    embed_loader.build_pretrained_embed(
        embed_name,
        embedding_path,
        preprocess_dir,
        embedding_epoch=embedding_epoch,
        mode='i'
    )

    config.logger.info("Creating LSTM1... ")
    # LSTM1
    embed_name = 'LSTM'
    preprocess_dir = '../../data/numeracy600k/NumeralAsNumeralLSTM/'
    embedding_path = '../../data/numeracy600k/save/LSTM-1'
    config.pretrained_embed_name = 'LSTM1'
    embed_loader.build_pretrained_embed(
        embed_name,
        embedding_path,
        preprocess_dir,
        embedding_epoch=embedding_epoch,
        mode='i'
    )