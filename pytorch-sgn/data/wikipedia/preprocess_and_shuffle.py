import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
# from gensim.corpora import WikiCorpus
from modified_wiki_corpus import WikiCorpus
import argparse
import re
import glob
import random
import os

# RE_TOKEN = re.compile(r"((\d+(,\d{3})*(\.\d+)?)|([^\W\d]+)|(\S))", re.UNICODE) # normal number
# RE_TOKEN = re.compile(r"(((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?)|([^\W\d]+)|(\S))") # number, negative, fraction
RE_TOKEN = re.compile(
    r"(((\d{1,2}:\d{2} )|(\w+-\d+)|(\w+-\w+)|((-?\d+(,\d{3})*(\.\d+)?)\/(-?\d+(,\d{3})*(\.\d+)?))|(-?\d+(,\d{3})*(\.\d+)?))|([^\W\d]+)|(\S))",
    re.UNICODE)

RE_TIME = re.compile(r'\d{1,2}:\d{2} ', re.UNICODE)

def custom_tokenizer(text, token_min_len=1, token_max_len=20, lower=True):
    tokens = []
    for token in RE_TOKEN.findall(text):
        if isinstance(token, tuple):
            token = token[0]

        # time like 3:38
        if RE_TIME.findall(token) != []:
            temp_list = token.split(':')
            if float(temp_list[1]) > 60:
                continue
            else:
                tokens += [temp_list[0].strip(), 'minutes', temp_list[1].strip(), 'seconds']

            continue

        if len(token) > token_max_len or len(token) < token_min_len:
            pass

        else:
            if lower:
                token = token.lower()

            tokens.append(token)

    return tokens


def make_corpus(in_f, out_f):
    """Convert Wikipedia xml dump file to text corpus"""
    all_token_num = 0
    articles = 0
    all_articles = []
    random.seed(888)

    files = glob.glob(in_f + '/*.bz2')
    print(files)

    for f in files:
        print('processing file: {}'.format(f), 'num of tokens now: {}'.format(all_token_num))
        wiki = WikiCorpus(f, tokenizer_func=custom_tokenizer)


        for text in wiki.get_texts():
            all_token_num += len(text)
            # output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')

            articles = articles + 1
            all_articles.append(text)
            if (articles % 10000 == 0):
                print('Processed ' + str(articles) + ' articles, ' + str(all_token_num) + 'tokens in total')

    print('Processing complete!')
    print('Shuffling large dataset')
    random.shuffle(all_articles)
    print('Shuffling large dataset complete ... start writing')
    train_path = os.path.join(out_f, 'train.txt')
    test_path = os.path.join(out_f, 'test.txt')

    counter = 0
    rate = 0.8

    output = open(train_path, 'w', encoding='utf-8')
    for a in all_articles[:int(rate * len(all_articles))]:
        output.write(bytes(' '.join(a), 'utf-8').decode('utf-8') + '\n')
        counter += 1
        if (counter % 10000 == 0):
            print('Wrote {} articles to train.txt'.format(counter))
    output.close()

    counter = 0
    output = open(test_path, 'w', encoding='utf-8')
    for a in all_articles[int(rate * len(all_articles)):]:
        output.write(bytes(' '.join(a), 'utf-8').decode('utf-8') + '\n')
        counter += 1
        if (counter % 10000 == 0):
            print('Wrote {} articles to test.txt'.format(counter))
    output.close()

    print('Writing complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="wiki dumps dir path")
    parser.add_argument('--output', type=str, help="corpus file dir path")
    args = parser.parse_args()
    infile = args.input
    outfile = args.output
    make_corpus(infile, outfile)
