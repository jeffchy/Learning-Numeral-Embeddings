import glob
import os


if __name__ == '__main__':
    txts = glob.glob('train_dev.txt') # only use train.txt
    outf_path = 'leyanfemale.txt'

    fout = open(outf_path,'w',encoding='utf-8')
    tokens = []

    end_token = '.'

    for t in txts:
        with open(t, 'r', encoding='utf-8') as fin:
            in_lines = fin.readlines()

        for l in in_lines:

            if len(l) >= 3:
                word = l.strip().split()[0]
                tokens.append(word)

            elif l == '\n':
                tokens.append(end_token)

    fout.write(' '.join(tokens))
    fout.close()