import codecs

if __name__ == '__main__':
    split_rate = 8/9
    path = 'wikiraw/bz2/train.txt'
    train_path = 'wikiraw/bz2/train.txt'
    test_path = 'wikiraw/bz2/dev.txt'

    with codecs.open(path, 'r', encoding='utf-8') as f:
        a = f.read().strip().split()
        # print(a[:100])
        # print(len(a))
        # print(len(a) * split_rate)
        train = a[:int(len(a) * split_rate)]
        test = a[int(len(a) * split_rate):]
        # print(bytes(' '.join(a[:100]), 'utf-8').decode('utf-8'))
        with codecs.open(train_path, 'w', encoding='utf-8') as f:
            f.write(bytes(' '.join(train), 'utf-8').decode('utf-8'))
        with codecs.open(test_path, 'w', encoding='utf-8') as f:
            f.write(bytes(' '.join(test), 'utf-8').decode('utf-8'))

