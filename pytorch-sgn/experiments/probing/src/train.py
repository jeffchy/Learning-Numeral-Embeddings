import torch
from torch import nn
from dataset import ProbingListMaxDataset
from torch.utils.data import DataLoader
import argparse
from model import ListMax, CNN, TransformerModel
import pickle
import numpy as np
import random
import logging
import datetime
from prob_utils import create_dir

def train(args, logger, model_save_dir):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.infre:
        pretrain_embed = pickle.load(open('../embed_infre/{}'.format(args.embed), 'rb'))
        train_dataset = pickle.load(open('../data/train.infre.pkl', 'rb'))

    else:
        pretrain_embed = pickle.load(open('../embed/{}'.format(args.embed), 'rb'))
        train_dataset = pickle.load(open('../data/train.pkl','rb'))

    try:
        pretrain_embed = torch.from_numpy(pretrain_embed).float()
    except:
        pretrain_embed = pretrain_embed.float()

    train_dataset = ProbingListMaxDataset(train_dataset)
    dataLoader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True)
    if args.model == 'BiLSTM':
        model = ListMax(args.hidden_dim, pretrain_embed)
    elif args.model == 'CNN':
        model = CNN(pretrained=pretrain_embed)
    else:
        model = TransformerModel(pretrained=pretrain_embed, nhead=5, nhid=50, nlayers=2)


    # model = ListMaxTransformer(args.hidden_dim, pretrain_embed)
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dev_acc = 0
    best_dev_model = None
    best_dev_test_acc = 0
    counter = 0

    for epoch in range(1, args.n_epoch+1):
        train_loss = 0
        train_acc = 0
        model.train()
        iteration = 0
        for batch in dataLoader:
            optimizer.zero_grad()

            x = torch.stack(batch['input']) # 5 x bz
            y = batch['label'] # bz

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            output = model(x)
            loss = criterion(output, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == y).sum().item()

            iteration += 1
            # if iteration % args.iter_print == 0:
            #     logger.info('{}-{}-{}-{}'.format(epoch, iteration, train_loss, train_acc))

        train_loss = train_loss / len(train_dataset)
        train_acc = train_acc / len(train_dataset)
        dev_loss, dev_acc = val(model, mode='dev')
        test_loss, test_acc = val(model, mode='test')
        if dev_acc > best_dev_acc:
            best_dev_model = model.state_dict().copy()
            best_dev_acc = dev_acc
            best_dev_test_acc = test_acc
            counter = 0
        else:

            counter += 1

        logger.info('TRAIN: epoch:{}-loss:{}-acc:{}'.format(epoch, train_loss, train_acc))
        logger.info('DEV: epoch:{}-loss:{}-acc:{}'.format(epoch, dev_loss, dev_acc))
        logger.info('TEST: epoch:{}-loss:{}-acc:{}'.format(epoch, test_loss, test_acc))
        logger.info('BEST-DEV-ACC: {}, BEST-DEV-TEST-ACC:{}'.format(best_dev_acc, best_dev_test_acc))
        #
        # if counter > 30:
        #     break

    torch.save(best_dev_model, model_save_dir+'/model-{}-{}.pt'.format(best_dev_test_acc, args.lr))



def val(model, mode='dev'):
    temp = 'infre.' if args.infre else ''
    val_dataset = pickle.load(open('../data/{}.{}pkl'.format(mode, temp),'rb'))
    val_dataset = ProbingListMaxDataset(val_dataset)
    dataLoader = DataLoader(val_dataset, batch_size=args.batch_sz, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()

    val_loss = 0
    val_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in dataLoader:
            x = torch.stack(batch['input']) # 5 x bz
            y = batch['label'] # bz

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()
            val_acc += (output.argmax(1) == y).sum().item()

    val_loss = val_loss / len(val_dataset)
    val_acc = val_acc / len(val_dataset)

    return val_loss, val_acc




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', type=str, default='p-200', help="embed name")
    parser.add_argument('--lr', type=float, default='0.001', help="learning rate of ADAM")
    parser.add_argument('--seed', type=int, default=93221, help="random seed of training")
    parser.add_argument('--batch_sz', type=int, default=512, help="batch size")
    parser.add_argument('--n_epoch', type=int, default=5, help="num epoch")
    parser.add_argument('--hidden_dim', type=int, default=100, help="num hidden dim")
    parser.add_argument('--iter_print', type=int, default=100, help="infor for evry number of iters")
    parser.add_argument('--infre', type=int, default=0, help="if use infrequent mode")
    parser.add_argument('--model', type=str, default='BiLSTM', help="model")

    args = parser.parse_args()

    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m-%d-%H-%M-%S")

    assert args.infre in [0, 1]
    assert args.model in ['BiLSTM', 'CNN', 'T']

    model_dir = 'model_infre' if args.infre else 'model'
    model_save_dir = '../{}/{}/{}-{}'.format(model_dir, args.embed, datetime_str, args.model)


    # create dir
    create_dir('../{}/{}/'.format(model_dir, args.embed))
    create_dir(model_save_dir)

    logging.basicConfig(level=logging.INFO,
                        filename='{}/log.txt'.format(model_save_dir),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('{}'.format(model_save_dir))

    # for lr in [0.0001, 0.001, 0.0005, 0.0003, 0.01, 0.005]:
    #     args.lr = lr
    train(args, logger, model_save_dir)