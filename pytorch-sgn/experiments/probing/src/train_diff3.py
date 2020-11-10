import torch
from torch import nn
from dataset import ProbingListMaxDataset
from torch.utils.data import DataLoader
import argparse
from model import ListMax, CNN, TransformerModel, MLP, MLP3Diff
import pickle
import numpy as np
import random
import logging
import datetime
from prob_utils import create_dir

def train(args, logger, model_save_dir, val_dataset, test_dataset, train_dataset):
    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.infre:
        pretrain_embed = pickle.load(open('../embed_infre_low/{}'.format(args.embed), 'rb'))
    else:
        pretrain_embed = pickle.load(open('../embed_infre_low/{}'.format(args.embed), 'rb'))


    try:
        pretrain_embed = torch.from_numpy(pretrain_embed).float()
    except:
        pretrain_embed = pretrain_embed.float()

    dataLoader = DataLoader(train_dataset, batch_size=args.batch_sz, shuffle=True)
    if args.model == 'MLP':
        model = MLP3Diff(args.hidden_dim, pretrain_embed)

    # model = ListMaxTransformer(args.hidden_dim, pretrain_embed)
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=args.gamma)

    best_dev_loss = float('+inf')
    best_dev_model = None
    best_dev_test_loss = 0
    counter = 0

    for epoch in range(1, args.n_epoch+1):
        train_loss = 0
        model.train()
        iteration = 0
        for batch in dataLoader:

            optimizer.zero_grad()
            x = torch.stack(batch['input']) # 5 x bz
            y = batch['label'].float() # bz

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            output = model(x)
            loss = criterion(output, y)
            train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            iteration += 1
            # if iteration % args.iter_print == 0:
            #     logger.info('{}-{}-{}-{}'.format(epoch, iteration, train_loss, train_acc))

        train_loss = train_loss / len(train_dataset)
        dev_loss = val(model, val_dataset)
        test_loss = val(model, test_dataset)

        scheduler.step()

        if dev_loss < best_dev_loss:
            best_dev_model = model.state_dict().copy()
            best_dev_loss = dev_loss
            best_dev_test_loss = test_loss
            counter = 0
        else:

            counter += 1

        if epoch % 5 == 0:
            logger.info('=================================================')
            logger.info('TRAIN: epoch:{}-loss:{}'.format(epoch, train_loss))
            logger.info('DEV: epoch:{}-loss:{}'.format(epoch, dev_loss))
            logger.info('TEST: epoch:{}-loss:{}'.format(epoch, test_loss))
            logger.info('BEST-DEV-LOSS: {}, BEST-DEV-TEST-LOSS:{}'.format(best_dev_loss, best_dev_test_loss))

        if counter > 100:
            break

    logger.info('===================[][][][][]====================')
    logger.info('TRAIN: epoch:{}-loss:{}'.format(epoch, train_loss))
    logger.info('DEV: epoch:{}-loss:{}'.format(epoch, dev_loss))
    logger.info('TEST: epoch:{}-loss:{}'.format(epoch, test_loss))
    logger.info('BEST-DEV-LOSS: {}, BEST-DEV-TEST-LOSS:{}'.format(best_dev_loss, best_dev_test_loss))
    torch.save(best_dev_model, model_save_dir+'/model-{}-{}-{}-{}.pt'.format(best_dev_test_loss, args.lr, args.hidden_dim, args.gamma))



def val(model, dataset):

    dataLoader = DataLoader(dataset, batch_size=args.batch_sz, shuffle=True)

    criterion = torch.nn.MSELoss()

    val_loss = 0

    model.eval()
    with torch.no_grad():
        for batch in dataLoader:
            x = torch.stack(batch['input']) # 5 x bz
            y = batch['label'].float() # bz

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            output = model(x)
            loss = criterion(output, y)
            val_loss += loss.item()

    val_loss = val_loss / len(val_dataset)

    del dataLoader

    return val_loss




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed', type=str, default='p-200', help="embed name")
    parser.add_argument('--lr', type=float, default='0.001', help="learning rate of ADAM")
    parser.add_argument('--seed', type=int, default=123, help="random seed of training")
    parser.add_argument('--batch_sz', type=int, default=512, help="batch size")
    parser.add_argument('--n_epoch', type=int, default=5, help="num epoch")
    parser.add_argument('--hidden_dim', type=int, default=10, help="num hidden dim")
    parser.add_argument('--iter_print', type=int, default=100, help="infor for evry number of iters")
    parser.add_argument('--infre', type=int, default=1, help="if use infrequent mode")
    parser.add_argument('--model', type=str, default='MLP', help="model")
    parser.add_argument('--gamma', type=float, default=1.0, help="gamma for LR schedule")

    args = parser.parse_args()

    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m-%d-%H-%M-%S")

    assert args.infre in [0, 1]
    assert args.model in ['MLP']

    run = 'MLP3Diff'
    # model_dir = 'model_infre' if args.infre else 'model'
    model_dir = 'model_infre_low'

    model_save_dir = '../{}/{}/{}/{}-{}'.format(model_dir, run, args.embed, datetime_str, args.model)

    # create dir
    create_dir('../{}/{}/'.format(model_dir, run))
    create_dir('../{}/{}/{}'.format(model_dir, run, args.embed))
    create_dir(model_save_dir)


    logging.basicConfig(level=logging.INFO,
                        filename='{}/log.txt'.format(model_save_dir),
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('{}'.format(model_save_dir))

    # temp = 'infre.' if args.infre else ''
    temp = 'lownolog.'
    val_dataset = pickle.load(open('../data/{}.diff.{}pkl'.format('dev', temp),'rb'))
    val_dataset = ProbingListMaxDataset(val_dataset)
    test_dataset = pickle.load(open('../data/{}.diff.{}pkl'.format('test', temp),'rb'))
    test_dataset = ProbingListMaxDataset(test_dataset)
    train_dataset = pickle.load(open('../data/{}.diff.{}pkl'.format('train', temp),'rb'))
    train_dataset = ProbingListMaxDataset(train_dataset)

    for lr in [0.1, 0.5, 2, 8]:
        for step_gamma in [1.0]:
            for hidden_dim in [200]:
                args.gamma = step_gamma
                args.lr = lr
                args.hidden_dim = hidden_dim
                train(args, logger, model_save_dir, val_dataset, test_dataset, train_dataset)