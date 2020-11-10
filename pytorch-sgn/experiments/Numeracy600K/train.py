import torch
from torch.utils.data import DataLoader
from data import Numeracy600kDataset
from models import BiGRU, CNN
from evaluate import evaluate, f1_metrics
import datetime
import random
import numpy as np
import pickle

def train(config):

    # set seed to produce deterministic output
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)


    dataset = Numeracy600kDataset(
        'exps/{}/preproc.train.json'.format(config.exp_dir), 'exps/{}'.format(config.exp_dir))
    dataLoader = DataLoader(dataset, batch_size=config.batch_sz, shuffle=True)


    pretrained_embed = \
        pickle.load(open('exps/{}/embed/{}.emb'.format(config.exp_dir, config.pretrained_embed_name), 'rb')) \
            if config.pretrained_embed_name != 'rand' else None

    # TODO add embedding initialize
    vocab_size = len(dataset.idx2word)
    embedding_dim = config.embedding_dim

    # TODO add parse_arg and config
    n_epochs = config.n_epochs

    if config.model == 'BiGRU':
        model = BiGRU(vocab_size, embedding_dim, config, pretrained_embed)

    elif config.model == 'CNN':
        model = CNN(vocab_size, embedding_dim, config, pretrained_embed)

    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_dev_micro_f1 = 0
    best_model_macro_f1 = 0
    best_model_acc = 0
    best_model_loss = None
    best_dev_model = None
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()

        train_loss = 0
        train_acc = 0

        train_TP = [0, 0, 0, 0, 0, 0, 0, 0]
        train_FP = [0, 0, 0, 0, 0, 0, 0, 0]
        train_FN = [0, 0, 0, 0, 0, 0, 0, 0]

        for batch in dataLoader:
            optimizer.zero_grad()
            input_seqs = batch['input']
            input_lengths = batch['length']
            label = batch['label']

            if torch.cuda.is_available():
                input_seqs = input_seqs.cuda()
                input_lengths = input_lengths.cuda()
                label = label.cuda()

            output = model(input_seqs, input_lengths)
            loss = criterion(output, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == label).sum().item()

            # Update the 3 list
            for i in range(len(output.argmax(1))):  
                if output.argmax(1)[i] == label[i]:
                    train_TP[output.argmax(1)[i]] += 1
                else:
                    train_FP[output.argmax(1)[i]] += 1
                    train_FN[label[i]] += 1

        train_micro_f1, train_macro_f1 = f1_metrics(train_TP, train_FP, train_FN)

        train_loss = train_loss/len(dataset)
        train_acc = train_acc/len(dataset) 
        info_str = "Epoch {}: train_loss {} | train_acc {} | train_micro_f1 {} | train_macro_f1 {}".format(epoch, train_loss, train_acc, train_micro_f1, train_macro_f1)
        print(info_str)
        config.logger.info(info_str)

        # Evaluate on Validation, currently based on acc, may be extended to Macro and Micro F1
        dev_loss, dev_acc, dev_micro_f1, dev_macro_f1 = evaluate(config, model, mode='dev')
        if dev_micro_f1 > best_dev_micro_f1:
            best_dev_micro_f1 = dev_micro_f1
            best_model_acc = dev_acc
            best_model_macro_f1 = dev_macro_f1
            best_model_loss = dev_loss
            best_dev_model = model.state_dict().copy()
            best_epoch = epoch

    now = datetime.datetime.now()
    timestr = now.strftime("%Y%M%d-%H:%M:%S")
    save_str = "Saving Model {} in Epoch {}, Best Dev micro-f1: {}, Model macro-f1: {}, acc: {}, loss: {}".format(
        config.model, best_epoch, best_dev_micro_f1, best_model_macro_f1, best_model_acc, best_model_loss)

    config.logger.info(save_str)
    print(save_str)

    test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(config, model.load_state_dict(best_dev_model), mode='test')

    save_str = "Testing Best Dev Model {}, Test micro-f1: {}, Model macro-f1: {}, acc: {}, loss: {}".format(
        config.model, test_micro_f1, test_macro_f1, test_acc, test_loss)

    config.logger.info(save_str)
    print(save_str)

    model_save_path = 'exps/{}/{}/{}-{}-{}-e{}-f1-{}-{}-lr-{}.pt'.format(config.exp_dir, config.save_dir, config.pretrained_embed_name, config.model, timestr, best_epoch, best_dev_micro_f1, test_micro_f1, config.lr)
    torch.save(best_dev_model, model_save_path)
