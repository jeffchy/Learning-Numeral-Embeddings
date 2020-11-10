import torch
from torch.utils.data import DataLoader
from data import Numeracy600kDataset

def f1_metrics(eval_TP, eval_FP, eval_FN):
    eval_macro_f1 = 0

    #Micro F1
    TP = sum(eval_TP)
    FP = sum(eval_FP)
    FN = sum(eval_FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    if precision + recall == 0:
        eval_micro_f1 = 0
    else:
        eval_micro_f1 = 2 * (precision * recall) / (precision + recall)

    #Macro F1
    good_sample = 0
    for i in range(len(eval_TP)):
        if (eval_TP[i] + eval_FP[i]) == 0:
            continue
        else:
            precision = eval_TP[i] / (eval_TP[i] + eval_FP[i])
        if (eval_TP[i] + eval_FN[i]) == 0:
            continue
        else:
            recall = eval_TP[i] / (eval_TP[i] + eval_FN[i])
        if (precision + recall) == 0:
            continue
        else:
            eval_macro_f1 += 2 * (precision * recall) / (precision + recall)
            good_sample += 1

    if good_sample == 0:
        eval_macro_f1 = 0
    else:
        eval_macro_f1 = eval_macro_f1/good_sample

    return eval_micro_f1, eval_macro_f1



def evaluate(config, model, mode='dev'):

    dataset = Numeracy600kDataset(
        'exps/{}/preproc.{}.json'.format(config.exp_dir, mode), 'exps/{}'.format(config.exp_dir))
    dataLoader = DataLoader(dataset, batch_size=config.batch_sz)

    criterion = torch.nn.CrossEntropyLoss()

    eval_loss = 0
    eval_acc = 0

    eval_TP = [0, 0, 0, 0, 0, 0, 0, 0]
    eval_FP = [0, 0, 0, 0, 0, 0, 0, 0]
    eval_FN = [0, 0, 0, 0, 0, 0, 0, 0]

    model.eval()
    with torch.no_grad():
        for batch in dataLoader:
            input_seqs = batch['input']
            input_lengths = batch['length']
            label = batch['label']

            if torch.cuda.is_available():
                input_seqs = input_seqs.cuda()
                input_lengths = input_lengths.cuda()
                label = label.cuda()

            output = model(input_seqs, input_lengths)
            loss = criterion(output, label)
            eval_loss += loss.item()
            eval_acc += (output.argmax(1) == label).sum().item()

            # Update the 3 list
            for i in range(len(output.argmax(1))):  
                if output.argmax(1)[i] == label[i]:
                    eval_TP[output.argmax(1)[i]] += 1
                else:
                    eval_FP[output.argmax(1)[i]] += 1
                    eval_FN[label[i]] += 1

    eval_micro_f1, eval_macro_f1 = f1_metrics(eval_TP, eval_FP, eval_FN)
    eval_loss = eval_loss/len(dataset)
    eval_acc = eval_acc/len(dataset)
    info_str = "Epoch {}: dev_loss {} | dev_acc {} | dev_micro_f1 {} | dev_macro_f1 {}".format(mode, eval_loss, eval_acc, eval_micro_f1, eval_macro_f1)
    print(info_str)
    config.logger.info(info_str)

    return eval_loss, eval_acc, eval_micro_f1, eval_macro_f1

