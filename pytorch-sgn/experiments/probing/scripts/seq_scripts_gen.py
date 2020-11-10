from itertools import product
import os, datetime
from experiments.probing.src.prob_utils import create_dir

def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m-%d-%H-%M-%S")
    return datetime_str

def gen_seq_scripts(args):
    commands = []
    cuda_str = 'CUDA_VISIBLE_DEVICES'

    values = args.values()
    keys = list(args.keys())

    iter = product(*values)
    for i in iter:
        assert len(i) == len(keys)

        cmd = "{}={} {} {}".format(
            cuda_str, cuda, python_path, py
        )

        for j in range(len(keys)):
            param = keys[j]
            param_val = i[j]

            cmd += ' --{} {}'.format(param, param_val)

        commands.append(cmd)

    return commands

def gen_scripts_files(commands, batch, script_dir, script_name):


    create_dir(script_dir)
    files_number = int(len(commands) / batch) + 1
    datetime_str = create_datetime_str()
    for fn in range(files_number):
        file_name = os.path.join(script_dir, "{}.{}.{}.sh".format(datetime_str, fn, script_name))
        with open(file_name, 'w', encoding='utf-8') as f:
            f.writelines(header)
            f.writelines('\n')

            for cmd in commands[fn*batch: min((fn+1)*batch, len(commands))]:
                f.writelines(cmd)
                f.writelines('\n')


def gen_par_scripts():
    return


python_path = '/root/Anaconda3/envs/pytorch1.2/bin/python3'
cuda = 1
py = 'train_diff.py'
header = 'cd ../../src/'

if __name__ == '__main__':
    args_diff = {
        'lr': [0.1, 0.3, 0.6, 1],
        'n_epoch':[300],
        'hidden_dim': [50, 200],
        'gamma': [1.0],
        'run':['MLP3Diff'],
        'model':['MLP3'],
        'model_dir': ['high'],
        'embed_dir': ['embed'],
        'dataset_dir': ['highnolognocorr'],
        'embed':['p-300','p-log-200', 'p-log-300','p-log-500',
                 'gmm-rd-soft-500','gmm-rd-hard-500','gmm-log-rd-hard-300',
                 'gmm-log-rd-soft-300', 'gmm-log-rd-hard-500', 'gmm-log-rd-soft-500'],
        # 'embed': ['Token', 'LSTMnew', 'Fixed', 'p-200', 'p-500', 'p-log-500', 'gmm-rd-soft-300','gmm-rd-hard-300', 'uniform', 'normal'],
        # 'embed': ['uniform', 'normal'],
    }


    args = {
        'lr': [0.01, 0.03, 0.1, 0.3, 0.6, 1],
        'n_epoch':[1500],
        'hidden_dim': [50, 200],
        'gamma': [1.0],
        'run':['MLP1Decoding'],
        'model':['MLP1'],
        'model_dir': ['high'],
        'embed_dir': ['embed'],
        'dataset_dir': ['regre'],
        'embed':['p-300','p-log-200', 'p-log-300','p-log-500',
                 'gmm-rd-soft-500','gmm-rd-hard-500','gmm-log-rd-hard-300',
                 'gmm-log-rd-soft-300', 'gmm-log-rd-hard-500', 'gmm-log-rd-soft-500'],
        # 'embed': ['Token', 'LSTMnew', 'Fixed', 'p-200', 'p-500', 'p-log-500', 'gmm-rd-soft-300','gmm-rd-hard-300', 'uniform', 'normal'],
        # 'embed': ['uniform', 'normal'],
    }

    batch = 27

    commands = gen_seq_scripts(args_diff)
    gen_scripts_files(commands, batch, '../scripts/high_diff_new/', 'MLP3high')

