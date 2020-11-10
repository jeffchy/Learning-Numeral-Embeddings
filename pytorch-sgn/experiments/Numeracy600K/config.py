from models import BiGRU
import logging
import os

class Config(object):
    def __init__(self, args):
        self.seed = 1
        self.dropout_rate = 0.3
        self.rnn_hidden_size = 64
        self.n_epochs = 40
        self.lr = 0.001
        self.batch_sz = 3
        self.embedding_dim = 300
        self.exp_dir = 'title_numeracy_embed_10W_E10'
        self.save_dir = 'run0'


        # Embed Part
        self.pretrained_embed_name = 'token' # rand

        self.model = 'BiGRU'
        # self.model = 'CNN'

        self.run_info = "Shuffle Train Dataloader"

        if args != None:
            self.initialize_args(args)

        self.initailize_dir()
        self.create_logger()
        self.log_config()

    def initialize_args(self, args):
        self.seed = args.seed
        self.dropout_rate = args.drop_out
        self.lr = args.lr
        self.batch_sz = args.batch_sz
        self.save_dir = args.save_dir
        self.model = args.model
        self.pretrained_embed_name = args.pretrained_embed_name


    def create_logger(self):
        logging.basicConfig(level=logging.INFO,
                            filename='exps/{}/{}/log-{}-{}.txt'.format(self.exp_dir, self.save_dir, self.pretrained_embed_name, self.model),
                            datefmt='%Y/%m/%d %H:%M:%S',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('{} - {}'.format(self.exp_dir, self.save_dir))
        self.logger = logger

    def initailize_dir(self):
        if not os.path.exists('exps/{}/'.format(self.exp_dir)):
            os.mkdir('exps/{}/'.format(self.exp_dir))
            os.mkdir('exps/{}/{}'.format(self.exp_dir, self.save_dir))
        else:
            if not os.path.exists('exps/{}/{}'.format(self.exp_dir, self.save_dir)):
                os.mkdir('exps/{}/{}'.format(self.exp_dir, self.save_dir))
            if not os.path.exists('exps/{}/{}'.format(self.exp_dir, 'embed')):
                os.mkdir('exps/{}/{}'.format(self.exp_dir, 'embed'))


    def log_config(self):
        self.logger.info("=== Config Info ===")
        self.logger.info("seed: {}".format(self.seed))
        self.logger.info("dropout_rate: {}".format(self.dropout_rate))
        self.logger.info("rnn_hidden_size: {}".format(self.rnn_hidden_size))
        self.logger.info("n_epochs: {}".format(self.n_epochs))
        self.logger.info("lr: {}".format(self.lr))
        self.logger.info("batch_sz: {}".format(self.batch_sz))
        self.logger.info("embedding_dim: {}".format(self.embedding_dim))
        self.logger.info("exp_dir: {}".format(self.exp_dir))
        self.logger.info("save_dir: {}".format(self.save_dir))
        self.logger.info("pretrained_embed_name: {}".format(self.pretrained_embed_name))
        self.logger.info("model: {}".format(self.model))
        self.logger.info("run_info: {}".format(self.run_info))
        self.logger.info("=== Config Info End ===")
