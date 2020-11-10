from config import Config
from train import train
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="[CNN, BiGRU]", default="CNN")
    parser.add_argument("--pretrained_embed_name", type=str, help="[rand, token, LSTM, Fixed, p-{200}-{log}, g-{300}-{soft}-{log}]", required=True)
    parser.add_argument("--lr", type=float, help="lr for Adam", default=0.001)
    parser.add_argument("--batch_sz", type=int, help="batch size for Training", default=4096)
    parser.add_argument("--save_dir", type=str, help="dir name for saving", default="run0")
    parser.add_argument("--seed", type=int, help="random seed", default=1)
    parser.add_argument("--drop_out", type=float, help="drop out for final features", default=0.3)
    parser.add_argument("--n_epochs", type=int, help="epochs", default=40)

    args = parser.parse_args()
    config = Config(args)
    train(config)