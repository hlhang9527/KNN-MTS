from ast import parse
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from argparse import ArgumentParser

from easytorch import launch_runner
from basicts.runners import BaseTimeSeriesForecastingRunner as Runner


def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    # parser.add_argument('-c', '--cfg', default="checkpoints/STEP_100/6aa7ebdde8103e1a276a6c7f2b0fe7d7/STEP_PEMS04-3d-finetune.py", help='training config')
    # parser.add_argument('--ckpt', default="checkpoints/STEP_100/PEMS04_12_12_6aa7ebdde8103e1a276a6c7f2b0fe7d7/STEP_best_val_MAE.pt", help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument('-c', '--cfg', default="checkpoints/STEP_100/6f6d705d0f5bea017b9e9ee01e307792/STEP_PEMS04-3d-finetune.py", help='training config')
    parser.add_argument('--ckpt', default="checkpoints/STEP_100/PEMS04_12_12_6f6d705d0f5bea017b9e9ee01e307792/STEP_best_val_MAE.pt", help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument("--gpus", default="0", help="visible gpus")
    parser.add_argument("--task", default="plot", type=str)
    parser.add_argument("--dstore_dir", default="./data_store/D2STGNN", type=str)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--encoding_hidden_dim", default=512, type=int)
    parser.add_argument("--eval_len", default=20, type=int)
    return parser.parse_args()


def main(cfg: dict, runner: Runner, ckpt: str = None, task="create_data_store", dstore_dir = "/data/research/time_series/STEP/data_store",
        topk=10, encoding_hidden_dim=512, used_hidden="hiddens", eval_len=20):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')

    runner.load_model(ckpt_path=ckpt)
    if task == "test":
        runner.test_process(cfg)
    elif task == "create_data_store":
        runner.create_data_store(cfg=cfg, output_dir=dstore_dir, encoding_hidden_dim=encoding_hidden_dim)
    elif task == "knn_test":
        runner.test_knn_process(cfg=cfg, dstore_dir=dstore_dir, k=topk, used_hidden=used_hidden, knn_weight=0.75)


if __name__ == '__main__':
    args = parse_args()
    launch_runner(args.cfg, main, (args.ckpt, args.task, args.dstore_dir, args.topk, args.encoding_hidden_dim, args.eval_len), gpus=args.gpus)