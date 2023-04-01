import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
import numpy as np

from easydict import EasyDict
from basicts.utils import load_adj

from basicts.archs import STEP
from basicts.runners import STEPRunner
from basicts.losses import step_loss
from basicts.data import TimeSeriesForecastingDataset


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STEP(Hydro) configuration"
CFG.RUNNER = STEPRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "Hydro"
CFG.DATASET_TYPE = "Electricity"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.DATASET_ARGS = {
    "seq_len": 288 * 7 ,
    "start_seq_len": 288 * 2,
    }
CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STEP"
CFG.MODEL.ARCH = STEP
# adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "doubletransition")
adj_mx = [np.zeros((80,80)),np.zeros((80,80))]
CFG.MODEL.PARAM = {
    "dataset_name": CFG.DATASET_NAME,
    "pre_trained_tsformer_path": "/home/hallie/Desktop/paper_code/BasicTS_np2/checkpoints/TSFormer_100/Hydro_2016_12_b499dc17c3d829fda6ceec8f57db1445/TSFormer_best_val_MAE.pt",
    "tsformer_args": {
                    "patch_size":12,
                    "in_channel":1,
                    "embed_dim":96,
                    "num_heads":4,
                    "mlp_ratio":4,
                    "dropout":0.1,
                    "num_token":288 * 7 / 12,
                    "mask_ratio":0.75,
                    "encoder_depth":4,
                    "decoder_depth":1,
                    "mode":"forecasting"
    },
    "backend_args": {
                    "num_nodes" : 80,
                    "supports"  :[torch.tensor(i) for i in adj_mx],         # the supports are not used
                    "dropout"   : 0.3,
                    "gcn_bool"  : True,
                    "addaptadj" : True,
                    "aptinit"   : None,
                    "in_dim"    : 2,
                    "out_dim"   : 12,
                    "residual_channels" : 32,
                    "dilation_channels" : 32,
                    "skip_channels"     : 256,
                    "end_channels"      : 512,
                    "kernel_size"       : 2,
                    "blocks"            : 4,
                    "layers"            : 2
    },
    "dgl_args": {
                "dataset_name": CFG.DATASET_NAME,
                "k": 10,
                "input_seq_len": CFG.DATASET_INPUT_LEN,
                "output_seq_len": CFG.DATASET_OUTPUT_LEN
    }
}
CFG.MODEL.FROWARD_FEATURES = [0, 1, 2]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = True

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = step_loss
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM= {
    "lr":0.002,
    "weight_decay":1.0e-5,
    "eps":1.0e-8,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM= {
    "milestones":[1, 18, 36, 54, 72],
    "gamma":0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 1
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 8
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# evluation
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 8
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = True