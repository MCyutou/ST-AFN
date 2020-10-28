import collections
import typing
import torch
import numpy as np


class TrainConfig(typing.NamedTuple):
    T: int
    train_size: int
    batch_size: int
    loss_func: typing.Callable
    interval: int
    timestep: int
    isMean: bool


class TrainData(typing.NamedTuple):
    feats: np.ndarray
    targs: np.ndarray
    speeds: np.ndarray


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TCHA_Net = collections.namedtuple("TCHA", ["encoder", "decoder", "enc_opt", "dec_opt"])
