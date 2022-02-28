"""
Train Data2Vec for vision. Use the models from `models` module, e.g, BEiT
"""
import torch
import torch.nn as nn

from data2vec import Data2Vec
from models import BEiT


class Trainer:
    def __init__(self, cfg):
        self.model = ...
        self.optimizer = ...
        self.criterion = ...
        self.scheduler = ...

    def train_step(self, x):
        ...

    def valid_step(self, x):
        ...

    def train_epoch(self, dataloader, epoch_num):
        ...

    def train(self):
        ...

    def evaluate(self):
        ...
