"""
Train Data2Vec for text. Use the models under `models` module, e.g, RoBERTa
"""
import torch
import torch.nn as nn

from data2vec import Data2Vec
from models import Roberta
from dataset import WikiText


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
