"""
Train Data2Vec for text built on RoBERTa
"""
import torch
import torch.nn as nn

from models import Data2Vec, Roberta


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
