"""
Train Data2Vec for text. Use the models under `models` module, e.g, RoBERTa
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import omegaconf
from omegaconf import DictConfig
from tqdm import tqdm

from text.encoder import Encoder, AutoTokenizer
from text.dataset import WikiText
from utils import AverageMeter
from data2vec import Data2Vec


class TextTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_epochs = self.cfg.train.num_epochs
        self.device = self.cfg.device
        # Model, Optim, Criterion
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder_checkpoint)
        self.encoder = Encoder(cfg=cfg)
        self.model = Data2Vec(encoder=self.encoder, cfg=cfg, mask_idx=self.tokenizer.mask_token_id)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.optimizer.lr)
        self.criterion = nn.SmoothL1Loss(reduction='none', beta=cfg.criterion.loss_beta)
        self.criterion.to(self.device)
        # Datasets & Data Loaders
        self.train_dataset = WikiText(cfg, 'train', self.tokenizer)
        self.valid_dataset = WikiText(cfg, 'test', self.tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.train.batch_size,
                                       collate_fn=self.train_dataset.collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=cfg.train.val_batch_size,
                                       collate_fn=self.valid_dataset.collate_fn)
        # Tensorboard
        self.tensorboard = SummaryWriter(log_dir=self.cfg.model.log_dir)

        # Trackers
        self.loss_tracker = AverageMeter('loss')

    def train_step(self, batch):
        src = batch['input_ids'].to(self.device)
        trg = batch['labels'].to(self.device)
        x, y = self.model(src, trg)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def test_step(self, batch):
        src = batch['input_ids'].to(self.device)
        trg = batch['labels'].to(self.device)
        x, y = self.model(src, trg)
        loss = self.criterion(x, y)

        return loss.item()

    def train_epoch(self, epoch_num):
        self.model.train()
        self.loss_tracker.reset()
        with tqdm(self.train_loader, unit="batch", desc=f'Epoch: {epoch_num}/{self.num_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for batch in iterator:
                loss = self.train_step(batch)
                self.model.ema_step()
                self.loss_tracker.update(loss)
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        self.loss_tracker.reset()
        with tqdm(self.valid_loader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.no_grad():
                for batch in iterator:
                    loss = self.test_step(batch)
                    self.loss_tracker.update(loss)
                    avg_loss = self.loss_tracker.avg
                    iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            print()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            # tensorboard
            self.tensorboard.add_scalar('train_loss', train_loss, epoch)
            self.tensorboard.add_scalar('val_loss', val_loss, epoch)

            should_save_weights = lambda x: not bool(x % self.cfg.train.save_interval)
            if should_save_weights(epoch):
                save_path = os.path.join(self.cfg.train.weights_dir, f'{epoch}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved Model at {save_path}')
