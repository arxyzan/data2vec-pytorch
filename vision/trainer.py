import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dall_e

from vision.encoder import Encoder
from vision.dataset import BEiTPretrainingDataset
from data2vec import Data2Vec
from utils import AverageMeter


class VisionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.num_epochs = cfg.train.num_epochs
        # Model, Criterion, Optimizer
        self.d_vae = dall_e.load_model(cfg.model.vae_checkpoint)
        self.encoder = Encoder(cfg=cfg)
        self.model = Data2Vec(encoder=self.encoder, cfg=cfg)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.optimizer.lr)
        self.criterion = nn.SmoothL1Loss(reduction='none', beta=cfg.criterion.loss_beta)
        self.criterion.to(self.device)
        # Datasets & Data Loaders
        self.dataset = BEiTPretrainingDataset(cfg, vae=self.d_vae)
        self.train_loader = DataLoader(self.dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)
        self.valid_loader = DataLoader(self.dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)

        # Tensorboard
        self.tensorboard = SummaryWriter(log_dir=self.cfg.model.log_dir)

        # Trackers
        self.loss_tracker = AverageMeter('loss')

    def train_step(self, batch):
        src, trg, mask = batch
        src = src.to(self.device)
        trg = trg.to(self.device)
        mask = mask.to(self.device)

        x, y = self.model(src, trg, mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def test_step(self, batch):
        src, trg, mask = batch
        src = src.to(self.device)
        trg = trg.to(self.device)
        mask = mask.to(self.device)

        x, y = self.model(src, trg, mask)
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

            should_save_weights = lambda x: not bool(x % self.cfg.train.save_ckpt_freq)
            if should_save_weights(epoch):
                save_path = os.path.join(self.cfg.train.weights_dir, f'{epoch}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved Model at {save_path}')
