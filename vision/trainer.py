import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vision.encoder import Encoder
from vision.dataset import MIMPretrainingDataset
from data2vec import Data2Vec
from utils import AverageMeter, maybe_save_checkpoint


class VisionTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.num_epochs = cfg.train.num_epochs
        self.ckpt_dir = cfg.train.checkpoints_dir
        self.save_ckpt_freq = cfg.train.save_ckpt_freq
        # Model, Criterion, Optimizer
        self.encoder = Encoder(cfg=cfg)
        self.model = Data2Vec(encoder=self.encoder, cfg=cfg)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.optimizer.lr)
        self.criterion = nn.SmoothL1Loss(reduction='none', beta=cfg.criterion.loss_beta)
        self.criterion.to(self.device)
        # Datasets & Data Loaders
        self.train_dataset = MIMPretrainingDataset(cfg, split='train')
        self.test_dataset = MIMPretrainingDataset(cfg, split='test')
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)

        # Tensorboard
        self.tensorboard = SummaryWriter(log_dir=self.cfg.train.log_dir)

        # Trackers
        self.loss_tracker = AverageMeter('loss')

    def train_step(self, batch):
        """
        Train one batch of data
        Args:
            batch: A batch of data, src, trg of shape [N, C, H, W] and mask of shape [N, num_total_patches]

        Returns:
            Loss value
        """
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
        """
        Evaluate one batch of data
        Args:
            batch: A batch of data, src, trg of shape [N, C, H, W] and mask of shape [N, num_total_patches]

        Returns:
            Loss value
        """
        src, trg, mask = batch
        src = src.to(self.device)
        trg = trg.to(self.device)
        mask = mask.to(self.device)

        x, y = self.model(src, trg, mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).sum().div(x.size(0))

        return loss.item()

    def train_epoch(self, epoch_num):
        """
        Train the model for one epoch
        Args:
            epoch_num: number of the current epoch

        Returns:
            Average loss through the whole epoch
        """
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
        """
        Evaluate the model on the test data
        Returns:
            Average loss on the test set
        """
        self.model.eval()
        self.loss_tracker.reset()
        with tqdm(self.test_loader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.no_grad():
                for batch in iterator:
                    loss = self.test_step(batch)
                    self.loss_tracker.update(loss)
                    avg_loss = self.loss_tracker.avg
                    iterator.set_postfix(loss=avg_loss)

        return avg_loss

    def train(self):
        """
        Train and evaluate the model on the datasets and save checkpoints and write summaries to TensorBoard.

        """
        for epoch in range(1, self.num_epochs + 1):
            print()
            train_loss = self.train_epoch(epoch)
            val_loss = self.evaluate()

            # tensorboard
            self.tensorboard.add_scalar('train_loss', train_loss, epoch)
            self.tensorboard.add_scalar('val_loss', val_loss, epoch)

            # save checkpoint
            maybe_save_checkpoint(self.model, self.optimizer, self.ckpt_dir, epoch, self.save_ckpt_freq)
