import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from omegaconf import DictConfig
from tqdm import tqdm

from audio.encoder import Encoder
from audio.dataset import TIMIT, DataCollatorForWav2Vec2Pretraining
from data2vec import Data2Vec
from utils import AverageMeter, maybe_save_checkpoint


class AudioTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.num_epochs = self.cfg.train.num_epochs
        self.device = self.cfg.device
        self.ckpt_dir = cfg.train.checkpoints_dir
        self.save_ckpt_freq = cfg.train.save_ckpt_freq
        # Model, Optim, Criterion
        self.encoder = Encoder(cfg=cfg)
        self.model = Data2Vec(encoder=self.encoder, cfg=cfg)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), cfg.optimizer.lr)
        self.criterion = nn.MSELoss(reduction='none')
        self.criterion.to(self.device)
        # Datasets & Data Loaders
        self.train_dataset = TIMIT(cfg, 'train')
        self.test_dataset = TIMIT(cfg, 'test')
        self.feature_extractor = self.train_dataset.feature_extractor
        self.data_collator = DataCollatorForWav2Vec2Pretraining(self.encoder.encoder, self.feature_extractor,
                                                                padding='longest')
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg.train.batch_size,
                                       collate_fn=self.data_collator)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg.train.val_batch_size,
                                      collate_fn=self.data_collator)
        # Tensorboard
        self.tensorboard = SummaryWriter(log_dir=self.cfg.train.log_dir)

        # Trackers
        self.loss_tracker = AverageMeter('loss')

    def train_step(self, batch):
        """
        Train one batch of data and return loss.

        Args:
            batch: A batch of data, inputs, labels and mask with shape [batch_size, seq_len]

        Returns:
            Loss value
        """
        src, mask = batch
        src, mask = src.to(self.device), mask.to(self.device)
        # src is not masked so can be used as trg. (src will be masked in the encoder forward)
        x, y = self.model(src, src, mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).div(x.size(0))
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def test_step(self, batch):
        """
        Test a model on one batch of data and return loss.

        Args:
            batch: A batch of data, inputs, labels and mask with shape [batch_size, seq_len]

        Returns:
            Loss value
        """
        src, mask = batch
        src, mask = src.to(self.device), mask.to(self.device)
        # src is not masked so can be used as trg. (src will be masked in the encoder forward)
        x, y = self.model(src, src, mask=mask)
        loss = self.criterion(x.float(), y.float()).sum(dim=-1).div(x.size(0))

        return loss.item()

    def train_epoch(self, epoch_num):
        """
        Train the model for one epoch and verbose using the progress bar.

        Args:
            epoch_num: number of the current epoch

        Returns:
            The average loss through the whole epoch
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
        Evaluate the model on the test set

        Returns:
            The average loss through the whole test dataset
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

            maybe_save_checkpoint(self.model, self.optimizer, self.ckpt_dir, epoch, self.save_ckpt_freq)
