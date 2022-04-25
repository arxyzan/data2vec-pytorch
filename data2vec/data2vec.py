import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class Data2Vec(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module): The encoder module like BEiT, ViT, etc.
         cfg (omegaconf.DictConfig): The config containing model properties
    """
    MODALITIES = ['vision', 'text', 'audio']

    def __init__(self, encoder, cfg, **kwargs):
        super(Data2Vec, self).__init__()
        self.modality = cfg.modality
        self.embed_dim = cfg.model.embed_dim
        self.encoder = encoder
        self.__dict__.update(kwargs)

        self.cfg = cfg
        self.ema = EMA(self.encoder, cfg)  # EMA acts as the teacher
        self.regression_head = self._build_regression_head()

        self.cfg = cfg
        self.ema_decay = self.cfg.model.ema_decay
        self.ema_end_decay = self.cfg.model.ema_end_decay
        self.ema_anneal_end_step = self.cfg.model.ema_anneal_end_step

    def _build_regression_head(self):
        """
        Construct the regression head consisting of linear and activation layers.

        Each modality might have its own regression block.

        Returns:
            A nn.Module layer or block of layers
        """
        if self.modality == 'text':
            return nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2),
                                 nn.GELU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim))

        if self.modality in ['audio', 'vision']:
            return nn.Linear(self.embed_dim, self.embed_dim)

    def ema_step(self):
        """
        One EMA step for the offline model until the ending decay value is reached
        """
        if self.ema_decay != self.ema_end_decay:
            if self.ema.num_updates >= self.ema_anneal_end_step:
                decay = self.ema_end_decay
            else:
                decay = self.ema.get_annealed_rate(
                    self.ema_decay,
                    self.ema_end_decay,
                    self.ema.num_updates,
                    self.ema_anneal_end_step,
                )
            self.ema.decay = decay
        if self.ema.decay < 1:
            self.ema.step(self.encoder)

    def forward(self, src, trg=None, mask=None, **kwargs):
        """
        Data2Vec forward method.

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)
            mask: bool masked indices, Note: if a modality requires the inputs to be masked before forward this param
            has no effect. (see the Encoder for each modality to see if it uses mask or not)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        # model forward in online mode (student)
        x = self.encoder(src, mask, **kwargs)['encoder_out']  # fetch the last layer outputs
        if trg is None:
            return x

        # model forward in offline mode (teacher)
        with torch.no_grad():
            self.ema.model.eval()
            y = self.ema.model(trg, ~mask, **kwargs)['encoder_states']  # fetch the last transformer layers outputs
            y = y[-self.cfg.model.average_top_k_layers:]  # take the last k transformer layers

            # Follow the same layer normalization procedure for text and vision
            if self.modality in ['vision', 'text']:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                if self.cfg.model.normalize_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

            # Use instance normalization for audio
            elif self.modality == 'audio':
                y = [F.instance_norm(tl.float()) for tl in y]
                y = sum(y) / len(y)
                if self.cfg.model.normalize_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        x = x[mask]
        y = y[mask]

        x = self.regression_head(x)

        return x, y
