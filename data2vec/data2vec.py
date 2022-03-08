import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class Data2Vec(nn.Module):
    """
    Data2Vec main module.

    Args:
         encoder (nn.Module)
         cfg (omegaconf.DictConfig)
    """

    def __init__(self, encoder, cfg, **kwargs):
        super(Data2Vec, self).__init__()
        self.__dict__.update(kwargs)
        self.encoder = Data2VecEncoder(encoder, cfg, **kwargs)
        # self.classification_head = nn.Linear(cfg.in_features, cfg.num_classes)
        self.classification_head = None

    def forward(self, src, trg=None):
        """
        Encode inputs and pass to encoder. Apply classification head if trg is not given

        Args:
            src: source tokens
            trg: target tokens. if provided it means the model is in training mode

        Returns:
            Either encoder outputs or classification outputs
        """
        encoder_output = self.encoder(src, trg)
        if trg is None:
            classification_output = self.classification_head(encoder_output)
            return classification_output
        else:
            return encoder_output


class Data2VecEncoder(nn.Module):
    """
    Encoder block of Data2Vec.

    This module consists of two parts; the encoder (student) and the EMA of encoder (teacher) which is only used in
    training. The encoder has to predict the representations of the masked inputs which are to be compared to the
    outputs from the EMA who predicts the representations of the unmasked inputs.

    Args:
        encoder (nn.Module): The encoder module that has to implement two methods: `extract_features` & `apply_mask`
        cfg (omegaconf.DictConfig)
    """
    MODALITIES = ['vision', 'text', 'audio']

    def __init__(self, encoder: nn.Module, cfg, **kwargs):
        super(Data2VecEncoder, self).__init__()
        self.modality = cfg.modality
        self.embed_dim = cfg.model.embed_dim
        assert cfg.modality in self.MODALITIES
        self.encoder = encoder
        self.__dict__.update(kwargs)

        self.cfg = cfg
        self.teacher = EMA(self.encoder, cfg)
        self.regression_head = self._build_regression_head()  # custom layers for projection

    def _build_regression_head(self):
        if self.modality == 'text':
            embed_dim = self.embed_dim
            curr_dim = embed_dim
            projs = []
            for i in range(self.cfg.model.head_layers - 1):
                next_dim = embed_dim * 2 if i == 0 else curr_dim
                projs.append(nn.Linear(curr_dim, next_dim))
                projs.append(nn.GELU())
                curr_dim = next_dim

            projs.append(nn.Linear(curr_dim, embed_dim))
            return nn.Sequential(*projs)

        if self.modality == 'audio':
            return nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, src, trg=None):
        """
        Forward method has two modes:
            `training`: Encoder predicts representations using masked inputs (src) and the teacher (Encoder EMA)
            predicts the representations using unmasked inputs (trg)

            `eval`: The encoder extracts features from the unmasked inputs. (trg is left `None` and `features_only` is
            `True`)

        Args:
            src: src tokens (masked inputs for training)
            trg: trg tokens (unmasked inputs for training but left as `None` otherwise)

        Returns:
            Either encoder outputs or a tuple of encoder + EMA outputs

        """
        x = self.encoder.extract_features(src)['encoder_out']
        if trg is None:
            return x

        with torch.no_grad():
            self.teacher.model.eval()

            y = self.teacher.model.extract_features(trg)['encoder_states']
            y = y[-self.cfg.model.average_top_k_layers:]

            if self.modality in ['vision', 'text']:  # Follow the same layer normalization procedure for text and vision
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                if self.cfg.norm_targets:
                    y = F.layer_norm(y.float(), y.shape[-1:])

            elif self.modality == 'audio':  # Use instance normalization for audio
                y = [tl.permute(1, 2, 0) for tl in y]
                y = [F.instance_norm(tl.float()) for tl in y]
                y = [tl.transpose(1, 2) for tl in y]
                y = sum(y) / len(y)
                if self.cfg.normalize_targets:
                    y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        masked_indices = src.eq(self.mask_idx)
        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        return x, y
