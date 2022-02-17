import torch
import torch.nn as nn
import torch.nn.functional as F
from .ema import EMA


class Data2Vec(nn.Module):
    def __init__(self, encoder, cfg):
        super(Data2Vec, self).__init__()
        self.encoder = Data2VecEncoder(encoder, cfg)
        self.classification_head = nn.Linear(cfg.in_features, cfg.num_classes)

    def forward(self, src, trg=None, do_classification=False):
        src = self.encoder.apply_mask(src)
        encoder_output = self.encoder(src, trg, features_only=not do_classification)
        if do_classification:
            classification_output = self.classification_head(encoder_output)
            return classification_output
        else:
            return encoder_output


class Data2VecEncoder(nn.Module):
    MODALITIES = ['vision', 'text', 'audio']

    def __init__(self, encoder: nn.Module, cfg):
        super(Data2VecEncoder, self).__init__()
        self.modality = cfg.modality
        assert cfg.modality in self.MODALITIES
        self.encoder = encoder

        self.cfg = cfg
        self.teacher = EMA(self.encoder, cfg)
        self.regression_head = nn.ModuleList()  # custom layers for projection

    def forward(self, src, trg, features_only=False):
        x = self.encoder.extract_features(src)
        if features_only:
            return x

        with torch.no_grad():
            self.teacher.model.eval()

            y = self.teacher.model(trg)
            y = y[self.cfg.teacher_features]
            y = y[-self.cfg.top_k_layers:]

            if self.modality in ['vision', 'text']:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]
                y = sum(y) / len(y)
                y = y.transpose(0, 1)
                y = F.layer_norm(y.float(), y.shape[-1:])

            elif self.modality == 'audio':
                y = [tl.permute(1, 2, 0) for tl in y]
                y = [F.instance_norm(tl.float()) for tl in y]
                y = [tl.transpose(1, 2) for tl in y]
                y = sum(y) / len(y)
                y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        masked_indices = src.eq(self.mask_idx)
        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        return x, y
