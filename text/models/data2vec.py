import torch
import torch.nn as nn
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.models.ema import EMA
from fairseq.dataclass.configs import EMAConfig


class Data2VecTextEncoder(nn.Module):
    """
    Encoder module for Data2Vec model for text
    """

    def __init__(self, cfg, encoder: FairseqEncoder, mask_idx):
        super(Data2VecTextEncoder, self).__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.teacher = self._build_teacher()
        self.mask_idx = mask_idx
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_scale = cfg.loss_scale
        self.regression_head = self._build_regression_head()
        self.num_updates = 0

    def _build_teacher(self):
        ema_config = EMAConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_transformer_layers_only:
            for k, _ in self.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_tokens.{k}")
            for k, _ in self.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_positions.{k}")
            if self.sentence_encoder.layernorm_embedding is not None:
                for k, _ in self.sentence_encoder.layernorm_embedding.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")
            if self.sentence_encoder.layer_norm is not None:
                for k, _ in self.sentence_encoder.layer_norm.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")

        ema = EMA(self.sentence_encoder, ema_config, skip_keys=skip_keys)
        return ema

    def _build_regression_head(self):
        embed_dim = self.cfg.transformer.encoder.embed_dim
        curr_dim = embed_dim
        projs = []
        for i in range(self.cfg.head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim

        projs.append(nn.Linear(curr_dim, embed_dim))
        regression_head = nn.Sequential(*projs)
        return regression_head

    def set_num_updates(self, num_updates):
        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)
        if self.training:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.teacher._set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.sentence_encoder)