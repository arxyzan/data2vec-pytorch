import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from fairseq.models.fairseq_encoder import FairseqEncoder
from ema import EMA


class Data2VecTextEncoder(nn.Module):
    """
    Encoder module for Data2Vec model for text
    """

    def __init__(self, cfg, encoder: FairseqEncoder, device, mask_idx):
        super(Data2VecTextEncoder, self).__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.device = device
        self.teacher = self._build_teacher()
        self.mask_idx = mask_idx
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_scale = cfg.loss_scale
        self.regression_head = self._build_regression_head()
        self.num_updates = 0

    def _build_teacher(self):
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

        ema = EMA(self.sentence_encoder, self.cfg, self.device, skip_keys=skip_keys)
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

        def _get_annealed_rate(start, end, curr_step, total_steps):
            r = end - start
            pct_remaining = 1 - curr_step / total_steps
            return end - r * pct_remaining

        if self.training:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = _get_annealed_rate(self.cfg.ema_decay,
                                               self.cfg.ema_end_decay,
                                               num_updates,
                                               self.cfg.ema_anneal_end_step)
                self.teacher.decay = decay
            if self.teacher.decay < 1:
                self.teacher.step(self.sentence_encoder)

    def forward(self, src_tokens, target_tokens=None, features_only=False, return_all_hiddens=False):

        x, extra = self.encoder.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if features_only:
            return x, extra
        with torch.no_grad():
            self.teacher.model.eval()
            encoder_out = self.teacher.model(target_tokens, return_all_hiddens=True)
            y = encoder_out["fc_results"]

            y = y[-self.average_top_k_layers:]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                y = [tl.permute(1, 2, 0) for tl in y]  # TBC -> BCT
                permuted = True

            if self.cfg.batch_norm_target_layer:
                y = [F.batch_norm(tl.float(), running_mean=None, running_var=None, training=True) for tl in y]

            if self.cfg.instance_norm_target_layer:
                y = [F.instance_norm(tl.float()) for tl in y]

            if permuted:
                y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

            if self.cfg.layer_norm_target_layer:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)

            if not permuted:
                y = y.transpose(0, 1)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        masked_indices = src_tokens.eq(self.mask_idx)

        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        N = x.size(-1)
        if self.cfg.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none", beta=self.cfg.loss_beta).sum(dim=-1)

        result = {
            "losses": {
                "main": loss.sum() / math.sqrt(N) if self.loss_scale <= 0 else loss.sum() * self.loss_scale
            },
            "sample_size": loss.numel(),
        }

        # logging other values
        other_logs = {"ema_decay": self.teacher.decay * 1000}
        result["logs"] = other_logs
        return result
