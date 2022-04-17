import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder model using HuggingFace for audio i.e, Wav2Vec2

    Args:
        cfg: An omegaconf.DictConf instance containing all the configurations.
        **kwargs: extra args which are set as model properties
    """

    def __init__(self, cfg, **kwargs):
        super(Encoder, self).__init__()
        self.cfg = cfg
        checkpoint = cfg.model.encoder_checkpoint
        model_config = AutoConfig.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_config(model_config)
        self.__dict__.update(kwargs)

    def forward(self, inputs, mask=None, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            inputs: raw audio array
            mask: bool masked indices
            **kwargs: keyword args specific to the encoder's forward method

        Returns:
            A dictionary of the encoder outputs including transformer layers outputs and attentions outputs
        """
        outputs = self.encoder(inputs, mask_time_indices=mask, output_hidden_states=True,
                               output_attentions=True, **kwargs)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]  # last encoder output (accumulated)
        attentions = outputs['attentions']
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


if __name__ == '__main__':
    from dataset import TIMIT, DataCollatorForWav2Vec2Pretraining
    from omegaconf import OmegaConf
    from transformers import Wav2Vec2FeatureExtractor
    from torch.utils.data import DataLoader

    cfg = OmegaConf.load('configs/wav2vec2-pretraining.yaml')
    feature_extractor = Wav2Vec2FeatureExtractor()
    model = Encoder(cfg)
    dataset = TIMIT(cfg, 'train')
    collate_fn = DataCollatorForWav2Vec2Pretraining(model.encoder, feature_extractor, padding='longest')
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    itr = iter(loader)
    inputs, mask = next(itr)
    features = model(inputs, mask)
    print(features)
