import torch
from transformers import AutoModel, AutoConfig, Wav2Vec2FeatureExtractor
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
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint)
        self.__dict__.update(kwargs)

    def forward(self, src, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            src: raw audio array
            **kwargs: keyword args specific to the encoder's forward method

        Returns:
            A dictionary of the encoder outputs including transformer layers outputs and attentions outputs
        """
        src = self.feature_extractor(src, return_tensors='pt')
        outputs = self.encoder(**src, output_hidden_states=True, output_attentions=True, **kwargs)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]  # last encoder output (accumulated)
        attentions = outputs['attentions']
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


if __name__ == '__main__':
    from datasets import load_dataset
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('configs/wav2vec2-pretraining.yaml')
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sample = dataset[0]['audio']['array']

    model = Encoder(cfg)

    features = model(sample)
    print(features)


