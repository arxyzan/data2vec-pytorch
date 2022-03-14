import torch
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2FeatureExtractor
import torch.nn as nn


class Wav2Vec2(nn.Module):
    """
    Wav2Vec2 model using HuggingFace.

    Args:
        cfg: An omegaconf.DictConf instance containing all the configurations.
        **kwargs: Wav2Vec2 configs
    """

    def __init__(self, cfg, **kwargs):
        super(Wav2Vec2, self).__init__()
        self.cfg = cfg
        self.feature_extractor = Wav2Vec2FeatureExtractor(**kwargs)
        self.encoder = Wav2Vec2Model(Wav2Vec2Config(**kwargs))

    def forward(self, src, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            src: raw audio array
            **kwargs: Input parameters to transformers.Wav2Vec2Model

        Returns:
            A dictionary of encoder outputs
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

    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sample = dataset[0]['audio']['array']

    model = Wav2Vec2(None)

    features = model(sample)
    print(features)


