import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder model using HuggingFace Transformers for vision e.g, BeiT

    Args:
        cfg: An omegaconf.DictConf instance containing all the configurations.
        **kwargs: extra args which are set as dataset properties
    """

    def __init__(self, cfg, **kwargs):
        super(Encoder, self).__init__()
        self.cfg = cfg
        checkpoint = cfg.model.encoder_checkpoint
        model_config = AutoConfig.from_pretrained(checkpoint)
        self.encoder = AutoModel.from_config(model_config)
        self.vocab_size = model_config.vocab_size
        self.mask_token = self.encoder.embeddings.mask_token
        self.__dict__.update(kwargs)

    def forward(self, inputs, mask=None, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            inputs: input pixels with shape [batch_size, channels, height, width]
            mask: bool masked indices
            **kwargs: keyword args specific to the encoder's forward method

        Returns:
            A dictionary of the encoder outputs including transformer layers outputs and attentions outputs
        """
        # Note: inputs are already masked for MIM so mask is not used
        outputs = self.encoder(pixel_values=inputs, output_hidden_states=True, output_attentions=True, **kwargs)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]  # last encoder output (accumulated)
        attentions = outputs['attentions']

        # remove cls token from outputs
        encoder_states = [output[:, 1:, :] for output in encoder_states]
        encoder_out = encoder_out[:, 1:, :]
        attentions = [output[:, 1:, 1:] for output in attentions]

        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import numpy as np
    from PIL import Image
    import requests
    from torchvision import transforms as T

    cfg = OmegaConf.load('configs/beit-pretraining.yaml')
    model = Encoder(cfg)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = T.Compose([T.Resize((224, 224)),
                       T.ToTensor(),
                       T.Normalize(mean=.5, std=.5)])(image).unsqueeze(0)
    outputs = model(image)
    print(outputs)
