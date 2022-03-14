import torch
from transformers import BeitFeatureExtractor, BeitModel, BeitConfig
import torch.nn as nn


class BEiT(nn.Module):
    """
    BEiT model using HuggingFace Transformers

    Args:
        cfg: An omegaconf.DictConf instance containing all the configurations.
        **kwargs: BEiT configs
    """

    def __init__(self, cfg, **kwargs):
        super(BEiT, self).__init__()
        self.cfg = cfg
        self.encoder = BeitModel(BeitConfig(**kwargs))

    def forward(self, src, **kwargs):
        """
        Forward inputs through the encoder and extract transformer/attention layers outputs

        Args:
            src: input pixels with shape [batch_size, channels, height, width]
            **kwargs: Input parameters to transformers.BeitModel

        Returns:
            A dictionary of encoder outputs
        """
        outputs = self.encoder(pixel_values=src, output_hidden_states=True, output_attentions=True, **kwargs)
        encoder_states = outputs['hidden_states'][:-1]  # encoder layers outputs separately
        encoder_out = outputs['hidden_states'][-1]  # last encoder output (accumulated)
        attentions = outputs['attentions']
        return {
            'encoder_states': encoder_states,
            'encoder_out': encoder_out,
            'attentions': attentions
        }


if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    import requests
    from torchvision import transforms as T

    model = BEiT(cfg=None)
    tokenizer = BEiT(BeitConfig)
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    image = T.Compose([T.Resize((224, 224)),
                       T.ToTensor(),
                       T.Normalize(mean=.5, std=.5)])(image).unsqueeze(0)
    outputs = model(image)
    print(outputs)
