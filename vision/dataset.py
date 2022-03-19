import torch
from torchvision.datasets import ImageFolder

from .transforms import BEiTTransform


class BEiTPretrainingDataset(ImageFolder):
    def __init__(self, cfg, vae, **kwargs):
        super(BEiTPretrainingDataset, self).__init__(root=cfg.dataset.path)
        # discrete Variational AutoEncoder model (DALL-E)
        self.d_vae = vae
        self.transform = BEiTTransform(cfg.dataset)
        self.device = cfg.device
        self.mask_token_id = cfg.dataset.mask_token_id
        self.pad_token_id = cfg.dataset.pad_token_id
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        image, image_for_vae, bool_masked_pos = self.transform(image)
        with torch.no_grad():
            mask = bool_masked_pos.reshape(1, 14, 14, 1, 1)
            image = image.reshape(3, 14, 14, 16, 16)
            masked_src = torch.masked_fill(image, mask, -1)
            visual_tokens = self.d_vae(image_for_vae.unsqueeze(0)).argmax(1)
            unmasked_trg = torch.masked_fill(visual_tokens, ~bool_masked_pos.bool(), self.pad_token_id)
        return masked_src.flatten(), unmasked_trg.flatten()


if __name__ == '__main__':
    from dall_e import load_model
    from omegaconf import OmegaConf
    from transformers import BeitModel, BeitConfig
    from torch.utils.data import DataLoader

    model = BeitModel(BeitConfig())
    d_vae = load_model('encoder.pkl')
    cfg = OmegaConf.load('configs/beit-pretraining.yaml')
    dataset = BEiTPretrainingDataset(cfg, d_vae, mask_token_id=8192)
    loader = DataLoader(dataset, batch_size=4)
    src, trg = next(iter(loader))
    print(src)
