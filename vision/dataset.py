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
        self.mask_token_id = 0.
        self.pad_token_id = 1.
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        # TODO figure out how the source and target must be constructed
        path, target = self.samples[index]
        image = self.loader(path)
        image, image_for_vae, bool_masked_pos = self.transform(image)
        with torch.no_grad():
            mask = bool_masked_pos.reshape(1, 14, 14, 1, 1)
            image = image.reshape(3, 14, 14, 16, 16)
            masked_src = torch.masked_fill(image, mask, self.mask_token_id)
            visual_tokens = self.d_vae(image_for_vae.unsqueeze(0)).argmax(1)
            unmasked_trg = torch.masked_fill(visual_tokens, ~bool_masked_pos.bool(), self.pad_token_id)
        return masked_src.reshape(3, 224, 224), unmasked_trg.flatten(), mask


if __name__ == '__main__':
    from dall_e import load_model
    from omegaconf import OmegaConf
    from transformers import BeitModel, BeitConfig
    from torch.utils.data import DataLoader

    model = BeitModel(BeitConfig())
    d_vae = load_model('encoder.pkl')
    cfg = OmegaConf.load('configs/beit-pretraining.yaml')
    cfg.dataset.path = 'dummy_data'
    dataset = BEiTPretrainingDataset(cfg, d_vae, mask_token_id=8192)
    loader = DataLoader(dataset, batch_size=4)
    src, trg = next(iter(loader))
    print(src)
