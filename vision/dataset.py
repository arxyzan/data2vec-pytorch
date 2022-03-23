import torch
from torchvision.datasets import ImageFolder

from .transform import MIMTransform


class BEiTPretrainingDataset(ImageFolder):

    def __init__(self, cfg, **kwargs):
        super(BEiTPretrainingDataset, self).__init__(root=cfg.dataset.path)
        self.transform = MIMTransform(cfg)
        self.device = cfg.device
        self.mask_token_id = 0.
        self.pad_token_id = 1.
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        # TODO figure out how the source and target must be constructed
        path, target = self.samples[index]
        image = self.loader(path)
        image, masked_image, bool_masked_pos = self.transform(image)
        return image, masked_image, bool_masked_pos


if __name__ == '__main__':
    from dall_e import load_model
    from omegaconf import OmegaConf
    from transformers import BeitModel, BeitConfig
    from torch.utils.data import DataLoader

    model = BeitModel(BeitConfig())
    d_vae = load_model('encoder.pkl')
    cfg = OmegaConf.load('configs/beit-pretraining.yaml')
    cfg.dataset.path = 'dummy_data'
    dataset = BEiTPretrainingDataset(cfg)
    loader = DataLoader(dataset, batch_size=4)
    src, trg = next(iter(loader))
    print(src)
