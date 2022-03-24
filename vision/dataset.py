import torch
from torchvision.datasets import ImageFolder

from .transforms import MIMTransform


class MIMPretrainingDataset(ImageFolder):

    def __init__(self, cfg, **kwargs):
        super(MIMPretrainingDataset, self).__init__(root=cfg.dataset.path)
        self.transform = MIMTransform(cfg.dataset)
        self.input_size = cfg.dataset.input_size
        self.device = cfg.device
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        # TODO figure out how the source and target must be constructed
        path, target = self.samples[index]
        image = self.loader(path)
        image, mask = self.transform(image)
        mask = mask.reshape(1, 14, 14, 1, 1)
        image = image.reshape(-1, 14, 14, 16, 16)
        masked_image = (image * mask).reshape(-1, self.input_size, self.input_size)
        target_image = (image * ~mask).reshape(-1, self.input_size, self.input_size)
        return masked_image, target_image, mask.reshape(14, 14)


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from transformers import BeitModel, BeitConfig
    from torch.utils.data import DataLoader

    model = BeitModel(BeitConfig())
    cfg = OmegaConf.load('configs/beit-pretraining.yaml')
    cfg.dataset.path = 'dummy_data'
    dataset = MIMPretrainingDataset(cfg)
    loader = DataLoader(dataset, batch_size=4)
    src, trg = next(iter(loader))
    print(src)
