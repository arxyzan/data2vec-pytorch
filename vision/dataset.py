import torch
from torchvision.datasets import ImageFolder

from .transforms import MIMTransform


class MIMPretrainingDataset(ImageFolder):
    """
    Dataset for Masked Image Modeling derived from BEiT.

    Given an image, the common transforms and augmentations are applied like random crop, color jitter, etc., then the
    image is split into 14x14 patches and some patches are masked randomly. The input image to the model is the masked
    image and the target image is the full image

    Args:
        cfg (DictConfig): config containing model, dataset, etc. properties
        split: either `train` or `test`
        **kwargs: extra args which are set as dataset properties

    """

    def __init__(self, cfg, split, **kwargs):
        super(MIMPretrainingDataset, self).__init__(root=cfg.dataset.path[split])
        self.transform = MIMTransform(cfg.dataset)
        self.input_size = cfg.dataset.input_size
        self.device = cfg.device
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        """
        Load image from disk, transform the image (augmentation and randomly mask some patches)

        Args:
            index: index to the image

        Returns:
            a tuple of masked image, unmasked target image and bool masked positions
        """
        path, target = self.samples[index]
        image = self.loader(path)
        image, mask = self.transform(image)
        mask = mask.reshape(1, 14, 14, 1, 1)
        image = image.reshape(-1, 14, 14, 16, 16)
        masked_image = (image * mask).reshape(-1, self.input_size, self.input_size)
        target_image = image.reshape(-1, self.input_size, self.input_size)
        return masked_image, target_image, mask.flatten().bool()


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    cfg = OmegaConf.load('configs/beit-pretraining.yaml')
    cfg.dataset.path = 'dummy_data'
    dataset = MIMPretrainingDataset(cfg, split='train')
    loader = DataLoader(dataset, batch_size=4)
    src, trg = next(iter(loader))
    print(src)
