from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from timm.data.transforms import RandomResizedCropAndInterpolation

from transforms import BEiTTransform
from datasets import load_dataset

def build_beit_pretraining_dataset(cfg):
    transform = BEiTTransform(cfg)
    dataset = ImageFolder(cfg.path, transform=transform)
    train_len = int(len(dataset) * cfg.split_size)
    test_len = len(dataset) - train_len
    train, test = random_split(dataset, lengths=[train_len, test_len])
    return train, test


def build_finetune_dataset(cfg):
    # TODO
    ...
