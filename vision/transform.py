"""
Image and Tensor transforms for Masked Image Modeling. Heavily inspired by https://github.com/Guillem96/data2vec-vision
"""
import math
import torch
import torchvision.transforms as T
from timm.data.transforms import RandomResizedCropAndInterpolation


class ViTPatchTransform:
    """
    Image to Patches transform derived from ViT
    """

    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, image: torch.Tensor):
        """
        Transform an image tensor to N patches
        Args:
            image:

        Returns:

        """
        if image.dim() == 4:
            ix = (2, 3)
        elif image.dim() == 3:
            ix = (1, 2)
        else:
            raise ValueError("Invalid dimensions")

        patches = image
        for i in ix:
            patches = patches.unfold(i, self.patch_size, self.patch_size)

        if image.dim() == 3:
            c = image.size(0)
            patches = patches.permute(1, 2, 0, 3, 4)
            return patches.reshape(-1, c, self.patch_size, self.patch_size)

        b, c, *_ = image.size()
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches.reshape(b, -1, c, self.patch_size, self.patch_size)


class BEiTMaskingTransform:
    def __init__(self, erase_percent):
        self.erase_percent = erase_percent
        self.rnd_erase_tfm = T.RandomErasing(p=1.,
                                             scale=(erase_percent, erase_percent),
                                             ratio=(0.3, 3.3))

    def __call__(self, image: torch.Tensor):
        """
        Apply random masking to the patches of image
        Args:
            image: image tensor

        Returns:

        """
        input_dim = image.dim()
        if input_dim == 4:
            num_patches, _, h, w = image.size()
            N = 1
        elif input_dim == 5:
            N, num_patches, _, h, w = image.size()
        else:
            raise ValueError(
                "Unexpected dimension numbers {0}".format(input_dim))

        num_patches_h = num_patches_w = int(math.sqrt(num_patches))
        masks = torch.ones(N, 1, num_patches_h, num_patches_w, device=image.device)
        masks = self.rnd_erase_tfm(masks)
        exp_mask = masks.view(N, 1, num_patches_h, num_patches_w, 1, 1)
        exp_mask = exp_mask.permute(0, 2, 3, 1, 4, 5)
        image = image.view(N, num_patches_h, num_patches_w, 3, h, w) * exp_mask
        image = image.view(N, num_patches_h * num_patches_w, 3, h, w)

        if input_dim == 4:
            return image.squeeze(), 1 - masks.view(num_patches_h * num_patches_w)

        return image, 1 - masks.view(N, num_patches_h * num_patches_w)


class PatchesToSequence:

    def __call__(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() == 5:
            batch_size, num_patches, *_ = patches.size()
            return patches.view(batch_size, num_patches, -1)
        num_patches = patches.size(0)
        return patches.view(num_patches, -1)


class MIMTransform:
    """
    Masked Image Modeling transforms
    """

    def __init__(self, cfg):
        self.image_size = cfg.dataset.input_size
        self.interpolation = cfg.dataset.interpolation
        self.patch_size = cfg.dataset.patch_size
        self.erase_percent = cfg.dataset.erase_percent
        self.common_transform = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4),
            T.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolation(size=self.image_size, interpolation=self.interpolation),
            T.ToTensor()
        ])
        self.patch_transform = ViTPatchTransform(self.patch_size)
        self.mask_transform = BEiTMaskingTransform(erase_percent=self.erase_percent)
        self.patch2sequence_transform = PatchesToSequence()

    def __call__(self, image):
        image = self.common_transform(image)
        patched_image = self.patch_transform(image)
        masked_image, bool_mask = self.mask_transform(patched_image)
        masked_image = self.patch2sequence_transform(masked_image)
        patched_image = self.patch2sequence_transform(patched_image)
        return patched_image, masked_image, bool_mask
