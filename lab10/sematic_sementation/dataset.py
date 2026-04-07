import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import Mask, Image as TVImage


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing segmentation masks.
            transform (torchvision.transforms.v2.Compose, optional): Transformations to apply to images and masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))  # Sorting ensures matching order

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns:
            image (Tensor): Transformed image.
            mask (Tensor): Transformed segmentation mask.
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask = np.array(mask) - 1  # Make it start from 0

        # Convert to torchvision-compatible tensors
        img = TVImage(img)
        mask = Mask(torch.tensor(mask, dtype=torch.int64))  # Convert to tensor with class labels

        # Apply transformations
        if self.transform:
            # TODO: Apply augmentation on both images and masks
            # Hint: You can apply the same transformation on both image and mask using `self.transform(images, masks)`
            img, mask = self.transform(img, mask)

        return img, mask
