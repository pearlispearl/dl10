import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage


class ObjectDataset(Dataset):
    def __init__(self, labels, transform=None):
        self.samples = list(labels.items())  # [(filepath, metadata)]
        self.transform = transform
        self.class_to_idx = {"Person": 0, "Car": 1, "Cat": 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, metadata = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        label = self.class_to_idx[metadata["class"]]
        bbox = torch.tensor(metadata["bbox"], dtype=torch.float32)  # [x, y, w, h] (normalized)

        # Convert bbox to absolute pixel values for augmentation
        w, h = img.size
        bbox_abs = BoundingBoxes(
            [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h], 
            format="xywh", 
            canvas_size=(h, w)  # (height, width) expected in BoundingBoxes
        )

        img = TVImage(img)  # Convert to torchvision tensor image format

        # Apply transformations (image + bbox)
        if self.transform:
            transformed = self.transform(img, bbox_abs)
            img, bbox_abs = transformed
        
        # Remove extra dimension
        bbox_abs = bbox_abs[0]

        # Convert bbox back to normalized format
        _, img_h, img_w = img.shape
        # print(img.shape, img_w, img_h)
        bbox_norm = torch.tensor([
            bbox_abs[0] / img_w, bbox_abs[1] / img_h, bbox_abs[2] / img_w, bbox_abs[3] / img_h
        ], dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.long), bbox_norm
