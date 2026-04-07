import os
import numpy as np
import glob
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import OxfordIIITPet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Create directory for train and valid sets
data_dir = './oxford-pet-dataset'
for s in ['train', 'valid', 'test']:
    for t in ['img', 'mask']:
        new_dir = f'./oxford-pet-dataset/{s}_{t}'
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)

# Download dataset
OxfordIIITPet(
    root='./',
    split='trainval',
    target_types='segmentation',
    download=True)

oxford_img_dir = './oxford-iiit-pet/images'
oxford_mask_dir = './oxford-iiit-pet/annotations/trimaps'

n_trains = 2000
n_valids = 500
n_tests = 500

# Split
img_paths = glob.glob(
    os.path.join(oxford_img_dir,'*.jpg'))

dataset_size = len(img_paths)
np.random.shuffle(img_paths)
train_paths = img_paths[:n_trains]
val_paths = img_paths[n_trains:n_trains+n_valids]
test_paths = img_paths[n_trains+n_valids:n_trains+n_valids+n_tests]

# Copy files
for split_name, paths in [('train', train_paths), ('valid', val_paths), ('test', test_paths)]:
    for src_img in paths:
        print(src_img)
        fn = os.path.basename(src_img)
        mask_fn = fn.replace('.jpg', '.png')
        src_mask = os.path.join(oxford_mask_dir, mask_fn)
        dst_img = os.path.join(data_dir, f'{split_name}_img', fn)
        dst_mask = os.path.join(data_dir, f'{split_name}_mask', mask_fn)
        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)