"""
PyTorch Dataset for FACH.

Split convention (matches UCCH):
  query    = last  <query_size>  samples
  database = first N - <query_size> samples
  train    = first <training_size> samples from database
             (training_size <= db_size, typically 10 000)

Supports two image modes:
  - VGG feature mode : images is (N, 4096) float32 numpy array
  - Raw image mode   : images is (N, 3, H, W) uint8 numpy array;
                       apply torchvision transforms before returning.
"""

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset as TorchDataset

# ImageNet mean/std
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def _make_transforms(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(img_size + 32),   # slight upscale then center-crop
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=_MEAN, std=_STD),
        ])


class Dataset(TorchDataset):
    """
    Args:
        opt      : config object — needs query_size, training_size, db_size.
        images   : (N, D) float32  or  (N, 3, H, W) uint8 numpy array.
        tags     : (N, T) float32 numpy array.
        labels   : (N, C) float32 numpy array.
        partition: 'train' | 'query' | 'database'
        img_size : target image size for raw-image mode (224 or 299 for IncV3).
    """

    def __init__(self, opt, images, tags, labels,
                 partition: str = 'train', img_size: int = 224):
        self.partition  = partition
        self.raw_images = images.ndim == 4
        N = labels.shape[0]

        db_end = N - opt.query_size
        if partition == 'query':
            idx = np.arange(db_end, N)
        elif partition == 'database':
            idx = np.arange(0, db_end)
        elif partition == 'train':
            size = min(opt.training_size, db_end)
            idx = np.arange(0, size)
        else:
            raise ValueError(f"Unknown partition: '{partition}'.")

        self.images = images[idx]
        self.tags   = tags[idx]
        self.labels = labels[idx]

        self.transform = _make_transforms(img_size, train=(partition == 'train'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.raw_images:
            img = img.transpose(1, 2, 0)    # (C,H,W) uint8 → (H,W,C)
            img = self.transform(img)        # Tensor (C, img_size, img_size)
        else:
            img = torch.from_numpy(img.astype(np.float32))

        txt   = torch.from_numpy(self.tags[idx].astype(np.float32))
        label = torch.from_numpy(self.labels[idx].astype(np.float32))

        if self.partition == 'train':
            return idx, img, txt, label
        else:
            return img, txt, label

    def get_labels(self):
        return torch.from_numpy(self.labels.astype(np.float32))
