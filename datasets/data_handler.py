"""
Data loading utilities for FACH.

Mirrors UCCH's data loading convention:
  - Raw images stored in h5py .mat files (key 'IAll', uint8, N×3×224×224)
  - VGG precomputed features in scipy .mat files (key 'XAll', float32, N×4096)
  - Text tags in scipy .mat files (key 'YAll')
  - Labels   in scipy .mat files (key 'LAll')

Data split (same as UCCH):
  query    = last  <test_size> samples
  database = first N - <test_size> samples  (also used as train set)
"""

import os
import numpy as np
import h5py
import scipy.io as sio

# ── Root directory: FACH/data/ ────────────────────────────────────────────────
_DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_data(data_name: str, use_vgg_feat: bool = True):
    """
    Load images (or VGG features), text tags, and labels.

    Args:
        data_name    : 'mirflickr25k' | 'nus_wide_tc10'
        use_vgg_feat : if True, load precomputed 4096-dim VGG features;
                       if False, load raw uint8 images (N×3×224×224).

    Returns:
        images : (N, 4096) float32  or  (N, 3, 224, 224) uint8
        tags   : (N, text_dim) float32
        labels : (N, num_class) float32
    """
    name = data_name.lower()
    if name == 'mirflickr25k':
        if use_vgg_feat:
            return _load_flickr25k_fea()
        else:
            return _load_flickr25k_raw()
    elif name == 'nus_wide_tc10':
        if use_vgg_feat:
            return _load_nuswide_fea()
        else:
            raise NotImplementedError(
                "Raw NUS-WIDE images not available in the current data directory. "
                "Use use_vgg_feat=True instead."
            )
    else:
        raise ValueError(f"Unknown dataset: {data_name}. "
                         f"Supported: 'mirflickr25k', 'nus_wide_tc10'")


# ──────────────────────────────────────────────────────────────────────────────
# FLICKR-25K — raw images  (data/FLICKR-25K/)
# ──────────────────────────────────────────────────────────────────────────────

def _load_flickr25k_raw():
    """
    Loads raw uint8 images (N×3×224×224) from h5py .mat file.
    Returns images as uint8 array (no normalization — callers apply transforms).
    """
    flickr_dir = os.path.join(_DATA_ROOT, 'FLICKR-25K')
    imgs_path   = os.path.join(flickr_dir, 'mirflickr25k-iall.mat')
    tags_path   = os.path.join(flickr_dir, 'mirflickr25k-yall.mat')
    labels_path = os.path.join(flickr_dir, 'mirflickr25k-lall.mat')

    with h5py.File(imgs_path, 'r') as f:
        # h5py reads MATLAB arrays with reversed dim order;
        # stored as (N, 3, 224, 224) uint8 — already in (N, C, H, W) shape
        images = f['IAll'][()]          # (N, 3, 224, 224), uint8

    tags   = sio.loadmat(tags_path)['YAll'].astype(np.float32)
    labels = sio.loadmat(labels_path)['LAll'].astype(np.float32)
    return images, tags, labels


# ──────────────────────────────────────────────────────────────────────────────
# FLICKR-25K — VGG features  (data/MIRFLICKR25K/)
# ──────────────────────────────────────────────────────────────────────────────

def _load_flickr25k_fea():
    """
    Loads precomputed 4096-dim VGG features for FLICKR-25K.
    Features are L2-normalised to unit length (common practice).
    """
    fea_dir     = os.path.join(_DATA_ROOT, 'MIRFLICKR25K')
    imgs_path   = os.path.join(fea_dir, 'mirflickr25k-iall-vgg.mat')
    tags_path   = os.path.join(fea_dir, 'mirflickr25k-yall.mat')
    labels_path = os.path.join(fea_dir, 'mirflickr25k-lall.mat')

    images = sio.loadmat(imgs_path)['XAll'].astype(np.float32)  # (N, 4096)
    images = _l2_normalize(images)

    tags   = sio.loadmat(tags_path)['YAll'].astype(np.float32)
    labels = sio.loadmat(labels_path)['LAll'].astype(np.float32)
    return images, tags, labels


# ──────────────────────────────────────────────────────────────────────────────
# NUS-WIDE-TC10 — VGG features  (data/NUS-WIDE-TC10/)
# ──────────────────────────────────────────────────────────────────────────────

def _load_nuswide_fea():
    """
    Loads precomputed 4096-dim VGG features for NUS-WIDE-TC10 (10 classes).
    """
    nus_dir     = os.path.join(_DATA_ROOT, 'NUS-WIDE-TC10')
    imgs_path   = os.path.join(nus_dir, 'nus-wide-tc10-xall-vgg.mat')
    tags_path   = os.path.join(nus_dir, 'nus-wide-tc10-yall.mat')
    labels_path = os.path.join(nus_dir, 'nus-wide-tc10-lall.mat')

    images = sio.loadmat(imgs_path)['XAll'].astype(np.float32)  # (N, 4096)
    images = _l2_normalize(images)

    tags   = sio.loadmat(tags_path)['YAll'].astype(np.float32)
    labels = sio.loadmat(labels_path)['LAll'].astype(np.float32)
    return images, tags, labels


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm
