"""
Run FACH Phase 1 training (no teachers — self-supervised fallback).
Saves ImgNet.pth and TxtNet.pth to checkpoints/mirflickr25k_64/
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from main import train

train(
    dataset='mirflickr25k',
    bit=64,
    device='cuda:0',
    max_epoch=20,
    pretrain_epochs=5,
    batch_size=64,
    lr=1e-4,
    margin=1.0,
    valid_freq=1,
    save_path='./checkpoints/substitute',
)
