"""
Available teacher hashing methods:
    DADH  — Deep Adversarial Discrete Hashing
    DCMH  — Deep Cross-Modal Hashing
    DGCPN — Deep Graph-based Cross-Modal Proximity Network
    UCCH  — Unsupervised Contrastive Cross-Modal Hashing
"""

from .base import HashNet, TextNet
from . import DADH, DCMH, DGCPN, UCCH

METHOD_NAMES = ['DADH', 'DCMH', 'DGCPN', 'UCCH']
