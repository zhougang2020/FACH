"""
Teacher model loader for FACH.

Paper uses 12 teacher models:
  - DADH × 6 backbones (AlexNet, VGG11, RN50, RN152, Inc-v3, DN161)
  - UCCH × 6 backbones (AlexNet, VGG11, RN50, RN152, Inc-v3, DN161)

Each teacher callable:  x_features (B, D) → h_real (B, K)

Checkpoint naming convention:
  checkpoints/teachers/<METHOD>/<METHOD>_<BACKBONE>_<BIT>_<DATASET>.pth

For DADH: the GEN model's image branch is used.
For UCCH: the UCCHHashNet image branch is used.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Learnable teacher weight vector W  (Eq. 9)
# ──────────────────────────────────────────────────────────────────────────────

class TeacherWeights(nn.Module):
    """
    Learnable scalar importance weight per teacher (W^T in Eq. 9).
    Initialised to uniform; trained jointly with the substitute model.
    """
    def __init__(self, num_teachers: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_teachers))

    def forward(self) -> torch.Tensor:
        """Returns softmax-normalised weights (M,)."""
        return F.softmax(self.w, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Generic wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TeacherWrapper(nn.Module):
    def __init__(self, model, forward_fn):
        super().__init__()
        self.model = model
        self._forward_fn = forward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(self.model, x)


# ──────────────────────────────────────────────────────────────────────────────
# DADH loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_dadh(ckpt_path: str, bit: int, opt_overrides: dict, device):
    """
    Load DADH GEN model image branch as teacher.
    Expects checkpoint at ckpt_path (full path to .pth file).
    """
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'DADH')
    sys.path.insert(0, os.path.abspath(base_dir))
    try:
        from models.gen_model import GEN

        image_dim  = opt_overrides.get('image_dim', 4096)
        text_dim   = opt_overrides.get('text_dim', 1386)
        hidden_dim = opt_overrides.get('hidden_dim', 8192)

        model = GEN(
            dropout=False,
            image_dim=image_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            output_dim=bit,
            pretrain_model=None,
        ).to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def fwd(m, x):
            # GEN.forward returns (x_code, y_code, f_x, f_y)
            x_code, _, _, _ = m(x, torch.zeros(x.shape[0], text_dim, device=x.device))
            return x_code

        return TeacherWrapper(model, fwd)
    finally:
        sys.path.pop(0)


# ──────────────────────────────────────────────────────────────────────────────
# UCCH loader
# ──────────────────────────────────────────────────────────────────────────────

def _load_ucch(ckpt_path: str, bit: int, opt_overrides: dict, device):
    """
    Load UCCH UCCHHashNet image branch as teacher.
    Expects checkpoint at ckpt_path (full path to .pth file).
    """
    from methods.UCCH import UCCHHashNet

    feat_dim = opt_overrides.get('feat_dim', 4096)
    text_dim = opt_overrides.get('text_dim', 1386)

    model = UCCHHashNet(
        bit=bit,
        text_dim=text_dim,
        feat_dim=feat_dim,
        backbone_name=None,   # feature mode
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def fwd(m, x):
        return m.forward_img(x)

    return TeacherWrapper(model, fwd)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    'DADH': _load_dadh,
    'UCCH': _load_ucch,
}


def load_teachers(teacher_configs: list, bit: int, device) -> list:
    """
    Load all teacher models specified in teacher_configs.

    Args:
        teacher_configs : list of dicts:
            {'type': 'DADH'|'UCCH', 'ckpt_path': str, 'overrides': dict}
        bit    : hash code length.
        device : torch.device.

    Returns:
        list of TeacherWrapper instances.
    """
    teachers = []
    for cfg in teacher_configs:
        model_type = cfg['type'].upper()
        ckpt_path  = cfg['ckpt_path']
        overrides  = cfg.get('overrides', {})

        if model_type not in _LOADERS:
            raise ValueError(
                f"Unknown teacher type '{model_type}'. "
                f"Supported: {list(_LOADERS.keys())}")

        if not os.path.exists(ckpt_path):
            print(f"[teacher_loader] WARNING: checkpoint not found: {ckpt_path} — skipping.")
            continue

        teacher = _LOADERS[model_type](ckpt_path, bit, overrides, device)
        teachers.append(teacher)
        print(f"[teacher_loader] Loaded {model_type} from {ckpt_path}")

    return teachers


# ──────────────────────────────────────────────────────────────────────────────
# Consensus target code  (voting strategy from CSQ, paper Sec. III-D)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_consensus_code(teachers: list, x: torch.Tensor) -> torch.Tensor:
    """
    t = sign(1/M * sum_m h^(m)(x))

    Returns (B, K) binary codes in {-1, +1}.
    """
    if not teachers:
        raise ValueError("No teachers provided.")
    agg = None
    M = len(teachers)
    for teacher in teachers:
        h = teacher(x)
        agg = h if agg is None else agg + h
    return (agg / M).sign()
