"""
Teacher model loader for FACH.

Supports loading pre-trained victim models from the sibling directories
(DADH, DCMH, DGCPN, UCCH) and wrapping them in a unified callable interface.

Each teacher callable:   x_features (B, D) → h_real (B, K)

Usage example:
    from teacher_loader import load_teachers
    teachers = load_teachers(opt.teacher_configs, opt.bit, opt.device)
    # teachers is a list of callables
"""

import os
import sys
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────────────────
# Generic wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TeacherWrapper(nn.Module):
    """Wraps an existing model so it exposes a unified forward(x) → h_real."""

    def __init__(self, model, forward_fn):
        super().__init__()
        self.model = model
        self._forward_fn = forward_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fn(self.model, x)


# ──────────────────────────────────────────────────────────────────────────────
# Per-method loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_dadh(ckpt_path: str, bit: int, opt_overrides: dict, device):
    """Load a pre-trained DADH generator as a teacher."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'DADH')
    sys.path.insert(0, os.path.abspath(base_dir))
    try:
        from models.gen_model import GEN
        import scipy.io as scio

        # Minimal config compatible with DADH
        class _Opt:
            dropout   = False
            image_dim = opt_overrides.get('image_dim', 4096)
            text_dim  = opt_overrides.get('text_dim', 1386)
            hidden_dim = opt_overrides.get('hidden_dim', 8192)

        pretrain = None
        model = GEN(
            _Opt.dropout, _Opt.image_dim, _Opt.text_dim,
            _Opt.hidden_dim, bit, pretrain_model=pretrain,
        ).to(device)

        state = torch.load(os.path.join(ckpt_path, 'GEN.pth'), map_location=device)
        model.load_state_dict(state)
        model.eval()

        def fwd(m, x):
            h_i, _, _, _ = m(x, torch.zeros(x.shape[0], _Opt.text_dim, device=x.device))
            return h_i

        return TeacherWrapper(model, fwd)
    finally:
        sys.path.pop(0)


def _load_dcmh(ckpt_path: str, bit: int, opt_overrides: dict, device):
    """Load a pre-trained DCMH image module as a teacher."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'DCMH')
    sys.path.insert(0, os.path.abspath(base_dir))
    try:
        from models.img_module import ImgModule

        model = ImgModule(bit).to(device)
        model.load(os.path.join(ckpt_path, 'ImgModule.pth'))
        model.eval()

        def fwd(m, x):
            return m(x)

        return TeacherWrapper(model, fwd)
    finally:
        sys.path.pop(0)


def _load_ucch(ckpt_path: str, bit: int, opt_overrides: dict, device):
    """Load a pre-trained UCCH image network as a teacher."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'UCCH')
    sys.path.insert(0, os.path.abspath(base_dir))
    try:
        from nets.img_net import ImgNet

        image_dim = opt_overrides.get('image_dim', 4096)
        model = ImgNet(bit, image_dim).to(device)
        state = torch.load(os.path.join(ckpt_path, 'ImgNet.pth'), map_location=device)
        model.load_state_dict(state)
        model.eval()

        def fwd(m, x):
            h, _ = m(x)
            return h

        return TeacherWrapper(model, fwd)
    finally:
        sys.path.pop(0)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

_LOADERS = {
    'DADH': _load_dadh,
    'DCMH': _load_dcmh,
    'UCCH': _load_ucch,
}


def load_teachers(teacher_configs: list, bit: int, device) -> list:
    """
    Load all teacher models specified in teacher_configs.

    Args:
        teacher_configs : list of dicts with keys:
                          {'type': str, 'ckpt_path': str, 'overrides': dict (opt)}
        bit             : hash code length.
        device          : torch.device.

    Returns:
        list of callable teacher wrappers (each: x → h_real).
    """
    teachers = []
    for cfg in teacher_configs:
        model_type = cfg['type'].upper()
        ckpt_path  = cfg['ckpt_path']
        overrides  = cfg.get('overrides', {})

        if model_type not in _LOADERS:
            raise ValueError(
                f"Unknown teacher type '{model_type}'. "
                f"Supported: {list(_LOADERS.keys())}"
            )
        loader = _LOADERS[model_type]
        teacher = loader(ckpt_path, bit, overrides, device)
        teachers.append(teacher)
        print(f"[teacher_loader] Loaded {model_type} from {ckpt_path}")

    return teachers


# ──────────────────────────────────────────────────────────────────────────────
# Consensus target code from teachers  (for L_ME target t)
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
        h = teacher(x)  # (B, K)
        agg = h if agg is None else agg + h
    return (agg / M).sign()
