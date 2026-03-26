"""
Configuration for FACH.

Usage:
    from config import opt
    opt.parse({'dataset': 'mirflickr25k', 'bit': 64, 'device': 'cuda:0'})
"""

import warnings
import torch


class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    # 'mirflickr25k'  or  'nus_wide_tc10'
    dataset         = 'mirflickr25k'
    use_vgg_feat    = True       # True  → VGG-4096 features (fast, default)
                                 # False → raw 224×224 images (needs use_backbone=True)

    # Dataset split sizes
    query_size      = 2000       # last N samples as query
    training_size   = 10000      # samples used for substitute model training
    db_size         = 18015      # FLICKR-25K: 20015 - 2000
    num_label       = 24

    # Input dimensions
    image_dim       = 4096       # VGG-F feature dim (ignored when use_backbone=True)
    text_dim        = 1386       # FLICKR-25K text dim

    # ── Model ─────────────────────────────────────────────────────────────────
    bit             = 64         # Hash code length K (16 / 32 / 64 / 128)
    hidden_dim      = 4096       # Hidden layer width in substitute model MLP
    use_backbone    = False      # True → VGG11 backbone on raw images
    dropout         = 0.0

    # ── Substitute model training ─────────────────────────────────────────────
    batch_size      = 64
    max_epoch       = 20
    lr              = 1e-4
    margin          = 1.5        # Boundary margin m in L_ME (Eq. 14)
    tau_freq        = 20         # Low-frequency threshold τ for mask M_low

    # Sensitivity loss variant for frequency analysis:
    # 'lt1' (triplet) | 'lt2' (margin, default) | 'lt3' (sign) | 'lt4' (contrastive)
    sensitivity_loss = 'lt2'

    # ── Teacher / victim models ───────────────────────────────────────────────
    # List of dicts: {'type': 'DADH'|'DCMH'|'UCCH', 'ckpt_path': '...', 'overrides': {}}
    teacher_configs = []

    # ── PGD attack ────────────────────────────────────────────────────────────
    T               = 100        # PGD iterations
    mu              = 0.001      # PGD step size
    delta           = 0.3        # Frequency-domain clip bound δ
    eps             = 8 / 255    # Spatial L∞ bound ε
    targeted        = False      # True → targeted attack

    # ── Evaluation ────────────────────────────────────────────────────────────
    valid_freq      = 1          # Validate every N epochs

    # ── Misc ──────────────────────────────────────────────────────────────────
    device          = 'cuda:0'
    save_path       = './checkpoints'

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_dataset(self, name: str):
        name = name.lower()
        if name == 'mirflickr25k':
            self.dataset      = 'mirflickr25k'
            self.query_size   = 2000
            self.db_size      = 18015   # 20015 - 2000
            self.training_size = 10000
            self.num_label    = 24
            self.image_dim    = 4096
            self.text_dim     = 1386
        elif name == 'nus_wide_tc10':
            self.dataset      = 'nus_wide_tc10'
            self.query_size   = 2100
            self.db_size      = 184477  # 186577 - 2100
            self.training_size = 10500
            self.num_label    = 10
            self.image_dim    = 4096
            self.text_dim     = 1000
        else:
            raise ValueError(f"Unknown dataset '{name}'. "
                             "Supported: 'mirflickr25k', 'nus_wide_tc10'")

    def parse(self, kwargs: dict):
        # Apply dataset preset first
        if 'dataset' in kwargs:
            self._apply_dataset(kwargs['dataset'])
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f"Config has no attribute '{k}'")
            setattr(self, k, v)

        # Resolve device string → torch.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        print("FACH Configuration:")
        for k, v in sorted(vars(self).items()):
            if not k.startswith('_'):
                print(f"  {k}: {v}")


opt = Config()
