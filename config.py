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
    dataset         = 'mirflickr25k'
    use_vgg_feat    = True

    # Dataset split sizes (Table I in paper)
    query_size      = 2000
    training_size   = 5000       # paper: FLICKR-25K train=5000
    db_size         = 18015      # 20015 - 2000
    num_label       = 24

    # Input dimensions
    image_dim       = 4096
    text_dim        = 1386

    # ── Model ─────────────────────────────────────────────────────────────────
    bit             = 64
    hidden_dim      = 4096
    use_backbone    = False
    dropout         = 0.0

    # ── Substitute model training ─────────────────────────────────────────────
    batch_size      = 64
    max_epoch       = 20
    pretrain_epochs = 5          # Phase 0: TxtNet pre-train epochs
    lr              = 1e-4
    margin          = 1.0        # m=1 (paper Sec. IV-B-3)
    tau_freq        = 20         # τ for low-freq mask

    # Sensitivity loss variant: 'lt1'|'lt2'|'lt3'|'lt4'
    sensitivity_loss = 'lt2'

    # ── Teacher / victim models ───────────────────────────────────────────────
    # List of dicts: {'type': 'DADH'|'UCCH', 'ckpt_path': '...', 'overrides': {}}
    teacher_configs = []

    # ── PGD attack ────────────────────────────────────────────────────────────
    T               = 100        # PGD iterations
    mu              = 0.001      # step size
    delta           = 0.3        # frequency-domain clip bound δ
    eps             = 8 / 255    # spatial L∞ bound ε
    madv            = 1.2        # hinge margin for Ladv (paper Sec. IV-B-3)
    targeted        = True       # paper uses targeted attack (t-mAP metric)

    # ── Evaluation ────────────────────────────────────────────────────────────
    valid_freq      = 1

    # ── Misc ──────────────────────────────────────────────────────────────────
    device          = 'cuda:0'
    save_path       = './checkpoints'

    # ──────────────────────────────────────────────────────────────────────────
    def _apply_dataset(self, name: str):
        name = name.lower()
        if name == 'mirflickr25k':
            self.dataset       = 'mirflickr25k'
            self.query_size    = 2000
            self.db_size       = 18015
            self.training_size = 5000    # paper Table I
            self.num_label     = 24
            self.image_dim     = 4096
            self.text_dim      = 1386
        elif name == 'nus_wide_tc10':
            self.dataset       = 'nus_wide_tc10'
            self.query_size    = 2100
            self.db_size       = 193734  # paper Table I: 195834 - 2100
            self.training_size = 10500   # paper Table I
            self.num_label     = 21      # paper uses 21 classes
            self.image_dim     = 4096
            self.text_dim      = 1000
        elif name == 'mscoco':
            self.dataset       = 'mscoco'
            self.query_size    = 2000
            self.db_size       = 121287  # 123287 - 2000
            self.training_size = 10000   # paper Table I
            self.num_label     = 80
            self.image_dim     = 4096
            self.text_dim      = 1024    # BERT features
        else:
            raise ValueError(f"Unknown dataset '{name}'. "
                             "Supported: 'mirflickr25k', 'nus_wide_tc10', 'mscoco'")

    def parse(self, kwargs: dict):
        if 'dataset' in kwargs:
            self._apply_dataset(kwargs['dataset'])
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn(f"Config has no attribute '{k}'")
            setattr(self, k, v)

        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        print("FACH Configuration:")
        for k, v in sorted(vars(self).items()):
            if not k.startswith('_'):
                print(f"  {k}: {v}")


opt = Config()
