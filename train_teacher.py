"""
Train a single teacher model (method × backbone combination).

Usage:
    python train_teacher.py --method DADH  --backbone VGG11 --bit 64 --dataset mirflickr25k
    python train_teacher.py --method DCMH  --backbone RN50  --bit 64 --dataset mirflickr25k
    python train_teacher.py --method DGCPN --backbone RN152 --bit 64 --dataset mirflickr25k
    python train_teacher.py --method UCCH  --backbone DN161 --bit 64 --dataset mirflickr25k

Checkpoints are saved to:
    checkpoints/teachers/<dataset>_<bit>/<METHOD>_<BACKBONE>.pth

After training several teachers, run train_substitute.py to distil them.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_data, Dataset
from methods.base import HashNet
from methods import DADH, DCMH, DGCPN, UCCH
from utils import calc_map_k
from backbones import BACKBONE_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Dataset config presets
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CFG = {
    'mirflickr25k': dict(
        query_size=2000, db_size=18015, training_size=10000,
        num_label=24, text_dim=1386,
    ),
    'nus_wide_tc10': dict(
        query_size=2100, db_size=184477, training_size=10500,
        num_label=10, text_dim=1000,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, images, tags, labels, cfg, device, img_size=224):
    model.eval()
    dl_kw = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    q_ds  = Dataset(cfg, images, tags, labels, partition='query',    img_size=img_size)
    db_ds = Dataset(cfg, images, tags, labels, partition='database', img_size=img_size)
    q_dl  = DataLoader(q_ds,  **dl_kw)
    db_dl = DataLoader(db_ds, **dl_kw)

    qBX, rBX, qBY, rBY = [], [], [], []
    for batch in q_dl:
        img, txt, _ = batch
        qBX.append(model.generate_img_code(img.to(device).float()).cpu())
        qBY.append(model.generate_txt_code(txt.to(device).float()).cpu())
    for batch in db_dl:
        img, txt, _ = batch
        rBX.append(model.generate_img_code(img.to(device).float()).cpu())
        rBY.append(model.generate_txt_code(txt.to(device).float()).cpu())

    qBX = torch.cat(qBX).to(device);  rBX = torch.cat(rBX).to(device)
    qBY = torch.cat(qBY).to(device);  rBY = torch.cat(rBY).to(device)
    q_lbl  = q_ds.get_labels().to(device)
    db_lbl = db_ds.get_labels().to(device)

    mapi2t = calc_map_k(qBX, rBY, q_lbl, db_lbl).item()
    mapt2i = calc_map_k(qBY, rBX, q_lbl, db_lbl).item()
    model.train()
    return mapi2t, mapt2i


# ──────────────────────────────────────────────────────────────────────────────
# Method dispatchers
# ──────────────────────────────────────────────────────────────────────────────

def _train_dadh(model, train_dl, cfg, device):
    from methods.DADH import Discriminator, create_optimizers, train_epoch

    # Feature dim used inside discriminator = hidden_dim // 2
    dis = Discriminator(
        feat_dim=cfg.hidden_dim // 2,
        hidden_dim=cfg.hidden_dim // 4,
        bit=cfg.bit,
    ).to(device)

    opt_G, opt_D_feat, opt_D_hash = create_optimizers(model, dis, cfg)

    N = cfg.training_size
    B_buf  = torch.randn(N, cfg.bit).sign()
    H_i_buf = torch.zeros(N, cfg.bit)
    H_t_buf = torch.zeros(N, cfg.bit)
    DADH.train_epoch._proj = None   # reset projection cache

    def one_epoch(epoch):
        return train_epoch(model, dis, train_dl,
                           opt_G, opt_D_feat, opt_D_hash,
                           B_buf, H_i_buf, H_t_buf,
                           epoch, cfg, device)
    return one_epoch


def _train_dcmh(model, train_dl, cfg, device):
    from methods.DCMH import create_optimizers, train_epoch

    opt_img, opt_txt = create_optimizers(model, cfg)
    N = cfg.training_size
    F_buf = torch.zeros(N, cfg.bit)
    G_buf = torch.zeros(N, cfg.bit)
    B_buf = torch.randn(N, cfg.bit).sign()

    def one_epoch(epoch):
        return train_epoch(model, train_dl, opt_img, opt_txt,
                           F_buf, G_buf, B_buf, epoch, cfg, device)
    return one_epoch


def _train_dgcpn(model, train_dl, cfg, device):
    from methods.DGCPN import create_optimizers, train_epoch

    opt_img, opt_txt = create_optimizers(model, cfg)

    def one_epoch(epoch):
        return train_epoch(model, train_dl, opt_img, opt_txt,
                           epoch, cfg, device)
    return one_epoch


def _train_ucch(model, train_dl, cfg, device):
    from methods.UCCH import NCEAverage, NCESoftmaxLoss, ContrastiveLoss, create_optimizer, train_epoch

    # Override global defaults with UCCH paper values
    if cfg.lr <= 5e-5:
        cfg.lr = 1e-4
    if cfg.weight_decay >= 1e-4:
        cfg.weight_decay = 1e-6
    cfg.ucch_alpha = 0.9

    n_data = cfg.training_size
    K      = min(4096, n_data - 1)   # negatives per sample
    warmup = 1                        # 1 warmup epoch (backbone frozen)

    # NCEAverage: T=0.9 is scaled internally to T*sqrt(bit)=7.2 for 64-bit
    contrast      = NCEAverage(cfg.bit, n_data, K, T=0.9, momentum=0.4).cuda()
    criterion_nce = NCESoftmaxLoss().to(device)
    criterion_cont = ContrastiveLoss(margin=0.2, shift=1.0).to(device)
    optimizer = create_optimizer(model, cfg)
    # CosineAnnealingLR: decay lr from initial value to 1e-6 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=1e-6)

    def one_epoch(epoch):
        loss = train_epoch(model, train_dl, optimizer,
                           contrast, criterion_nce, criterion_cont,
                           epoch, warmup, cfg, device)
        scheduler.step()
        return loss
    return one_epoch


_DISPATCHERS = {
    'DADH':  _train_dadh,
    'DCMH':  _train_dcmh,
    'DGCPN': _train_dgcpn,
    'UCCH':  _train_ucch,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train one teacher model (method × backbone) for FACH.')
    parser.add_argument('--method',   type=str, required=True,
                        choices=['DADH', 'DCMH', 'DGCPN', 'UCCH'],
                        help='Hashing method.')
    parser.add_argument('--backbone', type=str, required=True,
                        choices=BACKBONE_NAMES,
                        help='Image backbone.')
    parser.add_argument('--bit',      type=int, default=64,
                        help='Hash code length.')
    parser.add_argument('--dataset',  type=str, default='mirflickr25k',
                        choices=list(DATASET_CFG.keys()),
                        help='Dataset name.')
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',       type=float, default=5e-5)
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--device',   type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/teachers')
    parser.add_argument('--valid_freq', type=int, default=5,
                        help='Evaluate every N epochs.')
    # DADH specific
    parser.add_argument('--alpha',    type=float, default=10.0)
    parser.add_argument('--beta',     type=float, default=1.0)
    parser.add_argument('--gamma',    type=float, default=1.0)
    parser.add_argument('--mu',       type=float, default=1e-5)
    parser.add_argument('--lamb',     type=float, default=1.0)
    parser.add_argument('--margin',   type=float, default=0.4)
    # DCMH specific
    parser.add_argument('--eta',      type=float, default=1.0)
    parser.add_argument('--gamma_bal', type=float, default=1.0)
    # DGCPN specific
    parser.add_argument('--a1',       type=float, default=0.01)
    parser.add_argument('--a2',       type=float, default=0.3)
    parser.add_argument('--K_diag',   type=float, default=1.5)
    parser.add_argument('--knn_number', type=int, default=3000)
    parser.add_argument('--scale',    type=float, default=4000.0)
    parser.add_argument('--dw',       type=float, default=1.0)
    parser.add_argument('--cw',       type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # UCCH specific
    parser.add_argument('--tau',        type=float, default=0.07)
    parser.add_argument('--nce_weight', type=float, default=1.0)
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet-pretrained backbone (default: True). '
                             'Pass --no-pretrained to train from scratch like original UCCH.')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')

    cfg = parser.parse_args()
    cfg.max_epoch = cfg.epochs            # DCMH uses max_epoch for LR decay

    # Inherit dataset split sizes
    for k, v in DATASET_CFG[cfg.dataset].items():
        setattr(cfg, k, v)

    # UCCH trains on the full database (18015 for mirflickr25k), not the 10000
    # subset — this matches the original UCCH paper's training setup exactly.
    if cfg.method == 'UCCH':
        cfg.training_size = cfg.db_size

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Training teacher: {cfg.method} × {cfg.backbone}")
    print(f"  Dataset: {cfg.dataset}  |  Bit: {cfg.bit}")
    print(f"  Device:  {device}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    # UCCH paper trains on pre-extracted VGG features (mirflickr25k_fea mode),
    # not raw images — this is how it converges in ~10 epochs and reaches 0.75.
    # All other methods train end-to-end with raw images + backbone.
    ucch_fea_mode = (cfg.method == 'UCCH')
    img_size = 299 if cfg.backbone == 'IncV3' else 224
    print("Loading data ...")
    images, tags, labels = load_data(cfg.dataset, use_vgg_feat=ucch_fea_mode)
    train_ds = Dataset(cfg, images, tags, labels, partition='train',
                       img_size=img_size)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=0, drop_last=True)

    # ── Build model ───────────────────────────────────────────────────────────
    if cfg.method == 'UCCH':
        # Feature mode: pre-extracted VGG features → ImageNet hash head, no backbone.
        # Matches UCCH's reported results (nets/ImageNet.py hiden_layer=3,
        # nets/TextNet.py hiden_layer=2, hidden dim 8192, tanh+L2-norm output).
        from methods.UCCH import UCCHHashNet
        feat_dim = images.shape[1]   # 4096 for VGG features
        model = UCCHHashNet(
            bit=cfg.bit,
            text_dim=cfg.text_dim,
            feat_dim=feat_dim,
            backbone_name=None,       # no backbone — features already extracted
        ).to(device)
    else:
        model = HashNet(
            backbone_name=cfg.backbone,
            bit=cfg.bit,
            text_dim=cfg.text_dim,
            hidden_dim=cfg.hidden_dim,
            pretrained=cfg.pretrained,
        ).to(device)

    # ── Build training closure ────────────────────────────────────────────────
    one_epoch_fn = _DISPATCHERS[cfg.method](model, train_dl, cfg, device)

    # ── Output path ───────────────────────────────────────────────────────────
    # Structure: <save_dir>/<METHOD>/<METHOD>_<BACKBONE>_<BIT>.pth
    # Structure: <save_dir>/<METHOD>/<METHOD>_<BACKBONE>_<BIT>_<DATASET>.pth
    ckpt_dir  = os.path.join(cfg.save_dir, cfg.method)
    ckpt_path = os.path.join(ckpt_dir, f'{cfg.method}_{cfg.backbone}_{cfg.bit}_{cfg.dataset}.pth')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_avg = 0.0
    for epoch in range(cfg.epochs):
        loss = one_epoch_fn(epoch)
        print(f'  Epoch {epoch+1:3d}/{cfg.epochs}  loss={loss:.4f}')

        if (epoch + 1) % cfg.valid_freq == 0:
            mapi2t, mapt2i = evaluate(model, images, tags, labels, cfg, device, img_size)
            avg = 0.5 * (mapi2t + mapt2i)
            print(f'    MAP i→t={mapi2t:.4f}  t→i={mapt2i:.4f}  avg={avg:.4f}')
            if avg > best_avg:
                best_avg = avg
                model.save(ckpt_path)
                print(f'    ✓ Saved → {ckpt_path}')

    print(f'\nDone. Best avg MAP = {best_avg:.4f}')
    print(f'Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
