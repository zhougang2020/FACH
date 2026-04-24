"""
Train the FACH substitute model (Phase 1).

Key differences from initial version (aligned with paper):
  1. TextNet is pre-trained first using cross-modal hash loss, then frozen.
  2. Learnable teacher weight vector W (Eq. 9) trained jointly with ImgNet.
  3. Training set size: FLICKR-25K=5000, NUS-WIDE=10500 (paper Table I).
  4. margin m=1 (paper Sec. IV-B-3).
  5. 12 teachers: DADH × 6 backbones + UCCH × 6 backbones.

Usage:
    python train_substitute.py --dataset mirflickr25k --bit 64 --device cuda:0 \\
        --teacher_dir ./checkpoints/teachers

Checkpoints saved to:
    checkpoints/substitute/<dataset>_<bit>/ImgNet.pth
    checkpoints/substitute/<dataset>_<bit>/TxtNet.pth
"""

import os
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_data, Dataset
from models import ImgNet, TxtNet
from frequency import dct_2d, get_low_freq_mask, compute_sensitivity, compute_consensus_sensitivity
from losses import (
    distillation_loss, margin_enhanced_loss, standard_hash_loss,
    lt1_triplet, lt2_margin, lt3_sign, lt4_contrastive,
)
from utils import calc_map_k
from teacher_loader import load_teachers, compute_consensus_code, TeacherWeights


# ──────────────────────────────────────────────────────────────────────────────
# Dataset config presets  (paper Table I)
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CFG = {
    'mirflickr25k': dict(
        query_size=2000, db_size=18015, training_size=5000,
        num_label=24, text_dim=1386, image_dim=4096,
    ),
    'nus_wide_tc10': dict(
        query_size=2100, db_size=193734, training_size=10500,
        num_label=21, text_dim=1000, image_dim=4096,
    ),
    'mscoco': dict(
        query_size=2000, db_size=121287, training_size=10000,
        num_label=80, text_dim=1024, image_dim=4096,
    ),
}

# 12 teacher tags used in paper (DADH + UCCH, each × 6 backbones)
TEACHER_TAGS = [
    'DADH_AlexNet', 'DADH_VGG11', 'DADH_RN50', 'DADH_RN152', 'DADH_IncV3', 'DADH_DN161',
    'UCCH_AlexNet', 'UCCH_VGG11', 'UCCH_RN50', 'UCCH_RN152', 'UCCH_IncV3', 'UCCH_DN161',
]


# ──────────────────────────────────────────────────────────────────────────────
# Feature ↔ 2D helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_side(D: int) -> int:
    return math.ceil(math.sqrt(D))


def _feat_to_2d(x: torch.Tensor, side: int) -> torch.Tensor:
    B, D = x.shape
    D_pad = side * side
    if D < D_pad:
        x = torch.cat([x, torch.zeros(B, D_pad - D, device=x.device)], dim=1)
    return x[:, :D_pad].reshape(B, 1, side, side)


def _2d_to_feat(x_2d: torch.Tensor, D: int) -> torch.Tensor:
    return x_2d.flatten(1)[:, :D]


# ──────────────────────────────────────────────────────────────────────────────
# Teacher loading helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_teacher_configs(teacher_dir: str, dataset: str, bit: int,
                            text_dim: int) -> list:
    """
    Build teacher_configs list from teacher_dir.
    Looks for files: <teacher_dir>/<METHOD>/<METHOD>_<BACKBONE>_<BIT>_<DATASET>.pth
    """
    configs = []
    for tag in TEACHER_TAGS:
        method, backbone = tag.split('_', 1)
        fname = f'{method}_{backbone}_{bit}_{dataset}.pth'
        ckpt_path = os.path.join(teacher_dir, method, fname)
        overrides = {'text_dim': text_dim, 'feat_dim': 4096}
        configs.append({'type': method, 'ckpt_path': ckpt_path, 'overrides': overrides})
    return configs


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(img_net, txt_net, images, tags, labels, cfg, device):
    img_net.eval(); txt_net.eval()
    dl_kw = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    q_ds  = Dataset(cfg, images, tags, labels, partition='query')
    db_ds = Dataset(cfg, images, tags, labels, partition='database')
    q_dl  = DataLoader(q_ds,  **dl_kw)
    db_dl = DataLoader(db_ds, **dl_kw)

    qBX, rBX, qBY, rBY = [], [], [], []
    for batch in q_dl:
        img, txt, _ = batch
        qBX.append(img_net.generate_hash(img.to(device).float()).cpu())
        qBY.append(txt_net.generate_hash(txt.to(device).float()).cpu())
    for batch in db_dl:
        img, txt, _ = batch
        rBX.append(img_net.generate_hash(img.to(device).float()).cpu())
        rBY.append(txt_net.generate_hash(txt.to(device).float()).cpu())

    qBX = torch.cat(qBX).to(device);  rBX = torch.cat(rBX).to(device)
    qBY = torch.cat(qBY).to(device);  rBY = torch.cat(rBY).to(device)
    q_lbl  = q_ds.get_labels().to(device)
    db_lbl = db_ds.get_labels().to(device)

    mapi2t = calc_map_k(qBX, rBY, q_lbl, db_lbl).item()
    mapt2i = calc_map_k(qBY, rBX, q_lbl, db_lbl).item()
    img_net.train(); txt_net.train()
    return mapi2t, mapt2i


# ──────────────────────────────────────────────────────────────────────────────
# Phase 0: Pre-train TextNet  (paper Sec. IV-B-3)
# ──────────────────────────────────────────────────────────────────────────────

def pretrain_txtnet(txt_net, img_net_frozen, train_dl, cfg, device,
                    pretrain_epochs: int = 5):
    """
    Pre-train TextNet using cross-modal hash loss (Eq. 2 in paper).
    img_net_frozen provides pseudo image codes as supervision.
    """
    print("\n[Phase 0] Pre-training TextNet ...")
    optimizer = torch.optim.Adam(txt_net.parameters(), lr=cfg.lr)
    txt_net.train()
    img_net_frozen.eval()

    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0
        for idx, imgs, txts, lbls in tqdm(train_dl,
                                           desc=f'  TxtNet pretrain {epoch+1}/{pretrain_epochs}',
                                           leave=False):
            imgs = imgs.to(device).float()
            txts = txts.to(device).float()
            lbls = lbls.to(device).float()

            with torch.no_grad():
                h_img = img_net_frozen(imgs)

            h_txt = txt_net(txts)
            S = (lbls.mm(lbls.t()) > 0).float()
            loss = standard_hash_loss(h_img, h_txt, S)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'  TxtNet pretrain epoch {epoch+1}  loss={epoch_loss:.4f}')

    print("[Phase 0] TextNet pre-training done. Freezing TextNet.\n")
    for p in txt_net.parameters():
        p.requires_grad_(False)
    txt_net.eval()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train FACH substitute model (Phase 1).')
    parser.add_argument('--dataset',    type=str, default='mirflickr25k',
                        choices=list(DATASET_CFG.keys()))
    parser.add_argument('--bit',        type=int, default=64)
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--pretrain_epochs', type=int, default=5,
                        help='Epochs to pre-train TextNet before freezing.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--margin',     type=float, default=1.0,
                        help='Boundary margin m in L_ME (paper: m=1).')
    parser.add_argument('--tau_freq',   type=int, default=20)
    parser.add_argument('--sensitivity_loss', type=str, default='lt2',
                        choices=['lt1', 'lt2', 'lt3', 'lt4'])
    parser.add_argument('--teacher_dir', type=str,
                        default='./checkpoints/teachers')
    parser.add_argument('--save_dir',   type=str,
                        default='./checkpoints/substitute')
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--valid_freq', type=int, default=1)

    cfg = parser.parse_args()
    for k, v in DATASET_CFG[cfg.dataset].items():
        setattr(cfg, k, v)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Training FACH substitute model")
    print(f"  Dataset: {cfg.dataset}  |  Bit: {cfg.bit}")
    print(f"  Train size: {cfg.training_size}  |  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    images, tags, labels = load_data(cfg.dataset, use_vgg_feat=True)
    train_ds = Dataset(cfg, images, tags, labels, partition='train')
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=0, drop_last=False)

    # ── Build models ──────────────────────────────────────────────────────────
    img_net = ImgNet(
        bit=cfg.bit, image_dim=cfg.image_dim, hidden_dim=cfg.hidden_dim,
    ).to(device)
    txt_net = TxtNet(
        bit=cfg.bit, text_dim=cfg.text_dim, hidden_dim=cfg.hidden_dim,
    ).to(device)

    # ── Load teachers ─────────────────────────────────────────────────────────
    teacher_configs = _build_teacher_configs(
        cfg.teacher_dir, cfg.dataset, cfg.bit, cfg.text_dim)
    teachers = load_teachers(teacher_configs, cfg.bit, device)
    have_teachers = len(teachers) > 0
    print(f"\nTeachers loaded: {len(teachers)} / {len(TEACHER_TAGS)}")

    # Learnable teacher weight vector W (Eq. 9)
    teacher_weights = None
    if have_teachers:
        teacher_weights = TeacherWeights(len(teachers)).to(device)

    # ── Phase 0: Pre-train TextNet ────────────────────────────────────────────
    pretrain_txtnet(txt_net, img_net, train_dl, cfg, device,
                    pretrain_epochs=cfg.pretrain_epochs)

    # ── Phase 1: Train ImgNet (+ W) ───────────────────────────────────────────
    params = list(img_net.parameters())
    if teacher_weights is not None:
        params += list(teacher_weights.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    # Frequency setup
    side = _get_side(cfg.image_dim)
    mask = get_low_freq_mask(side, side, cfg.tau_freq, device)

    def _make_loss_fn(variant, lbls=None):
        if variant == 'lt1':
            return lambda h: lt1_triplet(h, lbls)
        elif variant == 'lt2':
            return lambda h: lt2_margin(h, m=cfg.margin)
        elif variant == 'lt3':
            return lambda h: lt3_sign(h)
        elif variant == 'lt4':
            return lambda h: lt4_contrastive(h, lbls)
        raise ValueError(f"Unknown sensitivity_loss: {variant}")

    ckpt_dir = os.path.join(cfg.save_dir, f'{cfg.dataset}_{cfg.bit}')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_avg = 0.0

    for epoch in range(cfg.epochs):
        img_net.train()
        epoch_loss = 0.0

        for idx, imgs, txts, lbls in tqdm(train_dl,
                                           desc=f'Epoch {epoch+1}/{cfg.epochs}'):
            imgs = imgs.to(device).float()
            txts = txts.to(device).float()
            lbls = lbls.to(device).float()

            h_img = img_net(imgs)
            loss_fn = _make_loss_fn(cfg.sensitivity_loss, lbls)

            if have_teachers:
                # ── Distillation with teachers ────────────────────────────────
                imgs_2d = _feat_to_2d(imgs, side)
                with torch.no_grad():
                    F_imgs = dct_2d(imgs_2d)

                def sub_fn(x_sp):
                    return img_net(_2d_to_feat(x_sp, cfg.image_dim))

                A_s = compute_sensitivity(sub_fn, F_imgs, mask, loss_fn)

                def _make_t_fn(teacher):
                    def fn(x_sp):
                        return teacher(_2d_to_feat(x_sp, cfg.image_dim))
                    return fn

                teacher_fns = [_make_t_fn(t) for t in teachers]
                W = teacher_weights() if teacher_weights is not None else None
                A_c = compute_consensus_sensitivity(
                    teacher_fns, F_imgs, mask, loss_fn, weights=W)

                with torch.no_grad():
                    t_code = compute_consensus_code(teachers, imgs)

                loss_img = distillation_loss(A_s, A_c, h_img, t_code, cfg.margin)

            else:
                # ── Self-supervised fallback ───────────────────────────────────
                t_code = h_img.detach().sign()
                loss_img = margin_enhanced_loss(h_img, t_code, cfg.margin)

                # Cross-modal hash loss with frozen TxtNet
                with torch.no_grad():
                    h_txt = txt_net(txts)
                S = (lbls.mm(lbls.t()) > 0).float()
                loss_img = loss_img + standard_hash_loss(h_img, h_txt, S)

            optimizer.zero_grad()
            loss_img.backward()
            optimizer.step()
            epoch_loss += loss_img.item()

        print(f'Epoch {epoch+1:3d}  loss={epoch_loss:.4f}')

        if (epoch + 1) % cfg.valid_freq == 0:
            mapi2t, mapt2i = evaluate(
                img_net, txt_net, images, tags, labels, cfg, device)
            avg = 0.5 * (mapi2t + mapt2i)
            print(f'  MAP i→t={mapi2t:.4f}  t→i={mapt2i:.4f}  avg={avg:.4f}')
            if avg > best_avg:
                best_avg = avg
                torch.save(img_net.state_dict(),
                           os.path.join(ckpt_dir, 'ImgNet.pth'))
                torch.save(txt_net.state_dict(),
                           os.path.join(ckpt_dir, 'TxtNet.pth'))
                if teacher_weights is not None:
                    torch.save(teacher_weights.state_dict(),
                               os.path.join(ckpt_dir, 'TeacherWeights.pth'))
                print(f'  Saved (avg={best_avg:.4f}) → {ckpt_dir}')

    print(f'\nDone. Best avg MAP = {best_avg:.4f}')
    print(f'Weights: {ckpt_dir}')


if __name__ == '__main__':
    main()
