"""
Train the FACH substitute model using pre-trained teacher models.

The substitute model (ImgNet + TxtNet) is trained with:
  Phase 1 loss = L_align  (JSD between frequency sensitivities, Eq. 13)
               + L_ME     (margin-enhanced hash-boundary loss, Eq. 14)

When no teacher checkpoints are given, the model falls back to
self-supervised training with cross-modal hash loss + L_ME.

Usage:
    # With teachers:
    python train_substitute.py \\
        --dataset mirflickr25k --bit 64 --device cuda:0 \\
        --teachers DADH_VGG11,DCMH_RN50,DGCPN_RN152,UCCH_DN161 \\
        --teacher_dir ./checkpoints/teachers

    # Without teachers (self-supervised):
    python train_substitute.py --dataset mirflickr25k --bit 64

Checkpoints saved to:
    checkpoints/substitute/<dataset>_<bit>/ImgNet.pth
    checkpoints/substitute/<dataset>_<bit>/TxtNet.pth
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_data, Dataset
from models import ImgNet, TxtNet
from methods.base import HashNet
from frequency import dct_2d, get_low_freq_mask, compute_sensitivity, compute_consensus_sensitivity
from losses import distillation_loss, margin_enhanced_loss, lt2_margin
from utils import calc_map_k
from backbones import BACKBONE_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Dataset config presets
# ──────────────────────────────────────────────────────────────────────────────

DATASET_CFG = {
    'mirflickr25k': dict(
        query_size=2000, db_size=18015, training_size=10000,
        num_label=24, text_dim=1386, image_dim=4096,
    ),
    'nus_wide_tc10': dict(
        query_size=2100, db_size=184477, training_size=10500,
        num_label=10, text_dim=1000, image_dim=4096,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature ↔ 2D helpers for DCT
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
# Teacher loading
# ──────────────────────────────────────────────────────────────────────────────

def load_teacher(tag: str, bit: int, text_dim: int, hidden_dim: int,
                 teacher_dir: str, dataset: str, device) -> HashNet:
    """
    Load a pre-trained HashNet teacher from checkpoint.

    tag format: '<METHOD>_<BACKBONE>'  e.g. 'DADH_VGG11', 'DCMH_RN50'
    Checkpoint path: <teacher_dir>/<METHOD>/<METHOD>_<BACKBONE>_<BIT>.pth
    """
    parts = tag.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid teacher tag '{tag}'. Expected '<METHOD>_<BACKBONE>'.")
    method, backbone = parts
    if backbone not in BACKBONE_NAMES:
        raise ValueError(f"Unknown backbone '{backbone}' in tag '{tag}'.")

    ckpt_path = os.path.join(teacher_dir, method,
                             f'{method}_{backbone}_{bit}_{dataset}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt_path}\n"
            f"Train it first with:\n"
            f"  python train_teacher.py --method {method} --backbone {backbone} "
            f"--bit {bit} --dataset {dataset}")

    teacher = HashNet(backbone_name=backbone, bit=bit,
                      text_dim=text_dim, hidden_dim=hidden_dim,
                      pretrained=False).to(device)
    teacher.load(ckpt_path, device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f'  [teacher] Loaded {tag}  from  {ckpt_path}')
    return teacher


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
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train FACH substitute model (Phase 1).')
    parser.add_argument('--dataset',    type=str, default='mirflickr25k',
                        choices=list(DATASET_CFG.keys()))
    parser.add_argument('--bit',        type=int, default=64)
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=4096)
    parser.add_argument('--margin',     type=float, default=1.5,
                        help='Boundary margin m in L_ME.')
    parser.add_argument('--tau_freq',   type=int, default=20,
                        help='Low-frequency threshold τ for DCT mask.')
    parser.add_argument('--teachers',   type=str, default='',
                        help='Comma-separated teacher tags: DADH_VGG11,DCMH_RN50,…')
    parser.add_argument('--teacher_dir', type=str,
                        default='./checkpoints/teachers',
                        help='Directory containing teacher checkpoints.')
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
    print(f"  Device:  {device}")
    print(f"{'='*60}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data ...")
    images, tags, labels = load_data(cfg.dataset, use_vgg_feat=True)
    train_ds = Dataset(cfg, images, tags, labels, partition='train')
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size,
                          shuffle=True, num_workers=0, drop_last=False)

    # ── Substitute model ──────────────────────────────────────────────────────
    img_net = ImgNet(
        bit=cfg.bit, image_dim=cfg.image_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    txt_net = TxtNet(
        bit=cfg.bit, text_dim=cfg.text_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(img_net.parameters()) + list(txt_net.parameters()), lr=cfg.lr)

    # ── Teacher models ────────────────────────────────────────────────────────
    teacher_tags = [t.strip() for t in cfg.teachers.split(',') if t.strip()]
    teachers = []
    for tag in teacher_tags:
        t = load_teacher(tag, cfg.bit, cfg.text_dim, cfg.hidden_dim,
                         cfg.teacher_dir, cfg.dataset, device)
        teachers.append(t)
    have_teachers = len(teachers) > 0
    print(f"\nTeachers loaded: {len(teachers)}")

    # ── Frequency setup ───────────────────────────────────────────────────────
    side = _get_side(cfg.image_dim)
    mask = get_low_freq_mask(side, side, cfg.tau_freq, device)

    loss_fn_global = lambda h: lt2_margin(h, m=cfg.margin)

    # ── Checkpoint path ───────────────────────────────────────────────────────
    ckpt_dir = os.path.join(cfg.save_dir, f'{cfg.dataset}_{cfg.bit}')
    os.makedirs(ckpt_dir, exist_ok=True)

    best_avg = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    for epoch in range(cfg.epochs):
        img_net.train(); txt_net.train()
        epoch_loss = 0.0

        for idx, imgs, txts, lbls in tqdm(train_dl,
                                           desc=f'Epoch {epoch+1}/{cfg.epochs}'):
            imgs = imgs.to(device).float()
            txts = txts.to(device).float()
            lbls = lbls.to(device).float()

            h_img = img_net(imgs)
            h_txt = txt_net(txts)

            if have_teachers:
                # ── Distillation with teachers ────────────────────────────────
                imgs_2d = _feat_to_2d(imgs, side)
                with torch.no_grad():
                    F_imgs = dct_2d(imgs_2d)

                # Substitute sensitivity A_s
                def sub_fn(x_sp):
                    return img_net(_2d_to_feat(x_sp, cfg.image_dim))

                A_s = compute_sensitivity(sub_fn, F_imgs, mask, loss_fn_global)

                # Teacher sensitivity functions
                def _make_t_fn(teacher):
                    def fn(x_sp):
                        return teacher.forward_img(_2d_to_feat(x_sp, cfg.image_dim))
                    return fn

                teacher_fns = [_make_t_fn(t) for t in teachers]
                A_c = compute_consensus_sensitivity(
                    teacher_fns, F_imgs, mask, loss_fn_global)

                # Consensus target codes t for L_ME  (Eq. 14)
                with torch.no_grad():
                    h_stack = torch.stack(
                        [t.forward_img(imgs) for t in teachers], dim=0)
                    t_code  = h_stack.mean(dim=0).sign()   # (B, K)

                loss_img = distillation_loss(A_s, A_c, h_img, t_code, cfg.margin)
                loss_txt = margin_enhanced_loss(h_txt, t_code, cfg.margin)
                total_loss = loss_img + loss_txt

            else:
                # ── Self-supervised fallback ───────────────────────────────────
                t_code = h_img.detach().sign()
                loss_img = margin_enhanced_loss(h_img, t_code, cfg.margin)
                loss_txt = margin_enhanced_loss(h_txt, t_code, cfg.margin)

                # Cross-modal pairwise hash loss
                inner = 0.5 * h_img.mm(h_txt.t())
                S     = (lbls.mm(lbls.t()) > 0).float()
                hash_loss = -(S * inner - torch.log(1 + torch.exp(inner))).mean()
                total_loss = loss_img + loss_txt + hash_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

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
                print(f'  ✓ Saved  (avg={best_avg:.4f})  → {ckpt_dir}')

    print(f'\nDone. Best avg MAP = {best_avg:.4f}')
    print(f'Weights: {ckpt_dir}')


if __name__ == '__main__':
    main()
