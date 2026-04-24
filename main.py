"""
FACH — Frequency-domain Adversarial Cross-modal Hashing
========================================================

Two-phase framework:
  Phase 0: Pre-train TextNet with cross-modal hash loss, then freeze.
  Phase 1: Train ImgNet + learnable teacher weights W via multi-teacher
           distillation (L_distill = L_align + L_ME, Eqs. 10-12).
  Phase 2: PGD attack in frequency domain guided by consensus A_c,
           using global semantic hash code as targeted anchor (Eq. 16).

Entry points:
  python main.py train  dataset=mirflickr25k bit=64 device=cuda:0
  python main.py attack dataset=mirflickr25k bit=64 device=cuda:0
  python main.py run    dataset=mirflickr25k bit=64 device=cuda:0
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import opt
from models import ImgNet, TxtNet
from datasets import load_data, Dataset
from frequency import (
    dct_2d, get_low_freq_mask,
    compute_sensitivity, compute_consensus_sensitivity,
)
from losses import (
    distillation_loss, margin_enhanced_loss, standard_hash_loss,
    lt1_triplet, lt2_margin, lt3_sign, lt4_contrastive,
)
from attack import pgd_attack, compute_global_semantic_hash
from utils import calc_map_k
from teacher_loader import (
    load_teachers, compute_consensus_code, TeacherWeights,
)


# ──────────────────────────────────────────────────────────────────────────────
# Feature ↔ 2D helpers
# ──────────────────────────────────────────────────────────────────────────────

def _feat_to_2d(x: torch.Tensor, side: int) -> torch.Tensor:
    B, D = x.shape
    D_pad = side * side
    if D < D_pad:
        x = torch.cat([x, torch.zeros(B, D_pad - D, device=x.device)], dim=1)
    return x[:, :D_pad].reshape(B, 1, side, side)


def _2d_to_feat(x_2d: torch.Tensor, D: int) -> torch.Tensor:
    return x_2d.flatten(1)[:, :D]


def _get_side(D: int) -> int:
    return math.ceil(math.sqrt(D))


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity loss factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_loss_fn(variant: str, labels=None):
    if variant == 'lt1':
        return lambda h: lt1_triplet(h, labels)
    elif variant == 'lt2':
        return lambda h: lt2_margin(h, m=opt.margin)
    elif variant == 'lt3':
        return lambda h: lt3_sign(h)
    elif variant == 'lt4':
        return lambda h: lt4_contrastive(h, labels)
    raise ValueError(f"Unknown sensitivity_loss: {variant}")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(img_net, txt_net, images, tags, labels, device):
    img_net.eval(); txt_net.eval()
    dl_kw = dict(batch_size=opt.batch_size, shuffle=False, num_workers=0)

    q_ds  = Dataset(opt, images, tags, labels, partition='query')
    db_ds = Dataset(opt, images, tags, labels, partition='database')
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

    qBX = torch.cat(qBX).to(device); rBX = torch.cat(rBX).to(device)
    qBY = torch.cat(qBY).to(device); rBY = torch.cat(rBY).to(device)
    q_lbl  = q_ds.get_labels().to(device)
    db_lbl = db_ds.get_labels().to(device)

    mapi2t = calc_map_k(qBX, rBY, q_lbl, db_lbl).item()
    mapt2i = calc_map_k(qBY, rBX, q_lbl, db_lbl).item()
    img_net.train(); txt_net.train()
    return mapi2t, mapt2i


# ──────────────────────────────────────────────────────────────────────────────
# Phase 0: Pre-train TextNet  (paper Sec. IV-B-3)
# ──────────────────────────────────────────────────────────────────────────────

def _pretrain_txtnet(txt_net, img_net, train_dl, device):
    print(f"\n[Phase 0] Pre-training TextNet for {opt.pretrain_epochs} epochs ...")
    optimizer = torch.optim.Adam(txt_net.parameters(), lr=opt.lr)
    txt_net.train(); img_net.eval()

    for epoch in range(opt.pretrain_epochs):
        epoch_loss = 0.0
        for idx, imgs, txts, lbls in tqdm(train_dl,
                                           desc=f'  TxtNet {epoch+1}/{opt.pretrain_epochs}',
                                           leave=False):
            imgs = imgs.to(device).float()
            txts = txts.to(device).float()
            lbls = lbls.to(device).float()

            with torch.no_grad():
                h_img = img_net(imgs)

            h_txt = txt_net(txts)
            S = (lbls.mm(lbls.t()) > 0).float()
            loss = standard_hash_loss(h_img, h_txt, S)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item()

        print(f'  TxtNet epoch {epoch+1}  loss={epoch_loss:.4f}')

    print("[Phase 0] Done. Freezing TextNet.\n")
    for p in txt_net.parameters():
        p.requires_grad_(False)
    txt_net.eval()


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Train substitute model
# ──────────────────────────────────────────────────────────────────────────────

def train(**kwargs):
    opt.parse(kwargs)
    device = opt.device

    print("Loading data ...")
    images, tags, labels = load_data(opt.dataset, use_vgg_feat=True)

    train_ds = Dataset(opt, images, tags, labels, partition='train')
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size,
                          shuffle=True, num_workers=0, drop_last=False)

    img_net = ImgNet(bit=opt.bit, image_dim=opt.image_dim,
                     hidden_dim=opt.hidden_dim).to(device)
    txt_net = TxtNet(bit=opt.bit, text_dim=opt.text_dim,
                     hidden_dim=opt.hidden_dim).to(device)

    # ── Load teachers ─────────────────────────────────────────────────────────
    teachers = load_teachers(opt.teacher_configs, opt.bit, device)
    have_teachers = len(teachers) > 0
    print(f"Teachers loaded: {len(teachers)}")

    teacher_weights = TeacherWeights(len(teachers)).to(device) if have_teachers else None

    # ── Phase 0: pre-train TxtNet ─────────────────────────────────────────────
    _pretrain_txtnet(txt_net, img_net, train_dl, device)

    # ── Phase 1: train ImgNet (+ W) ───────────────────────────────────────────
    params = list(img_net.parameters())
    if teacher_weights is not None:
        params += list(teacher_weights.parameters())
    optimizer = torch.optim.Adam(params, lr=opt.lr)

    side = _get_side(opt.image_dim)
    mask = get_low_freq_mask(side, side, opt.tau_freq, device)

    ckpt_dir = os.path.join(opt.save_path, f'{opt.dataset}_{opt.bit}')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_avg = 0.0

    for epoch in range(opt.max_epoch):
        img_net.train()
        epoch_loss = 0.0

        for idx, imgs, txts, lbls in tqdm(train_dl,
                                           desc=f'Epoch {epoch+1}/{opt.max_epoch}'):
            imgs = imgs.to(device).float()
            txts = txts.to(device).float()
            lbls = lbls.to(device).float()

            h_img = img_net(imgs)
            loss_fn = _make_loss_fn(opt.sensitivity_loss, lbls)

            if have_teachers:
                imgs_2d = _feat_to_2d(imgs, side)
                with torch.no_grad():
                    F_imgs = dct_2d(imgs_2d)

                def sub_fn(x_sp):
                    return img_net(_2d_to_feat(x_sp, opt.image_dim))

                A_s = compute_sensitivity(sub_fn, F_imgs, mask, loss_fn)

                def _make_t_fn(t):
                    def fn(x_sp):
                        return t(_2d_to_feat(x_sp, opt.image_dim))
                    return fn

                teacher_fns = [_make_t_fn(t) for t in teachers]
                W = teacher_weights() if teacher_weights is not None else None
                A_c = compute_consensus_sensitivity(
                    teacher_fns, F_imgs, mask, loss_fn, weights=W)

                with torch.no_grad():
                    t_code = compute_consensus_code(teachers, imgs)

                loss_img = distillation_loss(A_s, A_c, h_img, t_code, opt.margin)

            else:
                # No teachers: self-supervised fallback
                t_code = h_img.detach().sign()
                loss_img = margin_enhanced_loss(h_img, t_code, opt.margin)

                with torch.no_grad():
                    h_txt = txt_net(txts)
                S = (lbls.mm(lbls.t()) > 0).float()
                loss_img = loss_img + standard_hash_loss(h_img, h_txt, S)

            optimizer.zero_grad()
            loss_img.backward()
            optimizer.step()
            epoch_loss += loss_img.item()

        print(f'Epoch {epoch+1:3d}  loss={epoch_loss:.4f}')

        if (epoch + 1) % opt.valid_freq == 0:
            mapi2t, mapt2i = evaluate(img_net, txt_net, images, tags, labels, device)
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


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Adversarial attack
# ──────────────────────────────────────────────────────────────────────────────

def attack(**kwargs):
    opt.parse(kwargs)
    device = opt.device

    print("Loading data ...")
    images, tags, labels = load_data(opt.dataset, use_vgg_feat=True)

    q_ds  = Dataset(opt, images, tags, labels, partition='query')
    db_ds = Dataset(opt, images, tags, labels, partition='database')
    dl_kw = dict(batch_size=opt.batch_size, shuffle=False, num_workers=0)
    q_dl  = DataLoader(q_ds,  **dl_kw)
    db_dl = DataLoader(db_ds, **dl_kw)

    q_lbl  = q_ds.get_labels().to(device)
    db_lbl = db_ds.get_labels().to(device)

    # ── Load substitute model ─────────────────────────────────────────────────
    ckpt_dir = os.path.join(opt.save_path, f'{opt.dataset}_{opt.bit}')
    img_net = ImgNet(bit=opt.bit, image_dim=opt.image_dim,
                     hidden_dim=opt.hidden_dim).to(device)
    img_net.load_state_dict(
        torch.load(os.path.join(ckpt_dir, 'ImgNet.pth'), map_location=device))
    img_net.eval()

    # ── Frequency setup ───────────────────────────────────────────────────────
    side = _get_side(opt.image_dim)
    mask = get_low_freq_mask(side, side, opt.tau_freq, device)
    loss_fn = _make_loss_fn(opt.sensitivity_loss)

    def _get_A_c(x_feat):
        x_2d = _feat_to_2d(x_feat, side)
        F = dct_2d(x_2d.detach())

        def sub_fn(x_sp):
            return img_net(_2d_to_feat(x_sp, opt.image_dim))

        return compute_sensitivity(sub_fn, F, mask, loss_fn)

    # ── Pre-compute database codes from substitute (for global semantic hash) ─
    print("Computing database codes from substitute model ...")
    rBX_sub = []
    with torch.no_grad():
        for batch in db_dl:
            img = batch[0].to(device).float()
            rBX_sub.append(img_net(img).sign().cpu())
    rBX_sub = torch.cat(rBX_sub).to(device)

    # ── Victim model loop ─────────────────────────────────────────────────────
    victims = load_teachers(opt.teacher_configs, opt.bit, device)
    if not victims:
        print("[attack] No victims configured — using substitute as victim.")
        victims = [img_net]

    for v_idx, victim in enumerate(victims):
        victim.eval()
        print(f"\n=== Victim {v_idx+1}/{len(victims)} ===")

        # Clean database codes from victim
        rBX_v = []
        with torch.no_grad():
            for batch in db_dl:
                img = batch[0].to(device).float()
                rBX_v.append(victim(img).sign().cpu())
        rBX_v = torch.cat(rBX_v).to(device)

        # Clean query codes
        qBX_clean = []
        with torch.no_grad():
            for batch in q_dl:
                img = batch[0].to(device).float()
                qBX_clean.append(victim(img).sign().cpu())
        qBX_clean = torch.cat(qBX_clean).to(device)

        map_clean = calc_map_k(qBX_clean, rBX_v, q_lbl, db_lbl).item()
        print(f'  Clean mAP = {map_clean:.4f}')

        # Adversarial queries
        qBX_adv = []
        for batch in q_dl:
            img = batch[0].to(device).float()
            lbl = batch[2].to(device).float()

            with torch.no_grad():
                b_orig = victim(img).sign()

            # Global semantic hash target (Eq. 16)
            b_target = compute_global_semantic_hash(lbl, rBX_sub, db_lbl)

            A_c = _get_A_c(img)
            img_2d = _feat_to_2d(img, side)

            x_adv_2d = pgd_attack(
                victim_fn=lambda z: victim(_2d_to_feat(z, opt.image_dim)),
                x=img_2d,
                b_orig=b_orig,
                A_c=A_c,
                opt=opt,
                b_target=b_target,
            )
            x_adv = _2d_to_feat(x_adv_2d, opt.image_dim)

            with torch.no_grad():
                qBX_adv.append(victim(x_adv).sign().cpu())

        qBX_adv = torch.cat(qBX_adv).to(device)
        map_adv = calc_map_k(qBX_adv, rBX_v, q_lbl, db_lbl).item()
        print(f'  Adv   mAP = {map_adv:.4f}  (drop = {map_clean - map_adv:.4f})')


# ──────────────────────────────────────────────────────────────────────────────

def run(**kwargs):
    train(**kwargs)
    attack(**kwargs)


if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'attack': attack, 'run': run})
