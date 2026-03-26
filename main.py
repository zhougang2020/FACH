"""
FACH — Frequency-domain Adversarial Cross-modal Hashing
========================================================

Two-phase framework:
  Phase 1: Train substitute model H using multi-teacher distillation
           with frequency-domain alignment and hash-boundary enhancement.
  Phase 2: Generate adversarial examples via PGD in the frequency domain,
           guided by the consensus sensitivity map A_c from Phase 1.

Entry points:
  python main.py train  dataset=mirflickr25k bit=64 device=cuda:0
  python main.py attack dataset=mirflickr25k bit=64 device=cuda:0
  python main.py run    dataset=mirflickr25k bit=64 device=cuda:0
"""

import os
import math
import torch
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
    distillation_loss, margin_enhanced_loss,
    lt1_triplet, lt2_margin, lt3_sign, lt4_contrastive,
)
from attack import pgd_attack
from utils import calc_map_k
from teacher_loader import load_teachers, compute_consensus_code


# ──────────────────────────────────────────────────────────────────────────────
# Helper: reshape feature vectors to 2D for DCT
# ──────────────────────────────────────────────────────────────────────────────

def _feat_to_2d(x: torch.Tensor, side: int) -> torch.Tensor:
    """(B, D) → (B, 1, side, side) with zero-padding if needed."""
    B, D = x.shape
    D_pad = side * side
    if D < D_pad:
        x = torch.cat([x, torch.zeros(B, D_pad - D, device=x.device)], dim=1)
    return x[:, :D_pad].reshape(B, 1, side, side)


def _2d_to_feat(x_2d: torch.Tensor, D: int) -> torch.Tensor:
    """(B, 1, side, side) → (B, D)."""
    return x_2d.flatten(1)[:, :D]


def _get_side(D: int) -> int:
    """Smallest integer s such that s² ≥ D."""
    return math.ceil(math.sqrt(D))


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity loss factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_sensitivity_loss(variant: str, labels=None):
    if variant == 'lt1':
        def loss_fn(h): return lt1_triplet(h, labels)
    elif variant == 'lt2':
        def loss_fn(h): return lt2_margin(h, m=opt.margin)
    elif variant == 'lt3':
        def loss_fn(h): return lt3_sign(h)
    elif variant == 'lt4':
        def loss_fn(h): return lt4_contrastive(h, labels)
    else:
        raise ValueError(f"Unknown sensitivity_loss: {variant}")
    return loss_fn


# ──────────────────────────────────────────────────────────────────────────────
# Encoding helpers
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_img(model, loader, device) -> torch.Tensor:
    codes = []
    model.eval()
    for batch in loader:
        img = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        codes.append(model.generate_hash(img).cpu())
    model.train()
    return torch.cat(codes, 0)


@torch.no_grad()
def _encode_txt(model, loader, device) -> torch.Tensor:
    codes = []
    model.eval()
    for batch in loader:
        txt = batch[1].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        codes.append(model.generate_hash(txt).cpu())
    model.train()
    return torch.cat(codes, 0)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(img_net, txt_net, images, tags, labels, device):
    dl_kw = dict(batch_size=opt.batch_size, shuffle=False, num_workers=0)

    q_ds  = Dataset(opt, images, tags, labels, partition='query')
    db_ds = Dataset(opt, images, tags, labels, partition='database')

    q_dl  = DataLoader(q_ds,  **dl_kw)
    db_dl = DataLoader(db_ds, **dl_kw)

    qBX = _encode_img(img_net, q_dl,  device).to(device)
    rBX = _encode_img(img_net, db_dl, device).to(device)
    qBY = _encode_txt(txt_net, q_dl,  device).to(device)
    rBY = _encode_txt(txt_net, db_dl, device).to(device)

    q_lbl  = q_ds.get_labels().to(device)
    db_lbl = db_ds.get_labels().to(device)

    mapi2t = calc_map_k(qBX, rBY, q_lbl, db_lbl).item()
    mapt2i = calc_map_k(qBY, rBX, q_lbl, db_lbl).item()
    return mapi2t, mapt2i


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Train substitute model
# ──────────────────────────────────────────────────────────────────────────────

def train(**kwargs):
    """
    Phase 1: Train the substitute model with multi-teacher distillation.

    Checkpoints are saved to  <save_path>/<dataset>_<bit>/
    """
    opt.parse(kwargs)
    device = opt.device

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data ...")
    images, tags, labels = load_data(opt.dataset, use_vgg_feat=opt.use_vgg_feat)

    train_ds = Dataset(opt, images, tags, labels, partition='train')
    train_dl = DataLoader(train_ds, batch_size=opt.batch_size,
                          shuffle=True, num_workers=0, drop_last=False)

    # ── Models ────────────────────────────────────────────────────────────────
    img_net = ImgNet(
        bit=opt.bit, image_dim=opt.image_dim, hidden_dim=opt.hidden_dim,
        use_backbone=opt.use_backbone, dropout=opt.dropout,
    ).to(device)

    txt_net = TxtNet(
        bit=opt.bit, text_dim=opt.text_dim, hidden_dim=opt.hidden_dim,
        dropout=opt.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(img_net.parameters()) + list(txt_net.parameters()), lr=opt.lr,
    )

    # ── Teacher models ────────────────────────────────────────────────────────
    teachers = load_teachers(opt.teacher_configs, opt.bit, device)
    have_teachers = len(teachers) > 0

    # ── Frequency setup (feature mode) ───────────────────────────────────────
    side = _get_side(opt.image_dim)
    mask = get_low_freq_mask(side, side, opt.tau_freq, device)   # (side, side)

    # ── Training ──────────────────────────────────────────────────────────────
    ckpt_dir = os.path.join(opt.save_path, f'{opt.dataset}_{opt.bit}')
    os.makedirs(ckpt_dir, exist_ok=True)
    best_avg = 0.0

    for epoch in range(opt.max_epoch):
        img_net.train(); txt_net.train()
        epoch_loss = 0.0

        for idx, imgs, txts, lbls in tqdm(train_dl,
                                           desc=f'Epoch {epoch+1}/{opt.max_epoch}'):
            imgs = imgs.to(device)   # (B, 4096) or (B, 3, 224, 224)
            txts = txts.to(device)
            lbls = lbls.to(device)

            h_img = img_net(imgs)    # (B, K) real-valued
            h_txt = txt_net(txts)

            loss_fn = _make_sensitivity_loss(opt.sensitivity_loss, lbls)

            if have_teachers:
                # ── Full distillation with teachers ──────────────────────────
                if opt.use_vgg_feat:
                    # Feature mode: reshape to 2D for DCT
                    imgs_2d = _feat_to_2d(imgs, side)           # (B, 1, s, s)
                else:
                    imgs_2d = imgs                              # (B, 3, 224, 224)

                with torch.no_grad():
                    F_imgs = dct_2d(imgs_2d.float())

                # Substitute sensitivity
                def sub_fn(x_sp):
                    x_f = (_2d_to_feat(x_sp, opt.image_dim)
                           if opt.use_vgg_feat else x_sp)
                    return img_net(x_f)

                A_s = compute_sensitivity(sub_fn, F_imgs, mask, loss_fn)

                # Teacher sensitivity functions
                def _make_t_fn(teacher):
                    def fn(x_sp):
                        x_f = (_2d_to_feat(x_sp, opt.image_dim)
                               if opt.use_vgg_feat else x_sp)
                        return teacher(x_f)
                    return fn

                teacher_fns = [_make_t_fn(t) for t in teachers]
                A_c = compute_consensus_sensitivity(teacher_fns, F_imgs, mask, loss_fn)

                # Consensus target codes for L_ME
                with torch.no_grad():
                    t_code = compute_consensus_code(teachers, imgs)

                loss_img = distillation_loss(A_s, A_c, h_img, t_code, opt.margin)
                loss_txt = margin_enhanced_loss(h_txt, t_code, opt.margin)
                total_loss = loss_img + loss_txt

            else:
                # ── Self-training (no teachers) ───────────────────────────────
                t_code = h_img.detach().sign()

                loss_img = margin_enhanced_loss(h_img, t_code, opt.margin)
                loss_txt = margin_enhanced_loss(h_txt, t_code, opt.margin)

                # Cross-modal pairwise log-loss for basic hash learning
                inner   = 0.5 * h_img.mm(h_txt.t())
                S       = (lbls.mm(lbls.t()) > 0).float()
                hash_loss = -(S * inner - torch.log(1.0 + torch.exp(inner))).mean()

                total_loss = loss_img + loss_txt + hash_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

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
                print(f'  ✓ Saved  (avg={best_avg:.4f})  → {ckpt_dir}')

    print(f'\nTraining complete. Best avg MAP = {best_avg:.4f}')
    print(f'Weights saved to: {ckpt_dir}')


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Adversarial attack
# ──────────────────────────────────────────────────────────────────────────────

def attack(**kwargs):
    """
    Phase 2: PGD attack in frequency domain; report mAP drop on each victim.
    """
    opt.parse(kwargs)
    device = opt.device

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data ...")
    images, tags, labels = load_data(opt.dataset, use_vgg_feat=opt.use_vgg_feat)

    q_ds  = Dataset(opt, images, tags, labels, partition='query')
    db_ds = Dataset(opt, images, tags, labels, partition='database')

    dl_kw = dict(batch_size=opt.batch_size, shuffle=False, num_workers=0)
    q_dl  = DataLoader(q_ds,  **dl_kw)
    db_dl = DataLoader(db_ds, **dl_kw)

    q_lbl  = q_ds.get_labels().to(device)
    db_lbl = db_ds.get_labels().to(device)

    # ── Load substitute model ─────────────────────────────────────────────────
    ckpt_dir = os.path.join(opt.save_path, f'{opt.dataset}_{opt.bit}')
    img_net = ImgNet(
        bit=opt.bit, image_dim=opt.image_dim, hidden_dim=opt.hidden_dim,
    ).to(device)
    img_net.load_state_dict(
        torch.load(os.path.join(ckpt_dir, 'ImgNet.pth'), map_location=device))
    img_net.eval()

    # ── Frequency setup ───────────────────────────────────────────────────────
    side = _get_side(opt.image_dim)
    mask = get_low_freq_mask(side, side, opt.tau_freq, device)
    loss_fn = _make_sensitivity_loss(opt.sensitivity_loss)

    def get_A_c(x_feat):
        """Consensus sensitivity from substitute model."""
        x_2d = _feat_to_2d(x_feat, side)
        F = dct_2d(x_2d.float().detach())

        def sub_fn(x_sp):
            return img_net(_2d_to_feat(x_sp, opt.image_dim))

        A = compute_sensitivity(sub_fn, F, mask, loss_fn)   # (B, 1, s, s)
        return A

    # ── Victim model loop ─────────────────────────────────────────────────────
    victims = load_teachers(opt.teacher_configs, opt.bit, device)
    if not victims:
        print("[attack] No teacher_configs — using substitute model as victim.")
        victims = [img_net]

    for v_idx, victim in enumerate(victims):
        victim.eval()
        print(f"\n=== Victim {v_idx+1}/{len(victims)} ===")

        # --- Clean hashes ---
        with torch.no_grad():
            qBX_clean, rBX = [], []
            for batch in q_dl:
                img = batch[0].to(device)
                qBX_clean.append(victim(img).sign().cpu())
            for batch in db_dl:
                img = batch[0].to(device)
                rBX.append(victim(img).sign().cpu())
        qBX_clean = torch.cat(qBX_clean).to(device)
        rBX       = torch.cat(rBX).to(device)

        map_clean = calc_map_k(qBX_clean, rBX, q_lbl, db_lbl).item()
        print(f'  Clean mAP = {map_clean:.4f}')

        # --- Adversarial queries ---
        qBX_adv = []
        for batch in q_dl:
            img = batch[0].to(device)

            with torch.no_grad():
                b_orig = victim(img).sign()

            A_c_2d = get_A_c(img)                      # (B, 1, s, s)
            img_2d = _feat_to_2d(img, side)            # (B, 1, s, s)

            x_adv_2d = pgd_attack(
                victim_fn=lambda z: victim(_2d_to_feat(z, opt.image_dim)),
                x=img_2d.float(),
                b_orig=b_orig,
                A_c=A_c_2d.float(),
                opt=opt,
                b_target=None,
            )
            x_adv = _2d_to_feat(x_adv_2d, opt.image_dim)

            with torch.no_grad():
                qBX_adv.append(victim(x_adv).sign().cpu())

        qBX_adv = torch.cat(qBX_adv).to(device)
        map_adv = calc_map_k(qBX_adv, rBX, q_lbl, db_lbl).item()
        print(f'  Adv   mAP = {map_adv:.4f}  (drop = {map_clean - map_adv:.4f})')


# ──────────────────────────────────────────────────────────────────────────────

def run(**kwargs):
    """Run Phase 1 (train) then Phase 2 (attack)."""
    train(**kwargs)
    attack(**kwargs)


if __name__ == '__main__':
    import fire
    fire.Fire({'train': train, 'attack': attack, 'run': run})
