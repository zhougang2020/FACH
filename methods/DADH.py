"""
DADH — Deep Adversarial Discrete Hashing
=========================================
Reference: Cao et al., "Deep Adversarial Discrete Hashing for Cross-Modal Retrieval"

Loss components:
  1. Triplet loss  (cross-modal, Eq. in DADH paper)
  2. Quantization loss  ||B - h||²
  3. Adversarial loss   (GAN: feature + hash discriminators)

Architecture follows the original paper:
  Image : backbone → feature → hash  (via HashNet)
  Text  : MLP → feature → hash       (via HashNet)
  Discriminator: two sub-nets for feature-level and hash-level alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as autograd_grad


# ──────────────────────────────────────────────────────────────────────────────
# Discriminator (feature + hash level)
# ──────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int, bit: int):
        super().__init__()
        self.feature_dis = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.hash_dis = nn.Sequential(
            nn.Linear(bit, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def dis_feature(self, f: torch.Tensor) -> torch.Tensor:
        return self.feature_dis(f)

    def dis_hash(self, h: torch.Tensor) -> torch.Tensor:
        return self.hash_dis(h)


# ──────────────────────────────────────────────────────────────────────────────
# Triplet loss (cross-modal)
# ──────────────────────────────────────────────────────────────────────────────

def _triplet_loss(h_anchor: torch.Tensor,
                  h_pos:    torch.Tensor,
                  labels:   torch.Tensor,
                  margin:   float = 0.4) -> torch.Tensor:
    """
    Online hard-negative triplet loss in hash space.
    """
    S = (labels.mm(labels.t()) > 0).float()   # (B, B) similarity

    dist = torch.cdist(h_anchor, h_pos, p=2)  # (B, B) L2 distance

    pos_dist = (S * dist).max(dim=1)[0]
    neg_dist = ((1 - S) * dist + S * 1e9).min(dim=1)[0]

    loss = F.relu(pos_dist - neg_dist + margin).mean()
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Gradient penalty (WGAN-GP style)
# ──────────────────────────────────────────────────────────────────────────────

def _gradient_penalty(dis_fn, real: torch.Tensor, fake: torch.Tensor,
                       device, gp_weight: float = 10.0) -> torch.Tensor:
    B = real.shape[0]
    alpha = torch.rand(B, 1, device=device)
    interp = (alpha * real.detach() + (1 - alpha) * fake.detach()).requires_grad_(True)
    d_interp = dis_fn(interp)
    grads = autograd_grad(d_interp, interp,
                          grad_outputs=torch.ones_like(d_interp),
                          create_graph=True, retain_graph=True)[0]
    gp = ((grads.view(B, -1).norm(2, dim=1) - 1) ** 2).mean() * gp_weight
    return gp


# ──────────────────────────────────────────────────────────────────────────────
# Per-epoch training step
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, discriminator, loader, opt_G, opt_D_feat, opt_D_hash,
                B_buf, H_i_buf, H_t_buf, epoch, cfg, device):
    """
    One training epoch for DADH.

    Args:
        model        : HashNet instance.
        discriminator: Discriminator instance.
        loader       : training DataLoader yielding (idx, img, txt, label).
        opt_G        : Adam optimizer for model (generator).
        opt_D_feat   : Adam optimizer for feature discriminator.
        opt_D_hash   : Adam optimizer for hash discriminator.
        B_buf        : (N, K) binary code buffer.
        H_i_buf, H_t_buf : (N, K) real-valued hash buffers.
        epoch        : current epoch index (0-based).
        cfg          : config object (alpha, beta, gamma, margin, lamb, num_label).
        device       : torch.device.

    Returns:
        epoch_loss (float)
    """
    model.train(); discriminator.train()
    epoch_loss = 0.0

    for idx, imgs, txts, labels in loader:
        imgs   = imgs.to(device).float()
        txts   = txts.to(device).float()
        labels = labels.to(device).float()
        B      = imgs.shape[0]

        # ── Generator forward ─────────────────────────────────────────────────
        h_i = model.forward_img(imgs)           # (B, K)
        h_t = model.forward_txt(txts)           # (B, K)
        f_i = model.get_img_feat(imgs)          # (B, feat_dim)
        # Use image features as proxy text features for adversarial alignment
        # (text "feature" is the first hidden layer output)
        f_t = model.txt_net.net[:3](txts)       # up to first ReLU → (B, hidden//2)

        # Ensure f_i and f_t have the same dim for the discriminator
        # Project f_i to same dim as hidden if needed
        if not hasattr(train_epoch, '_proj') or train_epoch._proj is None:
            if f_i.shape[1] != f_t.shape[1]:
                proj = nn.Linear(f_i.shape[1], f_t.shape[1], bias=False).to(device)
                nn.init.eye_(proj.weight[:min(f_i.shape[1], f_t.shape[1]),
                                        :min(f_i.shape[1], f_t.shape[1])])
                train_epoch._proj = proj
            else:
                train_epoch._proj = nn.Identity().to(device)
        f_i_d = train_epoch._proj(f_i)

        H_i_buf[idx] = h_i.detach()
        H_t_buf[idx] = h_t.detach()

        # ── Feature discriminator update ──────────────────────────────────────
        opt_D_feat.zero_grad()
        d_real = -(cfg.gamma * torch.log(torch.sigmoid(discriminator.dis_feature(f_i_d.detach())) + 1e-8)).mean()
        d_fake = -(cfg.gamma * torch.log(1 - torch.sigmoid(discriminator.dis_feature(f_t.detach())) + 1e-8)).mean()
        gp_f = _gradient_penalty(discriminator.dis_feature, f_i_d, f_t, device)
        (d_real + d_fake + gp_f).backward()
        opt_D_feat.step()

        # ── Hash discriminator update ─────────────────────────────────────────
        opt_D_hash.zero_grad()
        d_real_h = -(cfg.gamma * torch.log(torch.sigmoid(discriminator.dis_hash(h_i.detach())) + 1e-8)).mean()
        d_fake_h = -(cfg.gamma * torch.log(1 - torch.sigmoid(discriminator.dis_hash(h_t.detach())) + 1e-8)).mean()
        gp_h = _gradient_penalty(discriminator.dis_hash, h_i, h_t, device)
        (d_real_h + d_fake_h + gp_h).backward()
        opt_D_hash.step()

        # ── Generator (model) update ──────────────────────────────────────────
        adv_feat = -torch.log(torch.sigmoid(discriminator.dis_feature(f_t)) + 1e-8).mean()
        adv_hash = -torch.log(torch.sigmoid(discriminator.dis_hash(h_t)) + 1e-8).mean()
        loss_adv = adv_feat + adv_hash

        tri_i2t = _triplet_loss(h_i, h_t, labels, cfg.margin)
        tri_t2i = _triplet_loss(h_t, h_i, labels, cfg.margin)
        loss_tri = tri_i2t + tri_t2i

        loss_quant = (torch.pow(B_buf[idx].to(device) - h_i, 2).sum() +
                      torch.pow(B_buf[idx].to(device) - h_t, 2).sum())

        loss_G = cfg.alpha * loss_tri + cfg.beta * loss_quant + cfg.gamma * loss_adv

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        epoch_loss += loss_G.item()

    # ── Update binary code buffer B ───────────────────────────────────────────
    L = loader.dataset.get_labels().to(device)
    P = torch.inverse(L.t().mm(L) + cfg.lamb * torch.eye(L.shape[1], device=device))
    P = P.mm(L.t()).mm(B_buf.to(device))
    B_new = (L.mm(P) + 0.5 * cfg.mu * (H_i_buf.to(device) + H_t_buf.to(device))).sign()
    B_buf.copy_(B_new.cpu())

    return epoch_loss


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer factory
# ──────────────────────────────────────────────────────────────────────────────

def create_optimizers(model, discriminator, cfg):
    """Returns (opt_G, opt_D_feat, opt_D_hash)."""
    opt_G = torch.optim.Adam(
        list(model.parameters()), lr=cfg.lr, weight_decay=5e-4)
    opt_D_feat = torch.optim.Adam(
        discriminator.feature_dis.parameters(), lr=cfg.lr,
        betas=(0.5, 0.9), weight_decay=1e-4)
    opt_D_hash = torch.optim.Adam(
        discriminator.hash_dis.parameters(), lr=cfg.lr,
        betas=(0.5, 0.9), weight_decay=1e-4)
    return opt_G, opt_D_feat, opt_D_hash
