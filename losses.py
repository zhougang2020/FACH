"""
Loss functions for FACH (aligned with paper equations).

  - js_divergence         : Jensen-Shannon Divergence
  - alignment_loss        : L_align (Eq. 10) — JSD between A_s and A_c
  - margin_enhanced_loss  : L_ME   (Eq. 11) — boundary-separation loss
  - distillation_loss     : L_distill (Eq. 12) — align + ME
  - adversarial_loss      : L_adv (Eq. 15) — hinge-based boundary-crossing loss
  - lt1/lt2/lt3/lt4       : frequency sensitivity signals (Eq. 8)
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Jensen-Shannon Divergence
# ──────────────────────────────────────────────────────────────────────────────

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1).mean()
    kl_qm = (q * (q / m).log()).sum(dim=-1).mean()
    return 0.5 * (kl_pm + kl_qm)


# ──────────────────────────────────────────────────────────────────────────────
# Alignment loss  L_align  (Eq. 10)
# ──────────────────────────────────────────────────────────────────────────────

def alignment_loss(A_s: torch.Tensor, A_c: torch.Tensor) -> torch.Tensor:
    """JSD between softmax-normalised sensitivity maps."""
    p = F.softmax(A_s.flatten(1), dim=-1)
    q = F.softmax(A_c.flatten(1), dim=-1)
    return js_divergence(p, q)


# ──────────────────────────────────────────────────────────────────────────────
# Margin-Enhanced loss  L_ME  (Eq. 11)
# ──────────────────────────────────────────────────────────────────────────────

def margin_enhanced_loss(
    h_s: torch.Tensor,
    t: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    L_ME = sum_i max(m - h_s_i * t_i, 0)

    h_s : (B, K) real-valued hash codes from substitute model.
    t   : (B, K) consensus target codes in {-1, +1}.
    """
    return F.relu(margin - h_s * t).sum(dim=1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Full distillation loss  L_distill  (Eq. 12)
# ──────────────────────────────────────────────────────────────────────────────

def distillation_loss(
    A_s: torch.Tensor,
    A_c: torch.Tensor,
    h_s: torch.Tensor,
    t: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """L_distill = L_align + L_ME"""
    return alignment_loss(A_s, A_c) + margin_enhanced_loss(h_s, t, margin)


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial loss  L_adv  (Eq. 15) — hinge-based boundary-crossing
# ──────────────────────────────────────────────────────────────────────────────

def adversarial_loss(
    h_adv: torch.Tensor,
    b: torch.Tensor,
    madv: float = 1.2,
    targeted: bool = True,
) -> torch.Tensor:
    """
    L_adv = sum_i max(madv - gamma * b_i * H_i(x'), 0)

    gamma = +1 (targeted: pull toward bt)
    gamma = -1 (untargeted: push away from bx)

    b     : (B, K) target hash code (targeted) or original hash code (untargeted).
    h_adv : (B, K) pre-quantization continuous output of victim model on x'.
    """
    gamma = 1.0 if targeted else -1.0
    return F.relu(madv - gamma * b * h_adv).sum(dim=1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Standard cross-modal hash loss  (for TextNet pre-training, Eq. 2 in paper)
# ──────────────────────────────────────────────────────────────────────────────

def standard_hash_loss(
    h_img: torch.Tensor,
    h_txt: torch.Tensor,
    S: torch.Tensor,
) -> torch.Tensor:
    """Pairwise log-likelihood loss for cross-modal hashing."""
    inner = 0.5 * h_img.mm(h_txt.t())
    log_loss = torch.log(1.0 + torch.exp(inner))
    return -(S * inner - log_loss).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Frequency sensitivity signals  Lt1 – Lt4  (Eq. 8)
# ──────────────────────────────────────────────────────────────────────────────

def lt1_triplet(h: torch.Tensor, labels: torch.Tensor, phi: float = None) -> torch.Tensor:
    """
    Lt1 = sum_{(i,j) in P} (||h_i - h_j||_2^2 - phi)^2   (Eq. 8, paper)

    phi_pos = 0  (positive pairs should be close)
    phi_neg = 2*sqrt(K)  (negative pairs should be far apart)
    """
    K = h.shape[-1]
    phi_neg = 2.0 * (K ** 0.5) if phi is None else phi

    sim = labels.mm(labels.t()) > 0
    dist = torch.cdist(h, h, p=2) ** 2

    pos_loss = (sim.float() * dist ** 2).mean()
    neg_loss = ((~sim).float() * (dist - phi_neg) ** 2).mean()
    return pos_loss + neg_loss


def lt2_margin(h: torch.Tensor, m: float = 1.0) -> torch.Tensor:
    """Lt2 = sum_i max(m - |h_i|, 0)"""
    return F.relu(m - h.abs()).mean()


def lt3_sign(h: torch.Tensor) -> torch.Tensor:
    """Lt3 = (1/K) * sign(h)^T * tanh(h)"""
    K = h.shape[-1]
    return (1.0 / K) * (h.sign() * torch.tanh(h)).sum(dim=-1).mean()


def lt4_contrastive(h: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Lt4: supervised contrastive loss in hash space."""
    h_norm = F.normalize(h, dim=-1)
    sim_mat = h_norm.mm(h_norm.t()) / tau

    B = h.shape[0]
    mask = torch.eye(B, device=h.device).bool()
    sim_mat = sim_mat.masked_fill(mask, -1e9)

    pos_mask = (labels.mm(labels.t()) > 0).float()
    pos_mask = pos_mask.masked_fill(mask, 0.0)

    log_prob = F.log_softmax(sim_mat, dim=-1)
    pos_sum = pos_mask.sum(dim=-1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=-1) / pos_sum
    return loss.mean()
