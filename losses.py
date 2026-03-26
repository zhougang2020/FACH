"""
Loss functions for FACH.

Implemented:
  - js_divergence         : Jensen-Shannon Divergence between two distributions
  - alignment_loss        : L_align (Eq. 13) — JSD between A_s and A_c
  - margin_enhanced_loss  : L_ME   (Eq. 14) — boundary-separation loss
  - distillation_loss     : L_distill (Eq. 16) — sum of align + ME losses
  - adversarial_loss      : L_adv (Eq. 19) — used during PGD attack
  - standard_hash_loss    : cross-modal pairwise log-likelihood (Eq. 3)
  - frequency_sensitivity_loss : Lt1/Lt2/Lt3/Lt4 options for sensitivity signal
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Jensen-Shannon Divergence
# ──────────────────────────────────────────────────────────────────────────────

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric JSD between two probability vectors (last dim is the event dim).

    JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m),  m = 0.5*(p+q)
    """
    p = p + eps
    q = q + eps
    m = 0.5 * (p + q)
    # KL(p||m) = sum p * log(p/m)
    kl_pm = (p * (p / m).log()).sum(dim=-1).mean()
    kl_qm = (q * (q / m).log()).sum(dim=-1).mean()
    return 0.5 * (kl_pm + kl_qm)


# ──────────────────────────────────────────────────────────────────────────────
# Alignment loss  L_align  (Eq. 13)
# ──────────────────────────────────────────────────────────────────────────────

def alignment_loss(A_s: torch.Tensor, A_c: torch.Tensor) -> torch.Tensor:
    """
    Aligns the substitute model's frequency sensitivity A_s with the
    teacher-consensus sensitivity A_c via JSD.

    A_s, A_c: any shape — they are flattened and softmax-normalised before JSD.
    """
    A_s_flat = A_s.flatten(1)  # (B, *)
    A_c_flat = A_c.flatten(1)  # (B, *)

    p = F.softmax(A_s_flat, dim=-1)
    q = F.softmax(A_c_flat, dim=-1)
    return js_divergence(p, q)


# ──────────────────────────────────────────────────────────────────────────────
# Margin-Enhanced loss  L_ME  (Eq. 14)
# ──────────────────────────────────────────────────────────────────────────────

def margin_enhanced_loss(
    h_s: torch.Tensor,
    t: torch.Tensor,
    margin: float = 1.5,
) -> torch.Tensor:
    """
    L_ME = sum_i max(m - h_s_i * t_i, 0)

    Args:
        h_s   : (B, K) real-valued hash codes from substitute model (before sign).
        t     : (B, K) consensus target codes, values in {-1, +1}.
        margin: separation margin m (default 1.5 from paper).
    """
    return F.relu(margin - h_s * t).sum(dim=1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Full distillation loss  L_distill  (Eq. 16)
# ──────────────────────────────────────────────────────────────────────────────

def distillation_loss(
    A_s: torch.Tensor,
    A_c: torch.Tensor,
    h_s: torch.Tensor,
    t: torch.Tensor,
    margin: float = 1.5,
) -> torch.Tensor:
    """L_distill = L_align + L_ME"""
    l_align = alignment_loss(A_s, A_c)
    l_me = margin_enhanced_loss(h_s, t, margin)
    return l_align + l_me


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial loss  L_adv  (Eq. 19)
# ──────────────────────────────────────────────────────────────────────────────

def adversarial_loss(
    h_adv: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.1,
    targeted: bool = False,
) -> torch.Tensor:
    """
    L_adv = (gamma / K) * b^T * tanh(alpha * h_adv)

    gamma = -1 (untargeted) or +1 (targeted).
    b     = original hash code (untargeted) or target hash code (targeted).
    """
    K = b.shape[-1]
    gamma = 1.0 if targeted else -1.0
    return (gamma / K) * (b * torch.tanh(alpha * h_adv)).sum(dim=-1).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Standard cross-modal hash loss  (Eq. 3)
# ──────────────────────────────────────────────────────────────────────────────

def standard_hash_loss(
    h_img: torch.Tensor,
    h_txt: torch.Tensor,
    S: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise log-likelihood loss for cross-modal hashing.

    L = -E_{i,j} [ 0.5 * S_ij * (h_i^v)^T h_j^t
                   - log(1 + exp(0.5 * (h_i^v)^T h_j^t)) ]

    Args:
        h_img: (B, K) image hash codes (real-valued).
        h_txt: (B, K) text  hash codes (real-valued).
        S    : (B, B) similarity matrix (1 if semantically similar, else 0).
    """
    inner = 0.5 * h_img.mm(h_txt.t())          # (B, B)
    log_loss = torch.log(1.0 + torch.exp(inner))  # (B, B)
    loss = -(S * inner - log_loss).mean()
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Frequency sensitivity signals  Lt1 – Lt4  (Eq. 11)
# ──────────────────────────────────────────────────────────────────────────────

def lt1_triplet(h: torch.Tensor, labels: torch.Tensor, phi_neg: float = None) -> torch.Tensor:
    """
    Lt1 = sum_{pos pairs} ||h_i - h_j||^2  +  sum_{neg pairs} max(phi_neg - ||h_i - h_j||^2, 0)

    phi_neg defaults to 2*sqrt(K) as suggested in the paper.
    """
    K = h.shape[-1]
    if phi_neg is None:
        phi_neg = 2.0 * (K ** 0.5)

    # Build pairwise similarity from labels
    sim = labels.mm(labels.t()) > 0  # (B, B)
    dist = torch.cdist(h, h, p=2) ** 2  # (B, B)

    pos_loss = (sim.float() * dist).mean()
    neg_loss = (F.relu(phi_neg - dist) * (~sim).float()).mean()
    return pos_loss + neg_loss


def lt2_margin(h: torch.Tensor, m: float = 1.5) -> torch.Tensor:
    """Lt2 = sum_i max(m - |h_i|, 0)  — pushes |h_i| > m away from zero."""
    return F.relu(m - h.abs()).mean()


def lt3_sign(h: torch.Tensor) -> torch.Tensor:
    """Lt3 = (1/K) * sign(h)^T * tanh(h)  — encourages decisive sign."""
    K = h.shape[-1]
    return (1.0 / K) * (h.sign() * torch.tanh(h)).sum(dim=-1).mean()


def lt4_contrastive(h: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Lt4: supervised contrastive loss in hash space.
    Same-class pairs attract; different-class pairs repel.
    """
    # Cosine similarity (h is not unit-normed, normalise first)
    h_norm = F.normalize(h, dim=-1)  # (B, K)
    sim_mat = h_norm.mm(h_norm.t()) / tau  # (B, B)

    # Mask diagonal
    B = h.shape[0]
    mask = torch.eye(B, device=h.device).bool()
    sim_mat = sim_mat.masked_fill(mask, -1e9)

    # Positive mask from labels
    pos_mask = (labels.mm(labels.t()) > 0).float()
    pos_mask = pos_mask.masked_fill(mask, 0.0)

    log_prob = F.log_softmax(sim_mat, dim=-1)
    # Mean over positive pairs only (avoid div-by-zero)
    pos_sum = pos_mask.sum(dim=-1).clamp(min=1)
    loss = -(log_prob * pos_mask).sum(dim=-1) / pos_sum
    return loss.mean()
