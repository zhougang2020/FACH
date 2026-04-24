"""
PGD adversarial attack in the frequency domain (FACH Phase 2).

Algorithm (Eqs. 13-15):
  ΔF_T = Π_{[-δ,δ]}(ΔF_{T-1} + μ · sign(∇_{F'_{T-1}} L_adv ⊙ A_c))
  x' = Clip_{[0,1]}(x + Π_{[-ε,ε]}(IDCT(F_0 + ΔF_T) - x))

  L_adv = sum_i max(madv - γ · b_i · H_i(x'), 0)   (Eq. 15, hinge-based)

Global semantic hash code as attack target (Eq. 16):
  b_q = sign(sum_{pos} w_i * b_i^pos - sum_{neg} w_j * b_j^neg)
  w_pos = s / Np,  w_neg = (1-s) / Nn
"""

import math
import torch
import torch.nn.functional as F

from frequency import dct_2d, idct_2d
from losses import adversarial_loss


# ──────────────────────────────────────────────────────────────────────────────
# α schedule (paper Sec. IV-B-3, following SAAT)
# ──────────────────────────────────────────────────────────────────────────────

def _alpha_schedule(t: int) -> float:
    """
    α = 0.1 for t in [0, 49]
    α = 0.2 for t in [50, 59]
    α = 0.3 for t in [60, 69]
    α = 0.5 for t in [70, 79]
    α = 0.7 for t in [80, 89]
    α = 1.0 for t in [90, T]
    """
    if t < 50:   return 0.1
    elif t < 60: return 0.2
    elif t < 70: return 0.3
    elif t < 80: return 0.5
    elif t < 90: return 0.7
    else:        return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Feature ↔ 2D helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_2d(x: torch.Tensor):
    if x.dim() == 4:
        return x, None
    B, D = x.shape
    s = int(math.isqrt(D))
    if s * s == D:
        return x.reshape(B, 1, s, s), ('square', B, D, s)
    s_next = s + 1
    D_pad = s_next * s_next
    x_padded = F.pad(x, (0, D_pad - D))
    return x_padded.reshape(B, 1, s_next, s_next), ('padded', B, D, s_next, D_pad)


def _from_2d(x_2d: torch.Tensor, info):
    if info is None:
        return x_2d
    if info[0] == 'square':
        _, B, D, _ = info
        return x_2d.reshape(B, D)
    _, B, D, s_next, D_pad = info
    return x_2d.reshape(B, D_pad)[:, :D]


# ──────────────────────────────────────────────────────────────────────────────
# Global semantic hash code generation  (Eq. 16)
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_semantic_hash(
    query_labels: torch.Tensor,
    db_codes: torch.Tensor,
    db_labels: torch.Tensor,
) -> torch.Tensor:
    """
    For each query, compute the global semantic hash code b_q (Eq. 16):
        b_q = sign(sum_{pos} w_i * b_i^pos - sum_{neg} w_j * b_j^neg)
        w_pos = s / Np,  w_neg = (1-s) / Nn

    Args:
        query_labels : (Q, C) multi-hot label matrix.
        db_codes     : (R, K) binary hash codes of database.
        db_labels    : (R, C) multi-hot label matrix.

    Returns:
        b_target : (Q, K) target hash codes in {-1, +1}.
    """
    # Cosine similarity between label vectors
    q_norm = F.normalize(query_labels.float(), dim=1)   # (Q, C)
    d_norm = F.normalize(db_labels.float(), dim=1)       # (R, C)
    S = q_norm.mm(d_norm.t())                            # (Q, R) in [0, 1]

    b_target = []
    for i in range(query_labels.shape[0]):
        s_i = S[i]                                       # (R,)
        pos_mask = s_i > 0
        neg_mask = ~pos_mask

        Np = pos_mask.sum().clamp(min=1).float()
        Nn = neg_mask.sum().clamp(min=1).float()

        w_pos = (s_i * pos_mask.float()) / Np            # (R,)
        w_neg = ((1 - s_i) * neg_mask.float()) / Nn      # (R,)

        agg = (w_pos.unsqueeze(1) * db_codes).sum(0) - \
              (w_neg.unsqueeze(1) * db_codes).sum(0)     # (K,)
        b_target.append(agg.sign())

    return torch.stack(b_target, dim=0)                  # (Q, K)


# ──────────────────────────────────────────────────────────────────────────────
# Main PGD attack  (Eqs. 13-15)
# ──────────────────────────────────────────────────────────────────────────────

def pgd_attack(
    victim_fn,
    x: torch.Tensor,
    b_orig: torch.Tensor,
    A_c: torch.Tensor,
    opt,
    b_target: torch.Tensor = None,
) -> torch.Tensor:
    """
    Frequency-domain PGD attack.

    Args:
        victim_fn : callable (x -> h_real) — victim model, no sign applied.
        x         : (B, C, H, W) or (B, D) clean input.
        b_orig    : (B, K) original binary hash code from victim model.
        A_c       : consensus sensitivity map, same shape as x (or 2D form).
        opt       : config with T, mu, delta, eps, madv, targeted, device.
        b_target  : (B, K) target hash code for targeted attack (None → untargeted).

    Returns:
        x_adv : adversarial input, same shape as x.
    """
    targeted = b_target is not None
    b = b_target if targeted else b_orig

    x_2d, reshape_info = _to_2d(x)
    A_c_2d, _ = _to_2d(A_c)

    with torch.no_grad():
        F0 = dct_2d(x_2d)

    dF = torch.zeros_like(F0)

    for t in range(opt.T):
        dF = dF.detach().requires_grad_(True)
        F_adv = F0 + dF
        x_adv_2d = idct_2d(F_adv)

        if reshape_info is None:
            x_adv_2d = x_adv_2d.clamp(0.0, 1.0)

        x_adv_flat = _from_2d(x_adv_2d, reshape_info)
        h_adv = victim_fn(x_adv_flat)

        loss = adversarial_loss(h_adv, b, madv=opt.madv, targeted=targeted)
        loss.backward()

        with torch.no_grad():
            grad = dF.grad * A_c_2d
            dF = dF + opt.mu * grad.sign()   # gradient ascent (maximise loss)
            dF = dF.clamp(-opt.delta, opt.delta)

    with torch.no_grad():
        F_final = F0 + dF
        x_adv_2d = idct_2d(F_final)
        x_adv_flat = _from_2d(x_adv_2d, reshape_info)
        x_flat = _from_2d(x_2d, reshape_info)

        delta_x = (x_adv_flat - x_flat).clamp(-opt.eps, opt.eps)
        x_adv_out = x_flat + delta_x

        if reshape_info is None:
            x_adv_out = x_adv_out.clamp(0.0, 1.0)

    return x_adv_out
