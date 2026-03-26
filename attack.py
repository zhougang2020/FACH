"""
PGD adversarial attack in the frequency domain (FACH Phase 2).

Algorithm (Eq. 17-18):
  For t = 1 … T:
    ΔF_t = Π_{[-δ,δ]}(ΔF_{t-1} + μ · sign(∇_{F'_{t-1}} L_adv ⊙ A_c))
  x' = Clip_{[0,1]}(x + Π_{[-ε,ε]}(IDCT(F_0 + ΔF_T) - x))

Two operating modes are supported:
  1. Image mode  (feature_mode=False):
       Input is (B, C, H, W) raw RGB images.
       DCT is applied channel-wise on H×W spatial dimensions.
  2. Feature mode (feature_mode=True):
       Input is (B, D) precomputed feature vectors.
       Feature vector is reshaped to (B, 1, sqrt(D), sqrt(D)) for 2D DCT,
       or DCT is applied as a 1D transform.
"""

import math
import torch
import torch.nn.functional as F

from frequency import dct_2d, idct_2d, get_low_freq_mask
from losses import adversarial_loss


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _alpha_schedule(t: int, T: int) -> float:
    """
    α schedule from the paper:
      α = 0.1  for t in [0, 49]
      α = 0.2  for t in [50, 59]
      α = 0.3  for t in [60, 69]
      α = 0.5  for t in [70, 79]
      α = 0.7  for t in [80, 89]
      α = 1.0  for t in [90, T]
    """
    if t < 50:
        return 0.1
    elif t < 60:
        return 0.2
    elif t < 70:
        return 0.3
    elif t < 80:
        return 0.5
    elif t < 90:
        return 0.7
    else:
        return 1.0


def _to_2d(x: torch.Tensor):
    """
    Map an arbitrary tensor to a form suitable for 2D DCT.

    - (B, C, H, W)  → returned as-is,           reshape_info = None
    - (B, D)        → try to reshape to (B, 1, s, s) where s=sqrt(D);
                       if D is not a perfect square, pad to next square.
    Returns (x_2d, reshape_info) where reshape_info stores original shape.
    """
    if x.dim() == 4:
        return x, None  # already (B, C, H, W)

    B, D = x.shape
    s = int(math.isqrt(D))
    if s * s == D:
        return x.reshape(B, 1, s, s), ('square', B, D, s)
    # Pad to next perfect square
    s_next = s + 1
    D_pad = s_next * s_next
    x_padded = F.pad(x, (0, D_pad - D))
    return x_padded.reshape(B, 1, s_next, s_next), ('padded', B, D, s_next, D_pad)


def _from_2d(x_2d: torch.Tensor, info):
    """Undo _to_2d."""
    if info is None:
        return x_2d
    if info[0] == 'square':
        _, B, D, _ = info
        return x_2d.reshape(B, D)
    _, B, D, s_next, D_pad = info
    return x_2d.reshape(B, D_pad)[:, :D]


# ──────────────────────────────────────────────────────────────────────────────
# Main attack function
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
        victim_fn  : callable (x → h_real) — victim model, no sign applied.
        x          : (B, C, H, W) or (B, D) clean input.
        b_orig     : (B, K) original binary hash code of x from victim model.
        A_c        : (B, C, H, W) or (B, D) consensus sensitivity from substitute.
        opt        : config object with fields: T, mu, delta, eps, device.
        b_target   : (B, K) target hash code for targeted attack (None → untargeted).

    Returns:
        x_adv : adversarial input, same shape as x, values clipped to [0,1]
                (or to feature range in feature mode).
    """
    targeted = b_target is not None
    b = b_target if targeted else b_orig

    # --- Convert to 2D for DCT ---
    x_2d, reshape_info = _to_2d(x)
    A_c_2d, _ = _to_2d(A_c)

    # Initial frequency representation
    with torch.no_grad():
        F0 = dct_2d(x_2d)          # (B, C, H, W)

    dF = torch.zeros_like(F0)      # perturbation in frequency domain

    for t in range(opt.T):
        alpha = _alpha_schedule(t, opt.T)

        dF = dF.detach().requires_grad_(True)
        F_adv = F0 + dF
        x_adv_2d = idct_2d(F_adv)

        # Clip spatial domain if image mode
        if reshape_info is None:
            x_adv_2d = x_adv_2d.clamp(0.0, 1.0)

        x_adv_flat = _from_2d(x_adv_2d, reshape_info)

        # Victim model output
        h_adv = victim_fn(x_adv_flat)

        loss = adversarial_loss(h_adv, b, alpha=alpha, targeted=targeted)
        loss.backward()

        with torch.no_grad():
            # Gradient guided by consensus sensitivity
            grad = dF.grad * A_c_2d          # (B, C, H, W)
            dF = dF - opt.mu * grad.sign()   # gradient ascent on loss
            dF = dF.clamp(-opt.delta, opt.delta)

    # --- Reconstruct adversarial sample ---
    with torch.no_grad():
        F_final = F0 + dF
        x_adv_2d = idct_2d(F_final)
        x_adv_flat = _from_2d(x_adv_2d, reshape_info)
        x_flat = _from_2d(x_2d, reshape_info)

        # Project perturbation to L∞ ε-ball in input space
        delta_x = (x_adv_flat - x_flat).clamp(-opt.eps, opt.eps)
        x_adv_out = x_flat + delta_x

        if reshape_info is None:
            x_adv_out = x_adv_out.clamp(0.0, 1.0)

    return x_adv_out


# ──────────────────────────────────────────────────────────────────────────────
# Batch attack wrapper
# ──────────────────────────────────────────────────────────────────────────────

def batch_attack(
    victim_fn,
    dataloader,
    substitute_img_fn,
    A_c_fn,
    opt,
    global_semantic_hash_fn=None,
) -> torch.Tensor:
    """
    Attack all query samples and return their adversarial counterparts.

    Args:
        victim_fn             : callable (x → h_real) victim model.
        dataloader            : DataLoader yielding clean query images.
        substitute_img_fn     : callable (x → h_real) substitute model.
        A_c_fn                : callable (x → A_c) returns consensus sensitivity.
        opt                   : config object.
        global_semantic_hash_fn: callable (x, labels → b_target) for targeted attacks;
                                 None for untargeted attacks.

    Returns:
        all_adv : (N, *input_shape) adversarial samples on CPU.
    """
    all_adv = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(opt.device)
        else:
            x = batch.to(opt.device)

        with torch.no_grad():
            b_orig = victim_fn(x).sign()
            A_c = A_c_fn(x)

        b_target = None
        if global_semantic_hash_fn is not None:
            b_target = global_semantic_hash_fn(x, b_orig)

        x_adv = pgd_attack(victim_fn, x, b_orig, A_c, opt, b_target)
        all_adv.append(x_adv.cpu())

    return torch.cat(all_adv, dim=0)
