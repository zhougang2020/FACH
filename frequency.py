"""
Differentiable 2D DCT / IDCT utilities for FACH.

All transforms are implemented via FFT so gradients flow through them
during PyTorch autograd.

Key additions vs initial version:
  - compute_consensus_sensitivity now supports a learnable weight vector W
    (Eq. 9 in paper: Ac = (1/M) * sum_m W^T * A_Tm)
"""

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1-D helpers
# ──────────────────────────────────────────────────────────────────────────────

def dct_1d(x: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D DCT-II along the last dimension."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.reshape(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip(1)], dim=1)
    Vc = torch.fft.fft(v, dim=1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device).unsqueeze(0) * (np.pi / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    return (2 * V).reshape(x_shape)


def idct_1d(X: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D IDCT-II along the last dimension."""
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.reshape(-1, N) / 2

    k = torch.arange(N, dtype=X.dtype, device=X.device).unsqueeze(0) * (np.pi / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip(1)[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V, dim=1).real

    x = v.new_zeros(v.shape)
    x[:, ::2] = v[:, : N - N // 2]
    x[:, 1::2] = v.flip(1)[:, : N // 2]
    return x.reshape(x_shape)


# ──────────────────────────────────────────────────────────────────────────────
# 2-D DCT / IDCT
# ──────────────────────────────────────────────────────────────────────────────

def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """2-D DCT: apply 1-D DCT along rows then columns."""
    X = dct_1d(x)
    X = dct_1d(X.transpose(-1, -2)).transpose(-1, -2)
    return X


def idct_2d(X: torch.Tensor) -> torch.Tensor:
    """2-D IDCT: apply 1-D IDCT along columns then rows."""
    x = idct_1d(X.transpose(-1, -2)).transpose(-1, -2)
    x = idct_1d(x)
    return x


# ──────────────────────────────────────────────────────────────────────────────
# Low-frequency mask  (Eq. 7)
# ──────────────────────────────────────────────────────────────────────────────

def get_low_freq_mask(h: int, w: int, tau: int, device: torch.device) -> torch.Tensor:
    """
    M_low(u,v) = 1 if 0 <= u,v <= tau, else 0.
    Returns (h, w).
    """
    tau = min(tau, h - 1, w - 1)
    mask = torch.zeros(h, w, device=device)
    mask[: tau + 1, : tau + 1] = 1.0
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity map  (Eqs. 5-6)
# ──────────────────────────────────────────────────────────────────────────────

def compute_sensitivity(
    model_fn,
    F_freq: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    """
    A(u,v) = |∂ Lt(model(IDCT(F))) / ∂ F(u,v)| ⊙ M_low

    Args:
        model_fn : callable (x_spatial -> hash_real).
        F_freq   : (B, C, H, W) frequency representation.
        mask     : (H, W) low-frequency mask.
        loss_fn  : callable (hash_real -> scalar loss).

    Returns:
        sensitivity: (B, C, H, W) masked absolute gradients.
    """
    F_tmp = F_freq.detach().requires_grad_(True)
    x_spatial = idct_2d(F_tmp)
    h = model_fn(x_spatial)
    loss = loss_fn(h)
    loss.backward()

    grad = F_tmp.grad
    sensitivity = grad.abs() * mask.unsqueeze(0).unsqueeze(0)
    return sensitivity.detach()


def compute_consensus_sensitivity(
    teacher_fns: list,
    F_freq: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Consensus sensitivity matrix Ac (Eq. 9):
        Ac = (1/M) * sum_m  W_m * A_{T_m}

    W is a learnable scalar weight per teacher (projected via W^T).
    If weights is None, uniform averaging is used.

    Args:
        teacher_fns : list of callables (x_spatial -> hash_real).
        F_freq      : (B, C, H, W) frequency representation.
        mask        : (H, W) low-frequency mask.
        loss_fn     : callable (hash_real -> scalar loss).
        weights     : (M,) tensor of scalar importance scores (W^T A_Tm).
                      If None, uniform 1/M weights are used.

    Returns:
        A_c : (B, C, H, W)
    """
    M = len(teacher_fns)
    if M == 0:
        raise ValueError("At least one teacher function is required.")

    if weights is None:
        weights = torch.ones(M, device=F_freq.device) / M
    else:
        # Normalise so weights sum to 1 (softmax-style, keeps scale stable)
        weights = F.softmax(weights, dim=0)

    A_c = None
    for m, teacher_fn in enumerate(teacher_fns):
        A_m = compute_sensitivity(teacher_fn, F_freq, mask, loss_fn)
        if A_c is None:
            A_c = weights[m] * A_m
        else:
            A_c = A_c + weights[m] * A_m

    return A_c  # (B, C, H, W)
