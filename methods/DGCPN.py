"""
DGCPN — Deep Graph-based Cross-modal Proximity Hashing
=======================================================
Reference: Yu et al., "Deep Graph-based Cross-Modal Hashing", ACM MM 2021

Core idea:
  Builds a batch-level graph proximity matrix S that combines:
    (1) first-order semantic similarity  (cosine of normalised features)
    (2) second-order proximity           (diffusion of the kNN graph)
  Then minimises MSE between normalised hash inner-products and S,
  plus a diagonal consistency term (same-sample cross-modal alignment).

Hyper-parameters (from paper / default):
    a1        : weight for text similarity in S_pair  (0.01)
    a2        : weight for second-order term in S     (0.3)
    K_diag    : target diagonal value                 (1.5)
    knn_number: number of neighbours zeroed in diffusion graph (3000 on flickr)
    scale     : scaling factor for diffusion proximity (4000)
    dw, cw    : weights for distance loss and consistency loss
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Graph proximity matrix
# ──────────────────────────────────────────────────────────────────────────────

def build_S(f_img: torch.Tensor, f_txt: torch.Tensor, cfg) -> torch.Tensor:
    """
    Compute the graph-based similarity matrix S for a batch.

    Args:
        f_img, f_txt : (B, D) normalised feature vectors.
        cfg          : config with a1, a2, knn_number, scale.

    Returns:
        S : (B, B) similarity matrix in [-1, 1].
    """
    B = f_img.shape[0]
    f_i = F.normalize(f_img, dim=1)
    f_t = F.normalize(f_txt, dim=1)

    S_I = f_i.mm(f_i.t())                     # (B, B)
    S_T = f_t.mm(f_t.t())
    S_pair = cfg.a1 * S_T + (1 - cfg.a1) * S_I

    # Second-order proximity via diffusion
    pro = f_t.mm(f_t.t()) * cfg.a1 + f_i.mm(f_i.t()) * (1 - cfg.a1)
    # Zero out knn_number smallest entries per row → keep only "far" pairs
    k = min(cfg.knn_number, B - 1)
    _, sort_idx = pro.sort(dim=1)
    # Zero nearest k neighbours (excluding self)
    rows = torch.arange(B, device=pro.device).unsqueeze(1).expand(B, k)
    cols = sort_idx[:, :k]
    pro[rows, cols] = 0.0
    # Zero diagonal
    pro.fill_diagonal_(0.0)
    # Row-normalise to get transition probability
    row_sum = pro.sum(dim=1, keepdim=True).clamp(min=1e-8)
    pro = pro / row_sum
    pro_dis = pro.mm(pro.t()) * cfg.scale

    S = S_pair * (1 - cfg.a2) + pro_dis * cfg.a2
    S = S * 2.0 - 1.0                         # rescale to [-1, 1]
    return S


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(h_img: torch.Tensor, h_txt: torch.Tensor,
                 f_img: torch.Tensor, f_txt: torch.Tensor,
                 cfg) -> torch.Tensor:
    """
    DGCPN combined loss.

    l_pair  : MSE between normalised intra/inter-modal inner-products and S.
    l_diag  : force h_img[i]^T h_txt[i] / K ≈ K_diag for each sample.
    """
    S = build_S(f_img, f_txt, cfg)

    B_I = F.normalize(h_img, dim=1)
    B_T = F.normalize(h_txt, dim=1)
    BI_BI = B_I.mm(B_I.t())
    BT_BT = B_T.mm(B_T.t())
    BI_BT = B_I.mm(B_T.t())

    I = torch.eye(h_img.shape[0], device=h_img.device)

    l_dist = (F.mse_loss(BT_BT * (1 - I), S * (1 - I)) +
              F.mse_loss(BI_BT * (1 - I), S * (1 - I)) +
              F.mse_loss(BI_BI * (1 - I), S * (1 - I)))

    diag_val = BI_BT.diagonal()
    target   = torch.full_like(diag_val, cfg.K_diag)
    l_diag   = F.mse_loss(diag_val, target)

    return cfg.dw * l_dist + cfg.cw * l_diag


# ──────────────────────────────────────────────────────────────────────────────
# Per-epoch training
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt_img, opt_txt, epoch, cfg, device):
    """
    One training epoch for DGCPN.

    Returns:
        epoch_loss (float)
    """
    model.train()
    epoch_loss = 0.0

    for idx, imgs, txts, labels in loader:
        imgs = imgs.to(device).float()
        txts = txts.to(device).float()

        h_img = model.forward_img(imgs)
        h_txt = model.forward_txt(txts)
        f_img = model.get_img_feat(imgs)
        # Use text hidden features as f_txt proxy
        f_txt = model.txt_net.net[:3](txts)    # up to first ReLU

        # Ensure same dimension for graph computation
        if f_img.shape[1] != f_txt.shape[1]:
            f_img = F.normalize(f_img[:, :f_txt.shape[1]], dim=1)
        f_txt = F.normalize(f_txt, dim=1)
        f_img = F.normalize(f_img, dim=1)

        loss = compute_loss(h_img, h_txt, f_img, f_txt, cfg)

        opt_img.zero_grad(); opt_txt.zero_grad()
        loss.backward()
        opt_img.step(); opt_txt.step()
        epoch_loss += loss.item()

    return epoch_loss


def create_optimizers(model, cfg):
    """Returns (opt_img, opt_txt) — SGD with momentum."""
    opt_img = torch.optim.SGD(
        list(model.backbone.parameters()) + list(model.img_hash.parameters()),
        lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    opt_txt = torch.optim.SGD(
        model.txt_net.parameters(),
        lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    return opt_img, opt_txt
