"""
DCMH — Deep Cross-Modal Hashing
================================
Reference: Jiang & Li, "Deep Cross-Modal Hashing", CVPR 2017

Loss:
  L = - sum_{i,j} [ 0.5 * S_ij * (h_i^v)^T h_j^t
                    - log(1 + exp(0.5 * (h_i^v)^T h_j^t)) ]
      + eta * (||F - B||² + ||G - B||²)   (quantization)
      + gamma * (||F 1||² + ||G 1||²)     (balance)

Binary codes B are updated outside the network in a closed-form step.
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ──────────────────────────────────────────────────────────────────────────────

def pairwise_log_loss(h_img: torch.Tensor,
                      h_txt: torch.Tensor,
                      S:     torch.Tensor) -> torch.Tensor:
    """
    Cross-modal log-likelihood loss (Eq. 3 in DCMH paper).

    h_img, h_txt : (B, K) real-valued.
    S            : (B, B) similarity matrix in {0, 1}.
    """
    inner = 0.5 * h_img.mm(h_txt.t())              # (B, B)
    log_term = torch.log(1.0 + torch.exp(inner))   # (B, B)
    return -(S * inner - log_term).mean()


def quantization_loss(h_img: torch.Tensor,
                      h_txt: torch.Tensor,
                      B:     torch.Tensor) -> torch.Tensor:
    """||h_img - B||² + ||h_txt - B||²"""
    return (F.mse_loss(h_img, B, reduction='sum') +
            F.mse_loss(h_txt, B, reduction='sum'))


def balance_loss(h_img: torch.Tensor,
                 h_txt: torch.Tensor) -> torch.Tensor:
    """Sum of squared column means (encourages balanced bits)."""
    return (h_img.mean(dim=0) ** 2).sum() + (h_txt.mean(dim=0) ** 2).sum()


# ──────────────────────────────────────────────────────────────────────────────
# Full combined loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(h_img, h_txt, B, labels, cfg):
    """
    Args:
        h_img, h_txt : (B, K) real-valued hashes.
        B            : (B, K) current binary code estimates.
        labels       : (B, C) multi-hot label matrix.
        cfg          : config with eta (quant weight) and gamma_bal (balance weight).
    """
    S = (labels.mm(labels.t()) > 0).float()
    l_log  = pairwise_log_loss(h_img, h_txt, S)
    l_quan = quantization_loss(h_img, h_txt, B)
    l_bal  = balance_loss(h_img, h_txt)
    return l_log + cfg.eta * l_quan + cfg.gamma_bal * l_bal


# ──────────────────────────────────────────────────────────────────────────────
# Binary code update (closed-form)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_B(F_all: torch.Tensor, G_all: torch.Tensor) -> torch.Tensor:
    """B = sign(F + G) where F, G are full-dataset hash buffers."""
    return (F_all + G_all).sign()


# ──────────────────────────────────────────────────────────────────────────────
# Per-epoch training step
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt_img, opt_txt, F_buf, G_buf, B_buf,
                epoch, cfg, device):
    """
    One training epoch for DCMH.

    Args:
        model           : HashNet instance.
        loader          : DataLoader yielding (idx, img, txt, label).
        opt_img, opt_txt: separate SGD optimizers for image/text branches.
        F_buf, G_buf    : (N, K) full-dataset hash-code buffers.
        B_buf           : (N, K) binary code buffer.
        epoch           : current epoch (0-based, used for LR decay).
        cfg             : config (eta, gamma_bal, lr, max_epoch).
        device          : torch.device.

    Returns:
        epoch_loss (float)
    """
    model.train()
    epoch_loss = 0.0

    # Linear LR decay
    lr_now = max(cfg.lr * (1 - epoch / cfg.max_epoch), 1e-6)
    for pg in opt_img.param_groups + opt_txt.param_groups:
        pg['lr'] = lr_now

    for idx, imgs, txts, labels in loader:
        imgs   = imgs.to(device).float()
        txts   = txts.to(device).float()
        labels = labels.to(device).float()

        h_img = model.forward_img(imgs)     # (B, K)
        h_txt = model.forward_txt(txts)

        F_buf[idx] = h_img.detach().cpu()
        G_buf[idx] = h_txt.detach().cpu()

        B_batch = B_buf[idx].to(device)
        loss = compute_loss(h_img, h_txt, B_batch, labels, cfg)

        opt_img.zero_grad(); opt_txt.zero_grad()
        loss.backward()
        opt_img.step(); opt_txt.step()
        epoch_loss += loss.item()

    # Update B after each epoch
    B_buf.copy_(update_B(F_buf, G_buf))
    return epoch_loss


def create_optimizers(model, cfg):
    """Returns (opt_img, opt_txt) — separate SGD for each branch."""
    opt_img = torch.optim.SGD(
        list(model.backbone.parameters()) + list(model.img_hash.parameters()),
        lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    opt_txt = torch.optim.SGD(
        model.txt_net.parameters(),
        lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    return opt_img, opt_txt
