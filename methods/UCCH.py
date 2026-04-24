"""
UCCH — direct integration of original UCCH components into FACH.

Components ported verbatim from UCCH/NCE/ and UCCH/src/utils.py:
  - AliasMethod  (alias_multinomial.py)
  - NCEAverage   (NCE/NCEAverage.py)   — memory bank + NCE loss
  - NCESoftmaxLoss (NCE/NCECriterion.py)
  - ContrastiveLoss (src/utils.py)
  - UCCHHashNet   — exact UCCH architecture (nets/ImageNet.py + nets/TextNet.py)
                    replacing base.HashNet for UCCH training

Architecture (UCCHHashNet):
  Image: backbone(x) [4096] → Linear→ReLU→Linear→ReLU→Linear→Tanh→L2-norm
         hidden dim 8192 (=1024*8), 3 hidden layers  (ImageNet hiden_layer=3)
  Text:  Linear→ReLU→Linear→Tanh→L2-norm
         hidden dim 8192, 2 hidden layers             (TextNet  hiden_layer=2)
  Output: unit-norm vector on the hypersphere (same as original UCCH)

Original UCCH hyper-params used:
  T=0.9  (NCEAverage scales it to T*sqrt(bit)=7.2 internally)
  K=4096 negatives, momentum=0.4
  alpha=0.9  (NCE weight), margin=0.2, shift=1
  lr=1e-4, weight_decay=1e-6
  warmup_epoch=1  (backbone frozen, in-batch negatives)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .alias_multinomial import AliasMethod


# ──────────────────────────────────────────────────────────────────────────────
# NCEAverage  (verbatim copy from UCCH/NCE/NCEAverage.py)
# ──────────────────────────────────────────────────────────────────────────────

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=True):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.K = K
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', F.normalize(rnd.sign(), dim=1))

    def cuda(self, device=None):
        super().cuda(device)
        self.multinomial.cuda()   # AliasMethod tensors must move to CUDA manually
        return self

    def update_memory(self, data):
        memory = 0
        for i in range(len(data)):
            memory += data[i]
        memory /= memory.norm(dim=1, keepdim=True)
        self.memory.mul_(0).add_(memory)

    def forward(self, l, ab, y, idx=None, epoch=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())
        batchSize = l.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        if momentum <= 0:
            weight = (l + ab) / 2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat([torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])], dim=1).to(weight.device).view([-1])
            weight = weight[inx].view([batchSize, batchSize, -1])
        else:
            weight = torch.index_select(self.memory, 0, idx.view(-1)).detach().view(batchSize, K + 1, inputSize)

        weight = weight.sign_()
        out_ab = torch.bmm(weight, ab.view(batchSize, inputSize, 1))
        out_l  = torch.bmm(weight, l.view(batchSize, inputSize, 1))
        if self.use_softmax:
            out_ab = torch.div(out_ab, T).contiguous()
            out_l  = torch.div(out_l,  T).contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l  = torch.exp(torch.div(out_l,  T))
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
            out_l  = torch.div(out_l,  Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # update memory with EMA of (l+ab)/2
        with torch.no_grad():
            l_mem = (l + ab) / 2.
            l_mem = l_mem / l_mem.norm(dim=1, keepdim=True)
            l_pos = torch.index_select(self.memory, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l_mem, 1 - momentum))
            l_pos = l_pos / l_pos.norm(dim=1, keepdim=True)
            self.memory.index_copy_(0, y, l_pos)

        return out_l, out_ab


# ──────────────────────────────────────────────────────────────────────────────
# NCESoftmaxLoss  (verbatim copy from UCCH/NCE/NCECriterion.py)
# ──────────────────────────────────────────────────────────────────────────────

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def forward(self, x):
        x = x.softmax(1)
        return -x[:, 0].log().mean()


# ──────────────────────────────────────────────────────────────────────────────
# ContrastiveLoss  (ported from UCCH/src/utils.py)
# Uses L2-normalised features to prevent exp() overflow with FACH's Tanh output.
# Original UCCH uses raw unbounded features; with Tanh codes dot-products
# can reach ±64 which causes exp(64)→inf with tau=1.
# Normalising to unit sphere keeps scores in [-1,1] → exp stays safe.
# ──────────────────────────────────────────────────────────────────────────────

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, shift=1.0):
        super().__init__()
        self.margin = margin
        self.shift  = shift

    def forward(self, im, s, tau=1.0):
        # L2-normalise → cosine similarity in [-1, 1]
        im = F.normalize(im, dim=1)
        s  = F.normalize(s,  dim=1)
        scores = im.mm(s.t())

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        mask_s  = (scores >= (d1 - self.margin)).float().detach()
        cost_s  = scores * mask_s  + (1. - mask_s)  * (scores - self.shift)
        mask_im = (scores >= (d2 - self.margin)).float().detach()
        cost_im = scores * mask_im + (1. - mask_im) * (scores - self.shift)

        loss = ((-cost_s.diag()  + tau * (cost_s  / tau).exp().sum(1).log() + self.margin).mean() +
                (-cost_im.diag() + tau * (cost_im / tau).exp().sum(0).log() + self.margin).mean())
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# UCCHHashNet — exact UCCH model (nets/ImageNet.py + nets/TextNet.py)
# ──────────────────────────────────────────────────────────────────────────────

class UCCHHashNet(nn.Module):
    """
    Verbatim port of UCCH's image+text model pair into FACH's HashNet interface.

    Supports two modes:

    Feature mode (backbone_name=None, matching UCCH's 'mirflickr25k_fea' path):
        image_model = ImageNet(y_dim=feat_dim, bit=bit, hiden_layer=3)
        → input is pre-extracted 4096-dim VGG features (no backbone)
        This is how the UCCH paper achieves its reported results.

    Raw-image mode (backbone_name given, for end-to-end training):
        image_model = Sequential(backbone, ImageNet(...))
        → slower convergence, typically needs 100+ epochs.

    Image head  (nets/ImageNet.py, hiden_layer=3):
        feat_dim → 8192 → ReLU → 8192 → ReLU → bit → Tanh → L2-norm
    Text encoder (nets/TextNet.py, hiden_layer=2):
        text_dim → 8192 → ReLU → bit → Tanh → L2-norm

    All outputs are L2-normalised unit vectors — NCEAverage and ContrastiveLoss
    receive unit-norm features as in the original UCCH training loop.
    """

    def __init__(self, bit: int, text_dim: int,
                 feat_dim: int = 4096,
                 backbone_name: str = None, pretrained: bool = False,
                 mid_num: int = 8192):
        super().__init__()
        self.bit = bit

        # Optional backbone for raw-image mode
        if backbone_name is not None:
            from backbones import build_backbone
            self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        else:
            self.backbone = None   # feature mode — input already feat_dim

        # Image hash head — mirrors ImageNet(y_dim=feat_dim, bit=bit, hiden_layer=3)
        self.img_hash = nn.Sequential(
            nn.Linear(feat_dim, mid_num),
            nn.ReLU(inplace=True),
            nn.Linear(mid_num, mid_num),
            nn.ReLU(inplace=True),
            nn.Linear(mid_num, bit),
        )

        # Text encoder — mirrors TextNet(y_dim=text_dim, bit=bit, hiden_layer=2)
        self.txt_net = nn.Sequential(
            nn.Linear(text_dim, mid_num),
            nn.ReLU(inplace=True),
            nn.Linear(mid_num, bit),
        )

    # ── FACH HashNet interface ────────────────────────────────────────────────

    def forward_img(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x) if self.backbone is not None else x
        out = self.img_hash(feat).tanh()
        return F.normalize(out, dim=1)   # unit-norm, matches UCCH ImageNet.forward

    def forward_txt(self, t: torch.Tensor) -> torch.Tensor:
        out = self.txt_net(t).tanh()
        return F.normalize(out, dim=1)   # unit-norm, matches UCCH TextNet.forward

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.forward_img(x), self.forward_txt(t)

    def generate_img_code(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_img(x).sign()

    def generate_txt_code(self, t: torch.Tensor) -> torch.Tensor:
        return self.forward_txt(t).sign()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str, device=None):
        self.load_state_dict(torch.load(path, map_location=device or 'cpu'))


# ──────────────────────────────────────────────────────────────────────────────
# Training epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer,
                contrast, criterion_nce, criterion_cont,
                epoch, warmup_epoch, cfg, device):
    """
    One training epoch.  Mirrors UCCH.py train() function:
      - Backbone frozen during warmup_epoch (in-batch negatives via epoch<0)
      - loss = alpha * (nce_i + nce_t) + (1-alpha) * contrastive
      - gradient clipping at 1.0
    """
    model.train()

    # Warmup: freeze backbone (match UCCH set_train(epoch < warmup_epoch))
    # backbone is None in feature mode — nothing to freeze, skip.
    if model.backbone is not None:
        if epoch < warmup_epoch:
            model.backbone.eval()
            for p in model.backbone.parameters():
                p.requires_grad_(False)
        else:
            model.backbone.train()
            for p in model.backbone.parameters():
                p.requires_grad_(True)

    alpha = getattr(cfg, 'ucch_alpha', 0.9)
    epoch_loss = 0.0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1}', leave=False)
    for idx, imgs, txts, labels in pbar:
        idx  = idx.to(device)
        imgs = imgs.to(device).float()
        txts = txts.to(device).float()

        # UCCHHashNet.forward_img/txt already returns unit-norm vectors
        # (tanh → L2-norm inside the model, matching UCCH's ImageNet/TextNet)
        h_img = model.forward_img(imgs)   # (B, bit)  unit-norm
        h_txt = model.forward_txt(txts)   # (B, bit)  unit-norm

        # NCE loss — verbatim UCCH formula
        # epoch - warmup_epoch < 0 → in-batch negatives (warmup mode)
        out_l, out_ab = contrast(h_img, h_txt, idx, epoch=epoch - warmup_epoch)
        l_loss  = criterion_nce(out_l)
        ab_loss = criterion_nce(out_ab)
        L_nce   = l_loss + ab_loss

        # Contrastive alignment loss
        L_cont = criterion_cont(h_img, h_txt)

        loss = alpha * L_nce + (1.0 - alpha) * L_cont

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return epoch_loss


def create_optimizer(model, cfg):
    lr = getattr(cfg, 'lr', 1e-4)
    wd = getattr(cfg, 'weight_decay', 1e-6)
    # Same LR for all parameters — matches original UCCH (trains from scratch)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
