"""Quick smoke test before full training."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config import opt
from models import ImgNet, TxtNet
from datasets import load_data, Dataset
from torch.utils.data import DataLoader
from frequency import dct_2d, get_low_freq_mask, compute_sensitivity
from losses import (distillation_loss, margin_enhanced_loss, standard_hash_loss,
                    lt2_margin)
from attack import compute_global_semantic_hash, pgd_attack
from teacher_loader import TeacherWeights
import math

device = torch.device('cuda:0')
opt.parse({'dataset': 'mirflickr25k', 'bit': 64, 'device': 'cuda:0'})

print("1. Loading data ...")
images, tags, labels = load_data('mirflickr25k', use_vgg_feat=True)
print(f"   images={images.shape} tags={tags.shape} labels={labels.shape}")

print("2. Building models ...")
img_net = ImgNet(bit=64, image_dim=4096, hidden_dim=4096).to(device)
txt_net = TxtNet(bit=64, text_dim=1386, hidden_dim=4096).to(device)

print("3. Dataset split ...")
train_ds = Dataset(opt, images, tags, labels, partition='train')
q_ds     = Dataset(opt, images, tags, labels, partition='query')
db_ds    = Dataset(opt, images, tags, labels, partition='database')
print(f"   train={len(train_ds)} query={len(q_ds)} db={len(db_ds)}")

print("4. Forward pass ...")
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
idx, imgs, txts, lbls = next(iter(train_dl))
imgs = imgs.to(device).float()
txts = txts.to(device).float()
lbls = lbls.to(device).float()

h_img = img_net(imgs)
h_txt = txt_net(txts)
print(f"   h_img={h_img.shape} h_txt={h_txt.shape}")

print("5. Frequency ops ...")
side = math.ceil(math.sqrt(4096))  # 64
D_pad = side * side
imgs_pad = torch.cat([imgs, torch.zeros(imgs.shape[0], D_pad - 4096, device=device)], dim=1)
imgs_2d = imgs_pad.reshape(imgs.shape[0], 1, side, side)
F = dct_2d(imgs_2d)
mask = get_low_freq_mask(side, side, 20, device)
loss_fn = lambda h: lt2_margin(h, m=1.0)
A_s = compute_sensitivity(lambda x: img_net(x.flatten(1)[:, :4096]), F, mask, loss_fn)
print(f"   F={F.shape} A_s={A_s.shape}")

print("6. Losses ...")
t_code = h_img.detach().sign()
l_me = margin_enhanced_loss(h_img, t_code, 1.0)
S = (lbls.mm(lbls.t()) > 0).float()
l_hash = standard_hash_loss(h_img, h_txt, S)
print(f"   L_ME={l_me.item():.4f}  L_hash={l_hash.item():.4f}")

print("7. TeacherWeights ...")
tw = TeacherWeights(3).to(device)
w = tw()
print(f"   weights={w}")

print("8. Global semantic hash ...")
q_lbl  = q_ds.get_labels().to(device)
db_lbl = db_ds.get_labels().to(device)
db_dl  = DataLoader(db_ds, batch_size=64, shuffle=False, num_workers=0)
db_codes = []
with torch.no_grad():
    for batch in db_dl:
        db_codes.append(img_net.generate_hash(batch[0].to(device).float()).cpu())
db_codes = torch.cat(db_codes).to(device)
b_target = compute_global_semantic_hash(q_lbl[:32], db_codes, db_lbl)
print(f"   b_target={b_target.shape}")

print("\n=== All checks passed ===")
