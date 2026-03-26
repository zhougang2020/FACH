

# FACH — Frequency-domain Adversarial Cross-modal Hashing

Reproduction of the paper:
> **Frequency-Domain Adversarial Robustness Evaluation for Deep Cross-Modal Hashing Systems**

---

## Overview

FACH is a black-box adversarial attack framework for cross-modal hashing systems.
It operates in two phases:

**Phase 1 — Substitute model training**
- Train a surrogate cross-modal hashing model (ImgNet + TxtNet) using
  multi-teacher knowledge distillation.
- Distillation is guided by two losses:
  - **L_align**: JSD between substitute and teacher frequency sensitivities (Eq. 13)
  - **L_ME**: Margin-enhanced loss pushing hash codes away from the zero boundary (Eq. 14)

**Phase 2 — Adversarial example generation**
- Use PGD in the frequency (DCT) domain, guided by the consensus sensitivity map A_c.
- Perturbations are clipped to [−δ, δ] in frequency space and [−ε, ε] in pixel space.

---

## File Structure

```
FACH/
├── main.py            # Entry point — train / attack / run
├── config.py          # All hyperparameters
├── frequency.py       # Differentiable 2D DCT/IDCT, sensitivity computation
├── losses.py          # L_align, L_ME, L_distill, L_adv, L_t1–L_t4
├── attack.py          # PGD frequency-domain attack
├── teacher_loader.py  # Load pre-trained victim models (DADH/DCMH/UCCH/…)
├── utils.py           # mAP, t-mAP evaluation metrics
├── models/
│   ├── img_net.py     # Image substitute model (VGG11 or MLP)
│   └── txt_net.py     # Text substitute model (3-layer MLP)
└── datasets/
    ├── data_handler.py  # .mat file loading (FLICKR-25K / NUS-WIDE / MS-COCO)
    └── dataset.py       # PyTorch Dataset with train/query/db splits
```

---

## Requirements

```bash
pip install torch torchvision scipy numpy tqdm fire
```

---

## Data Preparation

Place the dataset `.mat` files in `./data/`:

| Dataset    | File name            | Key fields                        |
|------------|----------------------|-----------------------------------|
| FLICKR-25K | `FLICKR-25K.mat`     | `images` (N×4096), `YAll`, `LAll` |
| NUS-WIDE   | `NUS-WIDE-TC21.mat`  | `image`, `text`, `label`          |
| MS-COCO    | `MS-COCO.mat`        | `images`, `tags`, `labels`        |

---

## Usage

### Phase 1: Train substitute model (without teachers)

```bash
cd FACH
python main.py train flag=mir bit=64 device=cuda:0
```

### Phase 1: Train with pre-trained victim models as teachers

```python
# Edit config.py or pass teacher_configs:
python main.py train \
  flag=mir bit=64 device=cuda:0 \
  "teacher_configs=[{'type':'DADH','ckpt_path':'../DADH/checkpoints/flickr25k_64'},{'type':'UCCH','ckpt_path':'../UCCH/checkpoints/flickr25k_64'}]"
```

### Phase 2: Attack evaluation

```bash
python main.py attack flag=mir bit=64 device=cuda:0 \
  "teacher_configs=[{'type':'DADH','ckpt_path':'../DADH/checkpoints/flickr25k_64'}]"
```

### Run both phases

```bash
python main.py run flag=mir bit=64 device=cuda:0
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bit` | 64 | Hash code length K (paper tests 16/32/64/128) |
| `margin` | 1.5 | Boundary margin m in L_ME (Eq. 14) |
| `tau_freq` | 20 | Low-frequency threshold τ for mask M_low |
| `T` | 100 | PGD attack iterations |
| `mu` | 0.001 | PGD step size |
| `delta` | 0.3 | Frequency-domain clipping bound δ |
| `eps` | 8/255 | Spatial L∞ perturbation bound ε |
| `lr` | 1e-4 | Adam learning rate for substitute model |
| `max_epoch` | 20 | Training epochs for substitute model |
| `sensitivity_loss` | `'lt2'` | Sensitivity signal: `lt1`/`lt2`/`lt3`/`lt4` |

---

## Algorithm Summary

### Sensitivity computation (Eqs. 8-9)

```
A(u,v) = |∂ L_t(model(IDCT(F))) / ∂ F(u,v)| ⊙ M_low(u,v)
```

where `M_low(u,v) = 1` if `0 ≤ u,v ≤ τ`, else 0.

### Consensus sensitivity (Eq. 12)

```
A_c = (1/M) Σ_m A_{T_m}
```

### Distillation losses

```
L_align = JSD(softmax(A_s) || softmax(A_c))          (Eq. 13)
L_ME    = Σ_i max(m - ĥ_{s,i} · t_i, 0)             (Eq. 14)
L_distill = L_align + L_ME                            (Eq. 16)
```

### PGD attack (Eq. 17-18)

```
ΔF_t = Π_{[-δ,δ]}(ΔF_{t-1} + μ · sign(∇ L_adv ⊙ A_c))
x' = Clip_{[0,1]}(x + Π_{[-ε,ε]}(IDCT(F_0 + ΔF_T) - x))
```

### Adversarial loss (Eq. 19)

```
L_adv = (γ/K) · b^T · tanh(α · h(x'))
γ = -1 (untargeted),  b = b_x (original hash code)
α schedule: 0.1 → 0.2 → 0.3 → 0.5 → 0.7 → 1.0
```

---

## Notes

- **Feature mode** (default): Works with precomputed 4096-dim VGG-F features from `.mat` files.
  The feature vector is reshaped to a 2D grid for DCT analysis.
- **Backbone mode** (`use_backbone=True`): Uses VGG11 on raw RGB images (requires image files,
  not `.mat` features). Set `use_backbone=True` in config.
- When no `teacher_configs` are provided, the substitute model trains in self-distillation
  mode (L_ME + cross-modal hash loss) — useful for ablation studies.
