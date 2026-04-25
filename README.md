# FACH — Frequency-domain Adversarial Cross-modal Hashing

> **Frequency-Domain Adversarial Robustness Evaluation for Deep Cross-Modal Hashing Systems**

---

## Overview

FACH is a **black-box adversarial attack framework** targeting cross-modal hashing retrieval systems. It operates in two stages:

**Stage (a) — Substitute Model Training**
- Extracts a frequency-sensitivity consensus matrix from multiple teacher models.
- Trains a substitute model via boundary-constrained distillation loss to achieve semantic consistency.
- TextNet is first pre-trained with cross-modal hashing loss and then frozen; ImgNet is jointly trained with a learnable weight vector.

**Stage (b) — Frequency-Domain Adversarial Example Generation**
- Performs PGD iterations in the DCT frequency domain, guided by the consensus matrix as the spectral prior.
- Uses a global semantic hash code as the targeted attack anchor.
- A hinge-based adversarial loss forces hash codes across the zero boundary.

---

## File Structure

```
FACH/
├── main.py              # Main entry point: train / attack / run
├── config.py            # All hyperparameters
├── frequency.py         # Differentiable 2D DCT/IDCT & sensitivity computation
├── losses.py            # L_align, L_ME, L_distill, L_adv, L_t1–L_t4
├── attack.py            # Frequency-domain PGD attack + global semantic hash code generation
├── teacher_loader.py    # Teacher model loading (DADH/UCCH) + TeacherWeights
├── train_substitute.py  # Standalone substitute model training script (CLI-friendly)
├── utils.py             # mAP / t-mAP evaluation
├── models/
│   ├── img_net.py       # Image substitute model (MLP or VGG11 backbone)
│   └── txt_net.py       # Text substitute model (3-layer MLP)
├── datasets/
│   ├── data_handler.py  # .mat file loading (FLICKR-25K / NUS-WIDE)
│   └── dataset.py       # PyTorch Dataset with train/query/db splits
└── checkpoints/
    ├── substitute/      # Trained substitute model weights
    │   └── mirflickr25k_64/
    │       ├── ImgNet.pth          # Image substitute model (trained)
    │       └── TxtNet.pth          # Text substitute model (trained)
    └── teachers/        # Teacher model weights
        └── UCCH/
            ├── UCCH_AlexNet_64_mirflickr25k.pth
            ├── UCCH_VGG11_64_mirflickr25k.pth
            └── UCCH_DN161_64_mirflickr25k.pth
```

---

## Requirements

```bash
pip install torch torchvision scipy numpy tqdm fire h5py
```

- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA is recommended

---

## Data Preparation

Place the dataset `.mat` files under the `./data/` directory:

| Dataset | Directory | Key Files |
|---------|-----------|-----------|
| FLICKR-25K | `data/MIRFLICKR25K/` | `mirflickr25k-iall-vgg.mat` (XAll), `mirflickr25k-yall.mat`, `mirflickr25k-lall.mat` |
| NUS-WIDE-TC10 | `data/NUS-WIDE-TC10/` | `nus-wide-tc10-xall-vgg.mat`, `nus-wide-tc10-yall.mat`, `nus-wide-tc10-lall.mat` |

Dataset splits (as in Table I of the paper):

| Dataset | Training Set | Query Set | Database |
|---------|-------------|-----------|----------|
| FLICKR-25K | 5,000 | 2,000 | 18,015 |
| NUS-WIDE-TC10 | 10,500 | 2,100 | 193,734 |

---

## Usage

### Phase 1: Train the Substitute Model

```bash
cd FACH

# Without teachers (self-supervised fallback)
python main.py train dataset=mirflickr25k bit=64 device=cuda:0

# With teacher models (teacher weights must be prepared in advance)
python train_substitute.py \
    --dataset mirflickr25k \
    --bit 64 \
    --device cuda:0 \
    --teacher_dir ./checkpoints/teachers
```

### Phase 2: Adversarial Attack Evaluation

```bash
python main.py attack \
    dataset=mirflickr25k \
    bit=64 \
    device=cuda:0 \
    "teacher_configs=[{'type':'UCCH','ckpt_path':'./checkpoints/teachers/UCCH/UCCH_AlexNet_64_mirflickr25k.pth','overrides':{}}]"
```

### One-Command Full Pipeline (Both Stages)

```bash
python main.py run dataset=mirflickr25k bit=64 device=cuda:0
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bit` | 64 | Hash code length (supports 16 / 32 / 64 / 128) |
| `margin` | 1.0 | Margin in the max-entropy loss |
| `madv` | 1.2 | Hinge margin in the adversarial loss |
| `tau_freq` | 20 | Low-frequency mask threshold |
| `T` | 100 | Number of PGD iterations |
| `mu` | 0.001 | PGD step size |
| `delta` | 0.3 | Frequency-domain perturbation clipping range |
| `eps` | 8/255 | Spatial L∞ perturbation budget |
| `lr` | 1e-4 | Adam learning rate |
| `max_epoch` | 20 | Substitute model training epochs |
| `pretrain_epochs` | 5 | TextNet pre-training epochs |
| `sensitivity_loss` | `lt2` | Frequency sensitivity signal: `lt1` / `lt2` / `lt3` / `lt4` |

---

## Teacher Model Weight Naming Convention

```
checkpoints/teachers/<METHOD>/<METHOD>_<BACKBONE>_<BIT>_<DATASET>.pth
```

**Examples:**

```
checkpoints/teachers/DADH/DADH_AlexNet_64_mirflickr25k.pth
checkpoints/teachers/UCCH/UCCH_VGG11_64_mirflickr25k.pth
```

Supported methods: `DADH`, `UCCH` (each with 6 backbones: AlexNet / VGG11 / RN50 / RN152 / IncV3 / DN161)
