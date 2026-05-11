"""
═══════════════════════════════════════════════════════════════════════════
U-Net++ TRAINING — VERSION 5.0 (MULTI-FORMAT DATASET SUPPORT)
═══════════════════════════════════════════════════════════════════════════

ALL IMPROVEMENTS (v5.0 on top of v4.0):
  ✓ Two separate dataset instances (val transform bug fix)
  ✓ Dice metric separately for disc and cup
  ✓ weighted_dice_focal — cup has 2x weight (smaller, harder)
  ✓ Mixed Precision Training (torch.amp)
  ✓ Gradient clipping
  ✓ Linear Warmup + Cosine Annealing
  ✓ Early stopping
  ✓ Weight decay in optimizer
  ✓ CLAHE + fundus-specific augmentations
  ✓ Fixed GaussNoise and ShiftScaleRotate (no deprecation warnings)
  ✓ Morphological post-processing (function ready for predict.py)
  ✓ [NEW] Support for two dataset layouts (see DATA LAYOUTS below)
  ✓ [NEW] Automatic resize — images don't have to be 512×512

DATA LAYOUTS SUPPORTED:
──────────────────────────────────────────────────────────────────────────

  Layout A — separate masks:
    full/          ← original eye images (.jpg / .png)
    masks/
      cup/         ← binary mask: <stem>_cup.png   (255 = cup)
      disc/        ← binary mask: <stem>_disc.png  (255 = disc)

  Layout B — merged mask (original / legacy):
    full2/         ← original eye images (.jpg / .png)
    masks2/        ← single grayscale mask: <stem>.png
                      pixel value 0 = background
                      pixel value 1 = disc
                      pixel value 2 = cup

  You can mix both layouts — just configure all four paths.
  If a path is None / empty the layout is skipped.

MASKS (internal representation): 0=background, 1=disc, 2=cup
═══════════════════════════════════════════════════════════════════════════
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold
import random

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# ── Layout A  (single masks folder, files: <stem>_cup.png / <stem>_disc.png)
# Set to None to disable this layout.
IMAGES_DIR_A = "../../../../data/full"   # images
MASKS_DIR_A  = "../../../../data/Masks"  # <stem>_cup.png + <stem>_disc.png

# ── Layout B  (merged mask: 0=bg, 1=disc, 2=cup) ──────────────────────────
# Set to None to disable this layout.
IMAGES_DIR_B   = "../../../../data/full2"     # images
MASKS_DIR_B    = "../../../../data/Masks2"    # <stem>.png

# ── General ───────────────────────────────────────────────────────────────
SAVE_DIR   = "models"

EPOCHS                  = 100
BATCH_SIZE              = 4       # T4 + AMP + UNet++ 512px
IMAGE_SIZE              = 512
LEARNING_RATE           = 0.0001
WARMUP_EPOCHS           = 5
EARLY_STOPPING_PATIENCE = 20
ENCODER                 = "efficientnet-b4"
PRETRAINED_PATH         = None    # path to .pth file to fine-tune, or None
USE_KFOLD               = False
N_FOLDS                 = 5
# "weighted_dice_focal" = cup 2x more important (RECOMMENDED)
# "dice_focal"          = equal weights
# "tversky"             = penalizes FN (missing glaucoma)
LOSS_TYPE               = "weighted_dice_focal"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:         {DEVICE}")
print(f"Epochs:         {EPOCHS} (early stopping after {EARLY_STOPPING_PATIENCE})")
print(f"Batch size:     {BATCH_SIZE}")
print(f"Loss function:  {LOSS_TYPE}")
print(f"Pretrained:     {'Fine-tuning: ' + PRETRAINED_PATH if PRETRAINED_PATH else 'From scratch (ImageNet encoder)'}")


# ═══════════════════════════════════════════════════════════════════════════
# MASK FILE HELPERS  (shared by both layouts)
# ═══════════════════════════════════════════════════════════════════════════

_MASK_EXTS = [".png", ".bmp", ".tif", ".tiff"]

def _find_mask(masks_dir: Path, stem: str):
    """Return the first existing mask file for *stem* (any supported extension), or None."""
    for ext in _MASK_EXTS:
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
# DATASET — LAYOUT A  (separate cup/disc masks)
# ═══════════════════════════════════════════════════════════════════════════

class GlaucomaDatasetSeparateMasks(Dataset):
    """
    Folder structure:
        images_dir/  ← <stem>.jpg / .png / .tif / …
        masks_dir/   ← <stem>_cup.png   (binary, 255 = cup region)
                       <stem>_disc.png  (binary, 255 = disc region)
                       (both files live in the SAME folder)
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.JPG")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.bmp")) +
            list(self.images_dir.glob("*.tif")) +
            list(self.images_dir.glob("*.tiff"))
        )
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}!")

        # Keep only samples that have BOTH masks in the shared folder
        valid = []
        missing = []
        for f in self.image_files:
            cup_path  = _find_mask(self.masks_dir, f"{f.stem}_cup")
            disc_path = _find_mask(self.masks_dir, f"{f.stem}_disc")
            if cup_path is not None and disc_path is not None:
                valid.append(f)
            else:
                missing.append(f.name)

        if missing:
            print(f"  [Layout A] WARNING: {len(missing)} image(s) skipped "
                  f"(missing _cup/_disc mask): {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")

        self.image_files = valid
        print(f"  [Layout A] {len(self.image_files)} valid samples "
              f"(images + _cup + _disc masks)")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path  = self.image_files[idx]
        image     = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        cup_path  = _find_mask(self.masks_dir, f"{img_path.stem}_cup")
        disc_path = _find_mask(self.masks_dir, f"{img_path.stem}_disc")

        cup_mask  = cv2.imread(str(cup_path),  cv2.IMREAD_GRAYSCALE)
        disc_mask = cv2.imread(str(disc_path), cv2.IMREAD_GRAYSCALE)

        # Build combined mask: 0=bg, 1=disc, 2=cup
        # Disc first, then cup overwrites (cup ⊂ disc anatomically)
        h, w  = image.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        mask[disc_mask > 127] = 1
        mask[cup_mask  > 127] = 2

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        return image, _mask_to_tensor(mask)


# ═══════════════════════════════════════════════════════════════════════════
# MASK FORMAT AUTO-DETECTION & NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════

def normalize_merged_mask(mask: np.ndarray) -> np.ndarray:
    """
    Accepts two mask formats and returns a normalised mask: 0=bg, 1=disc, 2=cup.

    Format A — standard (already normalised):
        pixel values: {0, 1, 2}   →  returned as-is

    Format B — visual grayscale (e.g. exported from image editors):
        white  (> 200)  = background   → 0
        gray   (50-200) = disc          → 1
        black  (< 50)   = cup           → 2

    Detection rule: if max(mask) > 2 we assume Format B.
    """
    if mask.max() <= 2:
        return mask  # already in standard format

    # ── Format B: remap visual grayscale → 0/1/2 ──────────────────────────
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask > 200]               = 0   # white  → background
    out[(mask >= 50) & (mask <= 200)] = 1   # gray   → disc
    out[mask < 50]                = 2   # black  → cup
    return out


# ═══════════════════════════════════════════════════════════════════════════
# DATASET — LAYOUT B  (merged mask: 0=bg, 1=disc, 2=cup)
# ═══════════════════════════════════════════════════════════════════════════

class GlaucomaDatasetMergedMask(Dataset):
    """
    Folder structure:
        images_dir/  ← <stem>.jpg or <stem>.png
        masks_dir/   ← <stem>.png  (grayscale: 0=bg, 1=disc, 2=cup)
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.transform  = transform

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.JPG")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.bmp")) +
            list(self.images_dir.glob("*.tif")) +
            list(self.images_dir.glob("*.tiff"))
        )
        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}!")

        # Keep only samples that have a matching mask (any supported extension)
        valid   = []
        missing = []
        for f in self.image_files:
            mask_path = _find_mask(self.masks_dir, f.stem)
            if mask_path is not None:
                valid.append(f)
            else:
                missing.append(f.name)

        if missing:
            print(f"  [Layout B] WARNING: {len(missing)} image(s) skipped "
                  f"(no mask): {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")

        self.image_files = valid
        print(f"  [Layout B] {len(self.image_files)} valid samples "
              f"(images + merged mask)")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path  = self.image_files[idx]
        image     = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        mask_path = _find_mask(self.masks_dir, img_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for: {img_path.name}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = normalize_merged_mask(mask)

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        return image, _mask_to_tensor(mask)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED MASK → TENSOR HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _mask_to_tensor(mask):
    """
    Converts a combined mask (0=bg, 1=disc, 2=cup) — either np.ndarray or
    torch.Tensor — into two binary channels:
        channel 0 → disc  (pixels == 1)
        channel 1 → cup   (pixels == 2)
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.long)
    else:
        mask = torch.round(mask.float()).long()

    return torch.stack([
        (mask == 1).float(),   # channel 0: disc
        (mask == 2).float(),   # channel 1: cup
    ], dim=0)


# ═══════════════════════════════════════════════════════════════════════════
# BUILD THE COMBINED DATASET (both layouts together)
# ═══════════════════════════════════════════════════════════════════════════

def build_datasets(train_transform, val_transform):
    """
    Returns a pair (train_ds, val_ds) combining all enabled layouts.
    Each layout is instantiated TWICE (once per transform) to avoid the
    val-transform bug where both splits share the same augmentation object.
    """
    train_parts = []
    val_parts   = []

    # ── Layout A ──────────────────────────────────────────────────────────
    if IMAGES_DIR_A and MASKS_DIR_A:
        p = Path(IMAGES_DIR_A)
        if p.exists():
            print("\n[Layout A] Loading separate-mask dataset …")
            train_parts.append(GlaucomaDatasetSeparateMasks(
                IMAGES_DIR_A, MASKS_DIR_A,
                transform=train_transform))
            val_parts.append(GlaucomaDatasetSeparateMasks(
                IMAGES_DIR_A, MASKS_DIR_A,
                transform=val_transform))
        else:
            print(f"[Layout A] Skipped — path not found: {IMAGES_DIR_A}")

    # ── Layout B ──────────────────────────────────────────────────────────
    if IMAGES_DIR_B and MASKS_DIR_B:
        p = Path(IMAGES_DIR_B)
        if p.exists():
            print("\n[Layout B] Loading merged-mask dataset …")
            train_parts.append(GlaucomaDatasetMergedMask(
                IMAGES_DIR_B, MASKS_DIR_B,
                transform=train_transform))
            val_parts.append(GlaucomaDatasetMergedMask(
                IMAGES_DIR_B, MASKS_DIR_B,
                transform=val_transform))
        else:
            print(f"[Layout B] Skipped — path not found: {IMAGES_DIR_B}")

    if not train_parts:
        raise RuntimeError("No dataset could be loaded! "
                           "Check IMAGES_DIR_A / IMAGES_DIR_B paths.")

    # ConcatDataset merges multiple datasets transparently
    if len(train_parts) == 1:
        return train_parts[0], val_parts[0]

    return ConcatDataset(train_parts), ConcatDataset(val_parts)


# ═══════════════════════════════════════════════════════════════════════════
# AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_train_transforms():
    return A.Compose([
        # Resize handles any input resolution → always outputs IMAGE_SIZE×IMAGE_SIZE
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # ── Geometric ─────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5
        ),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),

        # ── General photometric ───────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(0.0, 0.001), p=0.3),

        # ── Fundus-specific ───────────────────────────────────────────────
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            fill=0, p=0.2
        ),

        # ── Normalization ─────────────────────────────────────────────────
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# ═══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_criterion(loss_type):
    """
    weighted_dice_focal:
        Cup gets 2x more weight — it's smaller (~4x), harder
        and directly determines the quality of CDR (glaucoma indicator).

    dice_focal:
        Equal weights. Stable alternative.

    tversky:
        alpha=0.3, beta=0.7 → penalizes FN (misses) more strongly.
        Clinical rationale: an unnoticed glaucoma is more dangerous
        than a false alarm.
    """
    if loss_type == "weighted_dice_focal":
        dice_b  = smp.losses.DiceLoss(mode='binary')
        focal_b = smp.losses.FocalLoss(mode='binary')

        def criterion(outputs, masks):
            d_out = outputs[:, 0:1].contiguous()
            d_msk = masks[:, 0:1].contiguous()
            c_out = outputs[:, 1:2].contiguous()
            c_msk = masks[:, 1:2].contiguous()

            loss_disc = 0.5 * dice_b(d_out, d_msk) + 0.5 * focal_b(d_out, d_msk)
            loss_cup  = 0.5 * dice_b(c_out, c_msk) + 0.5 * focal_b(c_out, c_msk)

            # Cup: weight 2.0, disc: weight 1.0 → divided by 3.0 for normalization
            return (loss_disc * 1.0 + loss_cup * 2.0) / 3.0

        return criterion

    elif loss_type == "tversky":
        return smp.losses.TverskyLoss(mode='multilabel', alpha=0.3, beta=0.7)

    else:  # dice_focal
        dice_fn  = smp.losses.DiceLoss(mode='multilabel')
        focal_fn = smp.losses.FocalLoss(mode='multilabel')

        def criterion(outputs, masks):
            return 0.5 * dice_fn(outputs, masks) + 0.5 * focal_fn(outputs, masks)

        return criterion


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER — Linear Warmup + Cosine Annealing
# ═══════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, epochs, steps_per_epoch):
    """
    OneCycleLR combines warmup and cosine annealing smoothly and updates
    every batch instead of every epoch. It increases LR to max_lr, then
    anneals it down. Supports differential learning rates natively.
    """
    max_lrs = [group['lr'] for group in optimizer.param_groups]

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% of training spent warming up
    )


# ═══════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════

def dice_scores(pred_logits, target, smooth=1e-6):
    """Returns (dice_disc, dice_cup, dice_mean) — each class separately."""
    pred = (torch.sigmoid(pred_logits) > 0.5).float()

    def _d(p, t):
        i = (p * t).sum()
        u = p.sum() + t.sum()
        return ((2.0 * i + smooth) / (u + smooth)).item()

    d = _d(pred[:, 0], target[:, 0])
    c = _d(pred[:, 1], target[:, 1])
    return d, c, (d + c) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# POST-PROCESSING (to be used in predict.py)
# ═══════════════════════════════════════════════════════════════════════════

def morphological_postprocess(pred_disc_np, pred_cup_np):
    """
    1. Morphological Closing  → fills holes in masks
    2. Largest Component      → removes isolated artifacts
    3. Cup ⊂ Disc constraint  → cup must lie inside the disc (anatomy)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def clean(mask):
        m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            m = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        return m.astype(np.float32)

    disc = clean(pred_disc_np)
    cup  = clean(pred_cup_np) * disc
    return disc, cup


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LOOPS
# ═══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    tot_loss = tot_disc = tot_cup = 0.0
    use_amp  = (device == 'cuda')

    pbar = tqdm(loader, desc="[TRAIN]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            out  = model(images)
            loss = criterion(out, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        d, c, _ = dice_scores(out, masks)
        tot_loss += loss.item(); tot_disc += d; tot_cup += c
        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         disc=f'{d:.4f}', cup=f'{c:.4f}')

    n = len(loader)
    return tot_loss/n, tot_disc/n, tot_cup/n


def validate(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_disc = tot_cup = 0.0
    use_amp  = (device == 'cuda')

    with torch.no_grad():
        pbar = tqdm(loader, desc="[VAL]  ", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(images)
                loss = criterion(out, masks)
            d, c, _ = dice_scores(out, masks)
            tot_loss += loss.item(); tot_disc += d; tot_cup += c
            pbar.set_postfix(disc=f'{d:.4f}', cup=f'{c:.4f}')

    n = len(loader)
    return tot_loss/n, tot_disc/n, tot_cup/n


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_single_model(train_idx, val_idx, train_ds, val_ds,
                       save_name, epochs):

    train_loader = DataLoader(
        Subset(train_ds, train_idx), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=(DEVICE == 'cuda')
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=(DEVICE == 'cuda')
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, encoder_weights="imagenet",
        in_channels=3, classes=2, activation=None
    ).to(DEVICE)

    # ── Optional fine-tuning ──────────────────────────────────────────────
    if PRETRAINED_PATH and Path(PRETRAINED_PATH).exists():
        ck = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        print(f"  Loaded weights: {Path(PRETRAINED_PATH).name} "
              f"(previous Dice: {ck.get('dice', 0):.4f})")
    else:
        print("  Training from scratch (encoder pretrained on ImageNet)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {n_params:.1f}M")

    # ── Differential LR: encoder 10x slower than decoder ──────────────────
    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters(),
         'lr': LEARNING_RATE * 0.1},
        {'params': model.decoder.parameters(),
         'lr': LEARNING_RATE},
        {'params': model.segmentation_head.parameters(),
         'lr': LEARNING_RATE},
    ], weight_decay=1e-5)

    criterion = build_criterion(LOSS_TYPE)
    scheduler = build_scheduler(optimizer, epochs, len(train_loader))
    scaler    = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    best_dice      = 0.0
    no_improve_cnt = 0
    save_path      = Path(SAVE_DIR) / f"{save_name}.pth"

    for epoch in range(1, epochs + 1):
        lr_now = optimizer.param_groups[1]['lr']
        print(f"\nEpoch {epoch}/{epochs}  |  LR: {lr_now:.6f}")
        print("-" * 65)

        tr_loss, tr_disc, tr_cup = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE)
        va_loss, va_disc, va_cup = validate(
            model, val_loader, criterion, DEVICE)

        tr_mean = (tr_disc + tr_cup) / 2.0
        va_mean = (va_disc + va_cup) / 2.0

        print(f"Train → Loss: {tr_loss:.4f} | "
              f"Disc: {tr_disc:.4f} | Cup: {tr_cup:.4f} | Avg: {tr_mean:.4f}")
        print(f"Val   → Loss: {va_loss:.4f} | "
              f"Disc: {va_disc:.4f} | Cup: {va_cup:.4f} | Avg: {va_mean:.4f}")

        if va_mean > best_dice:
            best_dice      = va_mean
            no_improve_cnt = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice':                 va_mean,
                'dice_disc':            va_disc,
                'dice_cup':             va_cup,
                'encoder':              ENCODER,
                'loss_type':            LOSS_TYPE,
            }, save_path)
            print(f"  ✓ BEST!  Disc: {va_disc:.4f} | "
                  f"Cup: {va_cup:.4f} | Avg: {va_mean:.4f}  → {save_path.name}")
        else:
            no_improve_cnt += 1
            print(f"  ✗ No improvement ({no_improve_cnt}/{EARLY_STOPPING_PATIENCE})")
            if no_improve_cnt >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚑ Early stopping — no improvement for "
                      f"{EARLY_STOPPING_PATIENCE} epochs.")
                break

    return best_dice


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("=" * 65)
    print("U-Net++ v5.0  |  Target: Disc ≥ 93%  Cup ≥ 88%  Avg ≥ 90%")
    print("=" * 65)

    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")
    # CRITICAL: build_datasets creates SEPARATE instances per transform
    # so validation always uses val_transforms (no augmentation leakage)
    train_ds, val_ds = build_datasets(
        get_train_transforms(), get_val_transforms()
    )

    total = len(train_ds)
    print(f"\n  Total samples across all layouts: {total}")
    all_idx = list(range(total))

    if not USE_KFOLD:
        # ── Single 80/20 split ────────────────────────────────────────────
        rng = random.Random(42)
        idx = all_idx.copy()
        rng.shuffle(idx)
        cut       = int(0.8 * len(idx))
        train_idx = idx[:cut]
        val_idx   = idx[cut:]

        print(f"\nSplit 80/20  |  Train: {len(train_idx)}  Val: {len(val_idx)}")
        print("=" * 65)

        best = train_single_model(
            train_idx, val_idx, train_ds, val_ds,
            save_name="unetpp_best_v5", epochs=EPOCHS
        )

        print("\n" + "=" * 65)
        print("TRAINING COMPLETED!")
        print(f"Best Val Dice (average): {best:.4f} ({best*100:.1f}%)")
        print(f"Model: {SAVE_DIR}/unetpp_best_v5.pth")
        if best >= 0.90:
            print("✓ 90% target achieved!")
        else:
            print(f"✗ Missing to 90% target: {(0.90 - best)*100:.1f}%")
        print("=" * 65)

    else:
        # ── K-Fold Cross Validation ───────────────────────────────────────
        print(f"\n{N_FOLDS}-Fold CV  |  Each fold = up to {EPOCHS} epochs")
        print("=" * 65)

        kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        results = []

        for fold, (tr, va) in enumerate(kf.split(all_idx), 1):
            print(f"\n{'='*65}")
            print(f"FOLD {fold}/{N_FOLDS}  "
                  f"Train: {len(tr)}  Val: {len(va)}")
            print("=" * 65)
            best = train_single_model(
                tr.tolist(), va.tolist(), train_ds, val_ds,
                save_name=f"unetpp_v5_fold{fold}", epochs=EPOCHS
            )
            results.append(best)
            print(f"Fold {fold} Best: {best:.4f}")

        print("\n" + "=" * 65)
        print("K-FOLD COMPLETED!")
        for i, d in enumerate(results, 1):
            print(f"  Fold {i}: {d:.4f} ({d*100:.1f}%)")
        import numpy as np
        print(f"  Average: {np.mean(results):.4f} ± {np.std(results):.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main()