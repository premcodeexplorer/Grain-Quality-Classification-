import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch
import numpy as np
from collections import Counter

# ─── LABEL MAP ───────────────────────────────────────────────────────────────
# Folder names follow pattern: 0_NOR, 1_F&S, 2_SD etc.
# We strip the number prefix and map class name to our unified label.
# HD (5 images), UN (42), BP (29), IM (75-91) → merged into OTHER
# This is for tiny dataset only. Full dataset keeps all separate.

FOLDER_TO_LABEL = {
    'NOR': 0,   # Normal
    'F&S': 1,   # Fusarium & Shriveled
    'SD':  2,   # Sprouted
    'MY':  3,   # Moldy
    'AP':  4,   # Attacked by Pests
    'BN':  5,   # Broken
    'HD':  6,   # Heated      ─┐
    'UN':  6,   # Unripe       ├─ OTHER (too few in tiny)
    'BP':  6,   # Black Point  │
    'IM':  6,   # Impurity    ─┘
}

CLASS_NAMES = ['NOR', 'F&S', 'SD', 'MY', 'AP', 'BN', 'OTHER']
NUM_CLASSES  = 7   # 7 unified classes across all 4 grains

GRAINS = ['maize', 'rice', 'sorg', 'wheat']


def get_class_from_folder(folder_name):
    """
    '0_NOR' → 'NOR'
    '1_F&S' → 'F&S'
    '6_HD'  → 'HD'
    """
    # Strip leading number and underscore: '0_NOR' → 'NOR'
    parts = folder_name.split('_', 1)
    return parts[1] if len(parts) == 2 else folder_name


# ─── DATASET ─────────────────────────────────────────────────────────────────
class GrainDataset(Dataset):
    """
    Loads all 4 grains (maize, rice, sorg, wheat) from folder structure:

    tiny_data/
    ├── maize/
    │   ├── train/
    │   │   ├── 0_NOR/   ← PNG images inside
    │   │   ├── 1_F&S/
    │   │   └── ...
    │   └── test/
    │       └── ...
    ├── rice/  (same structure)
    ├── sorg/  (same structure)
    └── wheat/ (same structure)
    """

    def __init__(self, data_root, split='train', transform=None,
                 val_frac=0.1, seed=42):
        self.transform = transform
        self.samples   = []   # list of (img_path, label)

        all_samples = []

        print(f"\nScanning dataset...")
        for grain in GRAINS:
            grain_total = 0
            # Combine train + test folders from disk, then re-split ourselves
            for subfolder in ['train', 'test']:
                base = os.path.join(data_root, grain, subfolder)
                if not os.path.isdir(base):
                    continue

                for class_folder in sorted(os.listdir(base)):
                    class_path = os.path.join(base, class_folder)
                    if not os.path.isdir(class_path):
                        continue

                    class_name = get_class_from_folder(class_folder)
                    label      = FOLDER_TO_LABEL.get(class_name, 6)

                    for fname in os.listdir(class_path):
                        if fname.lower().endswith('.png'):
                            img_path = os.path.join(class_path, fname)
                            all_samples.append((img_path, label))
                            grain_total += 1

            print(f"  {grain:6s} → {grain_total:4d} images loaded")

        print(f"  {'TOTAL':6s} → {len(all_samples):4d} images\n")

        if len(all_samples) == 0:
            raise RuntimeError(
                "No images found! Check DATA_ROOT path.\n"
                f"Looking in: {data_root}"
            )

        # ── Stratified split into train / val / test ──────────────────────
        rng        = np.random.default_rng(seed)
        labels_arr = np.array([s[1] for s in all_samples], dtype=int)

        train_idx, val_idx, test_idx = [], [], []

        for cls in np.unique(labels_arr):
            idx = np.where(labels_arr == cls)[0].copy()
            rng.shuffle(idx)
            n      = len(idx)
            n_val  = max(1, int(n * val_frac))
            n_test = max(1, int(n * val_frac))   # same fraction for test
            test_idx .extend(idx[:n_test].tolist())
            val_idx  .extend(idx[n_test:n_test + n_val].tolist())
            train_idx.extend(idx[n_test + n_val:].tolist())

        split_map = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        chosen    = split_map[split]

        self.samples = [all_samples[i] for i in chosen]
        self.labels  = [s[1] for s in self.samples]

        # Print distribution
        dist = dict(sorted(Counter(self.labels).items()))
        dist_str = {CLASS_NAMES[k]: v for k, v in dist.items()}
        print(f"  [{split:5s}] {len(self.samples):4d} images")
        print(f"           {dist_str}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)


# ─── TRANSFORMS ──────────────────────────────────────────────────────────────
def get_transforms(img_size=224, stage='train'):
    mean = [0.485, 0.456, 0.406]   # ImageNet
    std  = [0.229, 0.224, 0.225]

    if stage == 'train':
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:   # val / test — no augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─── WEIGHTED SAMPLER (handles class imbalance) ───────────────────────────────
def get_sampler(dataset):
    counts  = Counter(dataset.labels)
    weights = [1.0 / counts[l] for l in dataset.labels]
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )


# ─── DATALOADERS ─────────────────────────────────────────────────────────────
def get_dataloaders(data_root, img_size=224, batch_size=16, num_workers=2):
    train_ds = GrainDataset(data_root, 'train', get_transforms(img_size, 'train'))
    val_ds   = GrainDataset(data_root, 'val',   get_transforms(img_size, 'val'))
    test_ds  = GrainDataset(data_root, 'test',  get_transforms(img_size, 'val'))

    sampler  = get_sampler(train_ds)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_dl, val_dl, test_dl, train_ds


# ─── QUICK TEST ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_ROOT = r'E:\GrainQuality\tiny_data'

    train_dl, val_dl, test_dl, train_ds = get_dataloaders(
        DATA_ROOT, img_size=224, batch_size=16
    )

    imgs, labels = next(iter(train_dl))
    print(f"\nBatch shape  : {imgs.shape}")
    print(f"Label sample : {labels.tolist()}")
    print(f"Classes      : {CLASS_NAMES}")
    print(f"\ndataset.py working correctly ✅")