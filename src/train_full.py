# ══════════════════════════════════════════════════════════════════════════════
#  train_full.py — Final Training Script
#  Model     : EfficientNetV2-S (best from experiments)
#  Hyperparams: lr=1e-4, img=256, dropout=0.2, batch=64 (best from tuning)
#  Data      : 40% stratified subset of full GrainSet (140K images)
#  Epochs    : 30 (fits in ~90 min on RTX 4090)
#  Features  : Auto-resume if electricity cuts, saves every 5 epochs
# ══════════════════════════════════════════════════════════════════════════════

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    'save_dir':         r'C:\GrainQuality\final_results',
    'img_size':         256,
    'lr':               1e-4,
    'dropout':          0.2,
    'batch_size':       64,
    'num_workers':      8,
    'num_classes':      7,
    'epochs':           30,
    'weight_decay':     0.01,
    'grad_clip':        1.0,
    'warmup_epochs':    3,
    'min_lr':           1e-6,
    'subset_fraction':  0.40,
    'seed':             42,
    'save_every':       5,     # save checkpoint every 5 epochs
}

# Grain paths — maize has nested folder
GRAIN_PATHS = {
    'maize': r'C:\GrainQuality\full_data\maize\maize',
    'rice':  r'C:\GrainQuality\full_data\rice',
    'sorg':  r'C:\GrainQuality\full_data\sorg',
    'wheat': r'C:\GrainQuality\full_data\wheat',
}

LABEL_MAP = {
    'NOR': 0, 'F&S': 1, 'SD': 2, 'MY': 3,
    'AP':  4, 'BN':  5,
    'HD':  6, 'UN':  6, 'BP': 6, 'IM': 6,
}
CLASS_NAMES = ['NOR', 'F&S', 'SD', 'MY', 'AP', 'BN', 'OTHER']


def get_class_from_folder(folder_name):
    parts = folder_name.split('_', 1)
    return parts[1] if len(parts) == 2 else folder_name


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class GrainDataset(Dataset):
    def __init__(self, split='train', transform=None,
                 subset_fraction=0.40, val_frac=0.1, seed=42):
        self.transform = transform
        rng = np.random.default_rng(seed)

        print(f"\nLoading {int(subset_fraction*100)}% subset...")
        all_samples = []

        for grain, grain_path in GRAIN_PATHS.items():
            if not os.path.isdir(grain_path):
                print(f"  ⚠️  {grain} not found — skipping")
                continue

            grain_samples = []
            for subfolder in ['train', 'test']:
                base = os.path.join(grain_path, subfolder)
                if not os.path.isdir(base):
                    continue
                for class_folder in sorted(os.listdir(base)):
                    class_path = os.path.join(base, class_folder)
                    if not os.path.isdir(class_path):
                        continue
                    class_name = get_class_from_folder(class_folder)
                    label = LABEL_MAP.get(class_name, 6)
                    for fname in os.listdir(class_path):
                        if fname.lower().endswith('.png'):
                            grain_samples.append(
                                (os.path.join(class_path, fname), label)
                            )

            # Stratified subset per grain
            grain_labels = np.array([s[1] for s in grain_samples])
            subset_idx = []
            for cls in np.unique(grain_labels):
                cls_idx = np.where(grain_labels == cls)[0]
                n_take = max(2, int(len(cls_idx) * subset_fraction))
                chosen = rng.choice(cls_idx, n_take, replace=False)
                subset_idx.extend(chosen.tolist())

            grain_subset = [grain_samples[i] for i in subset_idx]
            all_samples.extend(grain_subset)
            print(f"  {grain:6s} → {len(grain_subset):6d} / {len(grain_samples):6d} images")

        print(f"  TOTAL  → {len(all_samples):6d} images")

        if len(all_samples) == 0:
            raise RuntimeError("No images found! Check GRAIN_PATHS in config.")

        # Stratified train/val/test split
        labels_arr = np.array([s[1] for s in all_samples])
        train_idx, val_idx, test_idx = [], [], []
        for cls in np.unique(labels_arr):
            idx = np.where(labels_arr == cls)[0].copy()
            rng.shuffle(idx)
            n = len(idx)
            n_test = max(1, int(n * val_frac))
            n_val  = max(1, int(n * val_frac))
            test_idx .extend(idx[:n_test].tolist())
            val_idx  .extend(idx[n_test:n_test+n_val].tolist())
            train_idx.extend(idx[n_test+n_val:].tolist())

        chosen = {'train': train_idx, 'val': val_idx, 'test': test_idx}[split]
        self.samples = [all_samples[i] for i in chosen]
        self.labels  = [s[1] for s in self.samples]

        dist = {CLASS_NAMES[k]: v
                for k, v in sorted(Counter(self.labels).items())}
        print(f"  [{split:5s}] {len(self.samples):6d} images")
        print(f"           {dist}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(label)


def get_transforms(img_size=256, stage='train'):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if stage == 'train':
        return transforms.Compose([
            transforms.Resize((img_size+32, img_size+32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL + LOSS
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.ce    = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    def forward(self, inputs, targets):
        ce  = self.ce(inputs, targets)
        pt  = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()


class GrainModel(nn.Module):
    def __init__(self, num_classes=7, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=True,
            num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(512, num_classes)
        )
        print(f"Model      : EfficientNetV2-S")
        print(f"Features   : {feat_dim}")
        print(f"Parameters : {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        return self.classifier(self.backbone(x))


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / EVAL
# ══════════════════════════════════════════════════════════════════════════════
def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * (epoch+1) / warmup_epochs


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, cfg):
    model.train()
    warmup_lr(optimizer, epoch, cfg['warmup_epochs'], cfg['lr'])
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        total_loss += loss.item()
        all_preds .extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return total_loss/len(loader), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item()
        all_preds .extend(out.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    return total_loss/len(loader), acc, f1, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cfg    = CONFIG
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(cfg['save_dir'], exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  MilletSaarthi — Grain Quality Final Training")
    print(f"{'='*65}")
    print(f"  Device     : {device}")
    if device == 'cuda':
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM       : {vram:.1f} GB")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"  Batch size : {cfg['batch_size']}")
    print(f"  Image size : {cfg['img_size']}x{cfg['img_size']}")
    print(f"  LR         : {cfg['lr']}")
    print(f"  Data       : {int(cfg['subset_fraction']*100)}% of GrainSet")
    print(f"{'='*65}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = GrainDataset('train', get_transforms(cfg['img_size'], 'train'),
                            cfg['subset_fraction'])
    val_ds   = GrainDataset('val',   get_transforms(cfg['img_size'], 'val'),
                            cfg['subset_fraction'])
    test_ds  = GrainDataset('test',  get_transforms(cfg['img_size'], 'val'),
                            cfg['subset_fraction'])

    counts  = Counter(train_ds.labels)
    weights = [1.0 / counts[l] for l in train_ds.labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'],
                          sampler=sampler, num_workers=cfg['num_workers'],
                          pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=cfg['num_workers'],
                          pin_memory=True, persistent_workers=True)
    test_dl  = DataLoader(test_ds, batch_size=cfg['batch_size'],
                          shuffle=False, num_workers=cfg['num_workers'],
                          pin_memory=True, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GrainModel(cfg['num_classes'], cfg['dropout']).to(device)

    total = sum(counts.values())
    alpha = torch.tensor(
        [total / (cfg['num_classes'] * counts.get(i, 1))
         for i in range(cfg['num_classes'])],
        dtype=torch.float32).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=cfg['epochs']-cfg['warmup_epochs'],
                                  eta_min=cfg['min_lr'])

    # ── Auto-Resume from checkpoint ───────────────────────────────────────────
    start_epoch  = 0
    best_val_f1  = 0.0
    best_val_acc = 0.0
    history      = []

    resume_path = os.path.join(cfg['save_dir'], 'last_checkpoint.pth')
    if os.path.exists(resume_path):
        print(f"\n⚡ Resuming from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch  = ckpt['epoch']
        best_val_f1  = ckpt['best_val_f1']
        best_val_acc = ckpt['best_val_acc']
        history      = ckpt['history']
        print(f"   Resumed from epoch {start_epoch} | Best Val F1: {best_val_f1:.2f}%")

    # ── Training Loop ─────────────────────────────────────────────────────────
    start_time = time.time()

    print(f"\n{'Epoch':>5} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | "
          f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'LR':>8}")
    print('-' * 72)

    for epoch in range(start_epoch, cfg['epochs']):

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_dl, criterion, optimizer, device, epoch, cfg)
        va_loss, va_acc, va_f1, _, _ = evaluate(
            model, val_dl, criterion, device)

        if epoch >= cfg['warmup_epochs']:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        is_best    = va_f1 > best_val_f1

        print(f"{epoch+1:>5} | {tr_loss:>7.4f} | {tr_acc:>5.1f}% | "
              f"{tr_f1:>5.1f}% | {va_loss:>7.4f} | {va_acc:>5.1f}% | "
              f"{va_f1:>5.1f}% | {current_lr:>8.2e}"
              + (" ← BEST ✅" if is_best else ""))

        history.append({
            'epoch': epoch+1,
            'tr_loss': tr_loss, 'tr_acc': tr_acc, 'tr_f1': tr_f1,
            'va_loss': va_loss, 'va_acc': va_acc, 'va_f1': va_f1,
        })

        # Save best model
        if is_best:
            best_val_f1  = va_f1
            best_val_acc = va_acc
            torch.save({
                'epoch': epoch+1,
                'model': model.state_dict(),
                'val_f1': best_val_f1,
                'val_acc': best_val_acc,
            }, os.path.join(cfg['save_dir'], 'best_model.pth'))

        # Save checkpoint every 5 epochs (electricity protection)
        if (epoch + 1) % cfg['save_every'] == 0:
            torch.save({
                'epoch':        epoch+1,
                'model':        model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'scheduler':    scheduler.state_dict(),
                'best_val_f1':  best_val_f1,
                'best_val_acc': best_val_acc,
                'history':      history,
                'config':       cfg,
            }, resume_path)
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")

    # ── Final Test ────────────────────────────────────────────────────────────
    print(f"\nEvaluating best model on test set...")
    ckpt = torch.load(os.path.join(cfg['save_dir'], 'best_model.pth'),
                      map_location=device)
    model.load_state_dict(ckpt['model'])
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(
        model, test_dl, criterion, device)
    elapsed = (time.time() - start_time) / 60

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")
    print(f"  Best Val F1       : {best_val_f1:.2f}%")
    print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"  Test F1 (Macro)   : {te_f1:.2f}%")
    print(f"  Test Accuracy     : {te_acc:.2f}%")
    print(f"  Total Time        : {elapsed:.1f} min")
    print(f"{'='*65}")

    diff = te_f1 - 96.4
    if diff > 0:
        print(f"\n  ✅ BEATS PAPER by {diff:.2f}%!")
        print(f"     Our F1 : {te_f1:.2f}%")
        print(f"     Paper  : 96.4%")
    else:
        print(f"\n  ⚠️  Below paper by {abs(diff):.2f}%")

    print(f"\nPer-class breakdown:")
    print(classification_report(te_labels, te_preds,
                                target_names=CLASS_NAMES, zero_division=0))

    # Save results
    results = {
        'model':        'EfficientNetV2-S',
        'dataset':      'GrainSet 40% stratified',
        'best_val_f1':  best_val_f1,
        'best_val_acc': best_val_acc,
        'test_f1':      te_f1,
        'test_acc':     te_acc,
        'time_min':     elapsed,
        'beats_paper':  diff > 0,
        'improvement':  diff,
        'history':      history,
    }
    with open(os.path.join(cfg['save_dir'], 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Training curves
    epochs_list = [h['epoch'] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('EfficientNetV2-S — GrainSet Training', fontweight='bold')
    axes[0].plot(epochs_list, [h['tr_f1'] for h in history], label='Train F1', color='#2196F3')
    axes[0].plot(epochs_list, [h['va_f1'] for h in history], label='Val F1',   color='#4CAF50')
    axes[0].axhline(y=96.4, color='red', linestyle='--', label='Paper baseline (96.4%)')
    axes[0].set_title('Macro F1'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs_list, [h['tr_acc'] for h in history], label='Train Acc', color='#2196F3')
    axes[1].plot(epochs_list, [h['va_acc'] for h in history], label='Val Acc',   color='#4CAF50')
    axes[1].axhline(y=96.4, color='red', linestyle='--', label='Paper baseline')
    axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['save_dir'], 'training_curves.png'), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(te_labels, te_preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Confusion Matrix — Test Set', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg['save_dir'], 'confusion_matrix.png'), dpi=150)
    plt.close()

    print(f"\n  Saved to: {cfg['save_dir']}")
    print(f"    best_model.pth")
    print(f"    final_results.json")
    print(f"    training_curves.png")
    print(f"    confusion_matrix.png")
    print(f"\n  Done! 🎉")


if __name__ == '__main__':
    main()
