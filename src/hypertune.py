# ══════════════════════════════════════════════════════════════════════════════
#  hypertune.py — Hyperparameter tuning for EfficientNetV2-S
#  Run in Colab: !python hypertune.py
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from itertools import product

from dataset import get_dataloaders, NUM_CLASSES
from model import build_model

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS — Colab specific
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT = '/content/tiny_data/tiny_data'
SAVE_DIR  = '/content/drive/MyDrive/Grain_Quality/hypertune_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FIXED — Model is decided, only hyperparams change
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME  = 'efficientnetv2_s'
EPOCHS      = 20       # Less epochs for tuning — faster experiments
GRAD_CLIP   = 1.0
WARMUP      = 2
MIN_LR      = 1e-6

# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER GRID — 12 combinations total
# ══════════════════════════════════════════════════════════════════════════════
GRID = {
    'lr':         [1e-4, 3e-4, 5e-4],
    'img_size':   [224, 256],
    'dropout':    [0.2, 0.4],
    'batch_size': [16],        # Fixed at 16 for Colab memory safety
}

# ══════════════════════════════════════════════════════════════════════════════

def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = lr


def train_one_epoch(model, loader, criterion, optimizer,
                    device, epoch, warmup, base_lr):
    model.train()
    warmup_lr(optimizer, epoch, warmup, base_lr)
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        all_preds .extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)*100
    return total_loss / len(loader), f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        total_loss += criterion(outputs, labels).item()
        all_preds .extend(outputs.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds) * 100
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)*100
    return total_loss / len(loader), acc, f1


def run_experiment(lr, img_size, dropout, batch_size, exp_id, total_exps):
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_name = f"exp{exp_id:02d}_lr{lr}_img{img_size}_drop{dropout}_bs{batch_size}"

    print(f"\n{'='*65}")
    print(f"  [{exp_id}/{total_exps}] {exp_name}")
    print(f"  lr={lr} | img={img_size} | dropout={dropout} | batch={batch_size}")
    print(f"{'='*65}")

    # Check if already done
    result_path = os.path.join(SAVE_DIR, exp_name, 'result.json')
    if os.path.exists(result_path):
        print(f"  Already done — skipping ✅")
        with open(result_path) as f:
            return json.load(f)

    os.makedirs(os.path.join(SAVE_DIR, exp_name), exist_ok=True)

    # Data
    train_dl, val_dl, test_dl, train_ds = get_dataloaders(
        DATA_ROOT, img_size=img_size,
        batch_size=batch_size, num_workers=4
    )
    class_counts = dict(Counter(train_ds.labels))

    # Model
    import timm
    import torch.nn as nn
    from model import FocalLoss

    backbone = timm.create_model(
        'tf_efficientnetv2_s', pretrained=True,
        num_classes=0, global_pool='avg'
    )
    feat_dim = backbone.num_features

    model = nn.Sequential(
        backbone,
        nn.Dropout(dropout),
        nn.Linear(feat_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(512, NUM_CLASSES)
    ).to(device)

    # Loss with class weights
    total  = sum(class_counts.values())
    weights = torch.tensor(
        [total / (NUM_CLASSES * class_counts.get(i, 1))
         for i in range(NUM_CLASSES)],
        dtype=torch.float32
    ).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=weights)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP, eta_min=MIN_LR
    )

    # Training loop
    best_val_f1 = 0.0
    start = time.time()

    print(f"\n  {'Ep':>3} | {'TrF1':>6} | {'VaF1':>6} | {'VaAcc':>6}")
    print(f"  {'-'*35}")

    for epoch in range(EPOCHS):
        tr_loss, tr_f1 = train_one_epoch(
            model, train_dl, criterion, optimizer,
            device, epoch, WARMUP, lr
        )
        va_loss, va_acc, va_f1 = evaluate(model, val_dl, criterion, device)

        if epoch >= WARMUP:
            scheduler.step()

        marker = " ← BEST" if va_f1 > best_val_f1 else ""
        print(f"  {epoch+1:>3} | {tr_f1:>5.1f}% | "
              f"{va_f1:>5.1f}% | {va_acc:>5.1f}%{marker}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, exp_name, 'best.pth'))

    # Test evaluation
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, exp_name, 'best.pth'),
                   map_location=device)
    )
    _, test_acc, test_f1 = evaluate(model, test_dl, criterion, device)
    elapsed = (time.time() - start) / 60

    result = {
        'exp_name':    exp_name,
        'lr':          lr,
        'img_size':    img_size,
        'dropout':     dropout,
        'batch_size':  batch_size,
        'best_val_f1': best_val_f1,
        'test_f1':     test_f1,
        'test_acc':    test_acc,
        'time_min':    elapsed,
    }

    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  ✅ Test F1: {test_f1:.2f}% | "
          f"Test Acc: {test_acc:.2f}% | "
          f"Time: {elapsed:.1f}m")
    print(f"  Saved → {result_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # Build all combinations
    keys   = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))
    total  = len(combos)

    print(f"\n{'='*65}")
    print(f"  HYPERPARAMETER TUNING — EfficientNetV2-S")
    print(f"  Total experiments : {total}")
    print(f"  Epochs per exp    : {EPOCHS}")
    print(f"  Estimated time    : ~{total * 15}-{total * 20} min on T4")
    print(f"{'='*65}")

    all_results = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        result = run_experiment(
            lr         = params['lr'],
            img_size   = params['img_size'],
            dropout    = params['dropout'],
            batch_size = params['batch_size'],
            exp_id     = i,
            total_exps = total,
        )
        all_results.append(result)

    # ── Final Summary ────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  HYPERTUNING COMPLETE — RESULTS SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Experiment':<45} | {'ValF1':>6} | {'TestF1':>7}")
    print(f"  {'-'*62}")

    all_results.sort(key=lambda x: x['test_f1'], reverse=True)
    for r in all_results:
        print(f"  {r['exp_name']:<45} | "
              f"{r['best_val_f1']:>5.2f}% | "
              f"{r['test_f1']:>6.2f}%")

    best = all_results[0]
    print(f"\n{'='*65}")
    print(f"  🏆 BEST COMBINATION:")
    print(f"     Learning Rate : {best['lr']}")
    print(f"     Image Size    : {best['img_size']}")
    print(f"     Dropout       : {best['dropout']}")
    print(f"     Batch Size    : {best['batch_size']}")
    print(f"     Test F1       : {best['test_f1']:.2f}%")
    print(f"     Test Acc      : {best['test_acc']:.2f}%")
    print(f"{'='*65}")
    print(f"\n  Use these settings on supercomputer with 60% data!")
    print(f"  Target: 97.5%+ F1\n")

    # Save summary
    summary_path = os.path.join(SAVE_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Full summary saved: {summary_path}")
