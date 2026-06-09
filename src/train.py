import os
import json
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from dataset import get_dataloaders, NUM_CLASSES
from model import build_model, MODELS

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIG — change these to run different experiments
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Data ──────────────────────────────────────────────
    'data_root':    r'E:\GrainQuality\tiny_data',
    'img_size':     224,
    'batch_size':   16,       # Safe for 4GB VRAM
    'num_workers':  2,

    # ── Model ─────────────────────────────────────────────
    # Options: 'resnet50_se' | 'efficientnetv2_s' |
    #          'convnext_small' | 'efficientnet_b4' | 'swin_tiny'
    'model_name':   'efficientnet_b4',   # ← CHANGE THIS per experiment
    'num_classes':  NUM_CLASSES,           # 8 for tiny, 10 for full

    # ── Training ──────────────────────────────────────────
    'epochs':       30,
    'lr':           3e-4,
    'weight_decay': 0.01,
    'grad_clip':    1.0,

    # ── Cosine LR schedule ────────────────────────────────
    'warmup_epochs': 3,
    'min_lr':        1e-6,

    # ── Output ────────────────────────────────────────────
    'save_dir':     r'E:\GrainQuality\results',
}
# ══════════════════════════════════════════════════════════════════════════════


def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup for first N epochs."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = lr


def train_one_epoch(model, loader, criterion, optimizer,
                    device, grad_clip, epoch, warmup_epochs, base_lr):
    model.train()
    warmup_lr(optimizer, epoch, warmup_epochs, base_lr)

    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in tqdm(loader, desc=f'  Train', leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds .extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds) * 100
    f1       = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0) * 100
    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().tolist()
        all_preds .extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds) * 100
    f1       = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0) * 100
    return avg_loss, acc, f1


def train(config):
    # ── Setup ────────────────────────────────────────────────────────────────
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_name = config['model_name']
    save_dir = os.path.join(config['save_dir'], exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Experiment : {exp_name}")
    print(f" Device     : {device}")
    print(f" Save dir   : {save_dir}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dl, val_dl, test_dl, train_ds = get_dataloaders(
        config['data_root'],
        img_size    = config['img_size'],
        batch_size  = config['batch_size'],
        num_workers = config['num_workers'],
    )

    class_counts = dict(Counter(train_ds.labels))

    # ── Model ─────────────────────────────────────────────────────────────────
    model, criterion = build_model(
        model_name   = config['model_name'],
        num_classes  = config['num_classes'],
        class_counts = class_counts,
        device       = device,
    )

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config['lr'],
        weight_decay = config['weight_decay'],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = config['epochs'] - config['warmup_epochs'],
        eta_min= config['min_lr'],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1      = 0.0
    best_acc     = 0.0
    history      = []
    start_time   = time.time()

    print(f"\n{'Epoch':>5} | {'TrLoss':>7} | {'TrAcc':>6} | {'TrF1':>6} | "
          f"{'VaLoss':>7} | {'VaAcc':>6} | {'VaF1':>6} | {'LR':>8}")
    print('-' * 70)

    for epoch in range(config['epochs']):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_dl, criterion, optimizer,
            device, config['grad_clip'],
            epoch, config['warmup_epochs'], config['lr']
        )

        va_loss, va_acc, va_f1 = evaluate(model, val_dl, criterion, device)

        # Step scheduler after warmup
        if epoch >= config['warmup_epochs']:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch+1:>5} | {tr_loss:>7.4f} | {tr_acc:>5.1f}% | "
              f"{tr_f1:>5.1f}% | {va_loss:>7.4f} | {va_acc:>5.1f}% | "
              f"{va_f1:>5.1f}% | {current_lr:>8.2e}"
              + (" ← BEST" if va_f1 > best_f1 else ""))

        history.append({
            'epoch': epoch + 1,
            'tr_loss': tr_loss, 'tr_acc': tr_acc, 'tr_f1': tr_f1,
            'va_loss': va_loss, 'va_acc': va_acc, 'va_f1': va_f1,
            'lr': current_lr,
        })

        # Save best model
        if va_f1 > best_f1:
            best_f1  = va_f1
            best_acc = va_acc
            torch.save({
                'epoch':      epoch + 1,
                'model_name': config['model_name'],
                'state_dict': model.state_dict(),
                'val_f1':     best_f1,
                'val_acc':    best_acc,
                'config':     config,
            }, os.path.join(save_dir, 'best_model.pth'))

    # ── Final test evaluation ─────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'),
                      map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    te_loss, te_acc, te_f1 = evaluate(model, test_dl, criterion, device)

    elapsed = (time.time() - start_time) / 60

    print(f"\n{'='*60}")
    print(f" RESULTS — {exp_name}")
    print(f"{'='*60}")
    print(f"  Best Val F1       : {best_f1:.2f}%")
    print(f"  Best Val Accuracy : {best_acc:.2f}%")
    print(f"  Test F1           : {te_f1:.2f}%")
    print(f"  Test Accuracy     : {te_acc:.2f}%")
    print(f"  Time              : {elapsed:.1f} min")
    print(f"{'='*60}\n")

    # Save history + results
    results = {
        'model':     exp_name,
        'best_val_f1':  best_f1,
        'best_val_acc': best_acc,
        'test_f1':   te_f1,
        'test_acc':  te_acc,
        'time_min':  elapsed,
        'config':    config,
        'history':   history,
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_dir}\\results.json")
    return results


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = train(CONFIG)
