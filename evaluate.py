import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, accuracy_score)
from collections import Counter

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from model import build_model

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_ROOT = r'E:\GrainQuality\tiny_data'
RESULTS_DIR = r'E:\GrainQuality\results'


# ─── EVALUATE ONE MODEL ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model_name, save_dir, data_root=DATA_ROOT):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path = os.path.join(save_dir, model_name, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found for {model_name}")
        return None

    # Load checkpoint
    ckpt   = torch.load(ckpt_path, map_location=device)
    config = ckpt['config']

    # Build model
    model, criterion = build_model(
        model_name  = model_name,
        num_classes = NUM_CLASSES,
        device      = device,
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Load test data
    _, _, test_dl, _ = get_dataloaders(
        data_root,
        img_size   = config['img_size'],
        batch_size = config['batch_size'],
    )

    all_preds, all_labels = [], []
    for imgs, labels in test_dl:
        imgs   = imgs.to(device)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1).cpu().tolist()
        all_preds .extend(preds)
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    return {
        'model':      model_name,
        'accuracy':   acc,
        'macro_f1':   f1,
        'preds':      all_preds,
        'labels':     all_labels,
    }


# ─── COMPARE ALL EXPERIMENTS ──────────────────────────────────────────────────
def compare_all(results_dir=RESULTS_DIR):
    print(f"\n{'='*65}")
    print(f"  COMPARISON TABLE — All Experiments")
    print(f"{'='*65}")
    print(f"  {'Model':<22} | {'Val F1':>7} | {'Val Acc':>8} | {'Test F1':>7} | {'Test Acc':>8} | {'Time':>7}")
    print(f"  {'-'*60}")

    all_results = []
    for exp_folder in sorted(os.listdir(results_dir)):
        json_path = os.path.join(results_dir, exp_folder, 'results.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            r = json.load(f)
        all_results.append(r)
        print(f"  {r['model']:<22} | {r['best_val_f1']:>6.2f}% | "
              f"{r['best_val_acc']:>7.2f}% | {r['test_f1']:>6.2f}% | "
              f"{r['test_acc']:>7.2f}% | {r['time_min']:>5.1f}m")

    print(f"{'='*65}")

    if all_results:
        best = max(all_results, key=lambda x: x['test_f1'])
        print(f"\n  ✅ Best model: {best['model']}")
        print(f"     Test F1   : {best['test_f1']:.2f}%")
        print(f"     Test Acc  : {best['test_acc']:.2f}%")

    return all_results


# ─── PLOT TRAINING CURVES ─────────────────────────────────────────────────────
def plot_training_curves(results_dir=RESULTS_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — All Models', fontsize=14, fontweight='bold')

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']

    for i, exp_folder in enumerate(sorted(os.listdir(results_dir))):
        json_path = os.path.join(results_dir, exp_folder, 'results.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            r = json.load(f)

        epochs  = [h['epoch']  for h in r['history']]
        va_f1   = [h['va_f1']  for h in r['history']]
        va_acc  = [h['va_acc'] for h in r['history']]
        color   = colors[i % len(colors)]

        axes[0].plot(epochs, va_f1,  color=color, label=r['model'], linewidth=2)
        axes[1].plot(epochs, va_acc, color=color, label=r['model'], linewidth=2)

    axes[0].set_title('Validation F1 Score')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Macro F1 (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Draw paper baseline
    axes[0].axhline(y=96.4, color='red', linestyle='--',
                    linewidth=1.5, label='Paper baseline (96.4%)')
    axes[0].legend()

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining curves saved: {out_path}")
    plt.show()


# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
def plot_confusion_matrix(model_name, results_dir=RESULTS_DIR):
    result = evaluate_model(model_name, results_dir)
    if result is None:
        return

    cm = confusion_matrix(result['labels'], result['preds'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar_kws={'label': '%'})
    ax.set_title(f'Confusion Matrix — {model_name}\n'
                 f'Accuracy: {result["accuracy"]:.2f}%  |  '
                 f'Macro F1: {result["macro_f1"]:.2f}%',
                 fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()

    out_path = os.path.join(results_dir, model_name,
                            'confusion_matrix.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved: {out_path}")
    plt.show()

    # Also print classification report
    print(f"\nClassification Report — {model_name}")
    print(classification_report(result['labels'], result['preds'],
                                target_names=CLASS_NAMES, zero_division=0))


# ─── PAPER COMPARISON TABLE ───────────────────────────────────────────────────
def paper_comparison(results_dir=RESULTS_DIR):
    print(f"\n{'='*65}")
    print(f"  VS PAPER BASELINES (GrainSet — Nature 2023)")
    print(f"{'='*65}")
    print(f"  {'Method':<28} | {'Avg Acc':>8} | {'Macro F1':>9} | Source")
    print(f"  {'-'*60}")

    # Paper baselines
    paper_results = [
        ('Color Hist + SVM',        '~70%',  '70.6%', 'GrainSet paper'),
        ('SIFT + SVM',              '~35%',  '35.5%', 'GrainSet paper'),
        ('VGG19',                   '~94%',  '~93%',  'GrainSet paper'),
        ('Inception-v3',            '~94%',  '~93%',  'GrainSet paper'),
        ('ResNet-50',               '~95%',  '~95%',  'GrainSet paper'),
        ('ResNet-152',              '96.4%', '96.1%', 'GrainSet paper'),
        ('ViT',                     '~95%',  '~94%',  'GrainSet paper'),
        ('ResNet-50 + SE (best)',   'BEST',  'BEST',  'GrainSet paper'),
    ]

    for name, acc, f1, source in paper_results:
        print(f"  {name:<28} | {acc:>8} | {f1:>9} | {source}")

    print(f"  {'─'*60}")

    # Our results
    for exp_folder in sorted(os.listdir(results_dir)):
        json_path = os.path.join(results_dir, exp_folder, 'results.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            r = json.load(f)
        better = "✅ BETTER" if r['test_f1'] > 96.1 else ""
        print(f"  {r['model']:<28} | {r['test_acc']:>7.2f}% | "
              f"{r['test_f1']:>8.2f}% | OURS {better}")

    print(f"{'='*65}")


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    compare_all()
    plot_training_curves()
    paper_comparison()

    # Uncomment to see confusion matrix for specific model:
    # plot_confusion_matrix('efficientnetv2_s')
