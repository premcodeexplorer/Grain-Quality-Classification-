import torch
import torch.nn as nn
import timm


# ─── MODELS TO EXPERIMENT WITH ───────────────────────────────────────────────
# We will try all of these and pick the best one
# Format: 'experiment_name': 'timm_model_name'

MODELS = {
    'resnet50_se':       'resnet50',            # Baseline — same as paper
    'efficientnetv2_s':  'tf_efficientnetv2_s', # Our primary candidate
    'convnext_small':    'convnext_small',       # Our secondary candidate
    'efficientnet_b4':   'efficientnet_b4',      # Classic EfficientNet
    'swin_tiny':         'swin_tiny_patch4_window7_224',  # Transformer
}


# ─── FOCAL LOSS ───────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    gamma=2 : focuses on hard examples
    alpha   : class weights (computed from dataset)
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction
        self.ce        = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt      = torch.exp(-ce_loss)
        loss    = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# ─── GRAIN QUALITY MODEL ──────────────────────────────────────────────────────
class GrainQualityModel(nn.Module):
    def __init__(self, model_name='tf_efficientnetv2_s',
                 num_classes=8, pretrained=True, dropout=0.3):
        super().__init__()

        # Load backbone from timm (pretrained on ImageNet)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # Remove original classifier
            global_pool='avg'
        )

        # Get feature dimension from backbone
        feat_dim = self.backbone.num_features

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

        print(f"Model      : {model_name}")
        print(f"Features   : {feat_dim}")
        print(f"Classes    : {num_classes}")
        print(f"Parameters : {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ─── BUILD MODEL + LOSS ───────────────────────────────────────────────────────
def build_model(model_name='tf_efficientnetv2_s',
                num_classes=8,
                class_counts=None,
                device='cuda'):
    """
    model_name    : key from MODELS dict or direct timm name
    num_classes   : 8 for tiny, 10 for full dataset
    class_counts  : dict of {label: count} for computing class weights
    """
    # Resolve model name
    timm_name = MODELS.get(model_name, model_name)

    model = GrainQualityModel(
        model_name=timm_name,
        num_classes=num_classes
    ).to(device)

    # Compute class weights for Focal Loss from training data counts
    alpha = None
    if class_counts:
        total  = sum(class_counts.values())
        weights = torch.tensor(
            [total / (num_classes * class_counts.get(i, 1))
             for i in range(num_classes)],
            dtype=torch.float32
        ).to(device)
        alpha = weights
        print(f"Class weights: {[f'{w:.2f}' for w in weights.cpu()]}")

    criterion = FocalLoss(gamma=2.0, alpha=alpha)

    return model, criterion


# ─── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    for exp_name, timm_name in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {exp_name}")
        try:
            model, _ = build_model(timm_name, num_classes=8, device=device)
            dummy = torch.randn(4, 3, 224, 224).to(device)
            out   = model(dummy)
            print(f"Output shape: {out.shape}  ✅")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error: {e}  ❌")

    print(f"\nmodel.py working correctly ✅")
