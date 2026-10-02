"""EfficientNet-B0 adapted for log-Mel spectrogram input (Model 2)."""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EffNetB0Audio(nn.Module):
    def __init__(self, n_classes, pretrained=True, dropout=0.2, img_size=224):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = efficientnet_b0(weights=weights)
        in_f = self.net.classifier[1].in_features            # 1280
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, n_classes))
        self.img_size = img_size

    def forward(self, x):                                    # x: [B, 1, n_mels, T]
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)                             # 1 -> 3 channels for ImageNet weights
        return self.net(x)

    def param_groups(self, lr_head, lr_backbone, weight_decay):
        head = list(self.net.classifier.parameters())
        head_ids = {id(p) for p in head}
        backbone = [p for p in self.parameters() if id(p) not in head_ids]
        return [
            {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": head, "lr": lr_head, "weight_decay": weight_decay},
        ]
