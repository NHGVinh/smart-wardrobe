import torch.nn as nn
from torchvision import models


class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories, num_styles, weights=None):
        super().__init__()
        self.resnet = models.resnet18(weights=weights)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_category = nn.Linear(num_ftrs, num_categories)
        self.fc_style = nn.Linear(num_ftrs, num_styles)

    def forward(self, x):
        features = self.resnet(x)
        return self.fc_category(features), self.fc_style(features)


def freeze_resnet_friend_style(model):
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "resnet.layer4" in name or "fc_category" in name or "fc_style" in name:
            param.requires_grad = True
