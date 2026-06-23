import torch.nn as nn
from torchvision import models


# Tạo Kiến trúc 1 Não - 2 Đầu Ra
class MultiTaskResNet(nn.Module):
    def __init__(self, num_categories, num_styles, weights=models.ResNet18_Weights.DEFAULT):
        super(MultiTaskResNet, self).__init__()
        # BẮT BUỘC DÙNG DEFAULT KHI TRAIN ĐỂ CÓ KIẾN THỨC NỀN
        self.resnet = models.resnet18(weights=weights)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_category = nn.Linear(num_ftrs, num_categories)
        self.fc_style = nn.Linear(num_ftrs, num_styles)

    def forward(self, x):
        features = self.resnet(x)
        return self.fc_category(features), self.fc_style(features)


def unlock_last_block_and_heads(model):
    # 1. Đóng băng TOÀN BỘ não bộ lúc đầu
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 2. TUYỆT CHIÊU MỞ KHÓA: Chỉ đánh thức Block cuối cùng (layer4) và 2 đầu ra
    for name, param in model.named_parameters():
        if "resnet.layer4" in name or "fc_category" in name or "fc_style" in name:
            param.requires_grad = True
