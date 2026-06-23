from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.preprocessing.resnetPP import load_resnet_image


class ResNetFashionDataset(Dataset):
    def __init__(self, dataframe, image_dir, cat_to_idx, style_to_idx, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.cat_to_idx = cat_to_idx
        self.style_to_idx = style_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = load_resnet_image(self.image_dir / f"{row['id']}.jpg")
        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(self.cat_to_idx[row["label_name"]], dtype=torch.long),
            torch.tensor(self.style_to_idx[row["usage"]], dtype=torch.long),
        )
