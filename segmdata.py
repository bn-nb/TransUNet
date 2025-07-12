import os
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, InterpolationMode


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.float().view(-1)

        intsxn = (probs * targets).sum()
        dice = (2. * intsxn + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class ComboLoss(nn.Module):
    # this loss handles class imbalance properly
    def __init__(self, bce_wt=0.5, pos_wt=10.0):
        super().__init__()
        self.bce_wt = bce_wt
        if pos_wt is not None:
            if not isinstance(pos_wt, torch.Tensor):
                pos_wt = torch.Tensor([pos_wt])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_wt)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_wt * bce_loss + (1 - self.bce_wt) * dice_loss


class SynapseSegm(Dataset):
    def __init__(self, directory, threshold=0.05, transforms=None):
        self.directory = directory
        self.threshold = threshold
        self.file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]

        if transform is None:
            transform = Compose([
                lambda sample: {
                    'image': Resize((448, 448), interpolation=InterpolationMode.BILINEAR)(sample['image']),
                    'masks': Resize((448, 448), interpolation=InterpolationMode.NEAREST)(sample['masks'])
                }
            ])

        self.transform = transform

        if not self.file_list:
            raise RuntimeError(f"No .npz files found in directory: {directory}")
        else:
            print(len(self.file_list), "files were found!")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        image = data['image']
        masks = data['label']

        if image.ndim == 2: # (512, 512)
            image = np.expand_dims(image, axis=0)
            masks = np.expand_dims(masks, axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        masks = (masks > self.threshold).astype(np.uint8) # 0 or 1
        sample = {'image': image, 'masks': masks}

        if self.transform:
            sample = self.transform(sample)

        return sample['image'], sample['masks']
