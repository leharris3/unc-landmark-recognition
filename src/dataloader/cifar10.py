import torch
import torchvision as TV
import numpy as np

import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL.Image import Image


ROOT = "/playpen-ssd/levi/comp790-183/unc-landmark-recognition/data/cifar10"


class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, train=True, root=ROOT, download=False):
        super().__init__()
        self.dataset = TV.datasets.CIFAR10(root=root, train=train, download=download)
        self.tt = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        X, y = self.dataset[index]
        X = self.tt(X)
        return {
            "X": X, # pil IMG input
            "y": F.one_hot(torch.tensor(y), num_classes=10),
        }


if __name__ == "__main__":
    ds = CIFAR10(); ds[0]; print(ds[0])