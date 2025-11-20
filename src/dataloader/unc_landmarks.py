from tkinter import Image
import random
import torch
import cv2
import numpy as np
import albumentations

from glob import glob
from pathlib import Path
from torch.utils.data import Dataset


DS_BASE_DIR = "/playpen-ssd/levi/comp790-183/unc-landmark-recognition/data/unc-landmarks"


class UNCLandmarksDataset(Dataset):

    def __init__(self, split: str):
        
        super().__init__()
        
        img_fps = sorted(glob(f"{DS_BASE_DIR}/*/*/*.jpg", recursive=True) + glob(f"{DS_BASE_DIR}/*/*/*.JPG", recursive=True))
        
        # always shuffle our dataset the same way
        random.seed(42)
        random.shuffle(img_fps)
        
        self.oh_labels = {}
        
        idx = 0
        for fp in img_fps:
            img_class = Path(fp).parent.parent.name
            if img_class not in self.oh_labels: self.oh_labels[img_class] = idx; idx+=1
        
        # [(X, y)]
        self.data_buffer = [[fp, self.oh_labels[Path(fp).parent.parent.name]] for fp in img_fps]

        if split   == "train": self.data_buffer = self.data_buffer[:int(len(self.data_buffer) * .8) ]
        elif split == "val": self.data_buffer = self.data_buffer[int(len(self.data_buffer)  * .8):]
        else: raise Exception("bad split")

    def __len__(self): return len(self.data_buffer)

    @staticmethod
    def preproc_X(X:np.ndarray) -> torch.Tensor:

        # [H, W, C] -> [224, 224, C]
        transform = albumentations.Compose([
            albumentations.RandomCrop(height=224, width=224),
        ])
        return torch.tensor(transform(image=X)['image'])

    def __getitem__(self, index) -> dict:
        
        X, y_idx = self.data_buffer[index]

        # ehhhh.... a litle lazy load
        if type(X) is str:
            self.data_buffer[index][0] = (cv2.imread(X) / 255)
            X = self.data_buffer[index][0]

        H, W, C  = X.shape
        X        = UNCLandmarksDataset.preproc_X(X)
        X        = X.permute(-1, 0, 1) # -> [C, H, W]
        y        = torch.zeros(len(self.oh_labels))
        
        # one-hot categorical label
        y[y_idx] = 1

        return {
            "X": X,
            "y": y,
        }
    

if __name__ == "__main__":

    ds = UNCLandmarksDataset(split="train")
    ds[0]