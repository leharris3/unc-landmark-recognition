import torch
import torchvision
from torchvision.models.resnet import ResNet18_Weights


class Resnet18(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # init model w/ imagenet1k weights
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(512, 10) # relace w/ N=10 cls head

    def forward(self, x): return self.model(x)


if __name__ == "__main__":
    # from src.dataloader.cifar10 import CIFAR10
    # ds = CIFAR10()
    model = Resnet18()
    model(torch.rand(1, 3, 32, 32))