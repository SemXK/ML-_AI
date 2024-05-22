import torch
from torch import nn

class TinyVGG(nn.Module):
  def __init__(
    self,
    inputShape: int,
    hiddenUnits: int,
    outputShape: int
  ):
    super().__init__(),
    self.convBlock1 = nn.Sequential(
      nn.Conv2d(
        in_channels=inputShape,
        out_channels=hiddenUnits,
        kernel_size=3,
        stride=1,
        padding=0
      ),
      nn.ReLU(),
      nn.MaxPool2d(
        kernel_size=2,
        stride=2
      )
    )
    self.convBlock2 = nn.Sequential(
      nn.Conv2d(
        in_channels=hiddenUnits,
        out_channels=hiddenUnits,
        kernel_size=3,
        padding=0
      ),
      nn.ReLU(),
      nn.Conv2d(
        in_channels=hiddenUnits,
        out_channels=hiddenUnits,
        kernel_size=3,
        padding=0
      ),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(
        in_features=hiddenUnits*13*13 ,
        out_features=outputShape
      )
    )
  
  def forward(self, x):
    return self.classifier(self.convBlock2(self.convBlock1(x)))