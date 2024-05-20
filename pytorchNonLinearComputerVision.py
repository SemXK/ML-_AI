import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor 

from timeit import default_timer as tm
import matplotlib.pyplot as plt
from helper_function import accuracy_fn
from tqdm.auto import tqdm

# * Device-agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# * Time tracking
def trainTime(start:float, end: float, device: torch.device = None): 
  print(f"Total time passed: {device} : {end - start} seconds")


# *Setup train data
trainData = datasets.FashionMNIST(
  root="data",   
  train=True,
  download=True,
  transform=ToTensor(),
)
testData = datasets.FashionMNIST(
  root="data",   
  train=False,
  download=True,
  transform=ToTensor(),
)

# * prepare data
_batch = 32 
trainDataLoader = DataLoader(
  dataset=trainData,
  batch_size=_batch,
  shuffle=True
)

testDataLoader = DataLoader(
  dataset=testData,
  batch_size=_batch,
  shuffle=False
)
# * Non linear model (on cuda)
class NLFashionMNISTModel(nn.Module):
  def __init__(
      self,
      inputShape: int,
      hiddenUnits: int,
      outputShape: int,
    ):
    super().__init__(),
    self.layerStack = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=inputShape, out_features=hiddenUnits),
      nn.ReLU(),
      nn.Linear(in_features=hiddenUnits, out_features=outputShape),
      nn.ReLU(),
    )
  def forward(self, x):
    return self.layerStack(x)

cudaFashonModel = NLFashionMNISTModel(
  inputShape=28*28,
  hiddenUnits=10,
  outputShape=10
).to(device)

# * loss, optimizer
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
  params=cudaFashonModel.parameters(),
  lr=.1
)

# * Train step
def TrainStep(
  model: torch.nn.Module,
  dataLoader: torch.utils.data.DataLoader,
  lossfn: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  accuracyfn,
  device: torch.device = device
):
  trainLoss, trainAcc = 0 , 0
  model.train()
  for batch, (x, y) in enumerate(trainDataLoader):
    x, y = x.to(device), y.to(device)

    yPred = model(x)
    loss = lossfn(yPred, y)
    trainLoss += loss 
    trainAcc += accuracyfn(y, yPred.argmax(dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  trainLoss /= len(dataLoader)
  trainAcc /= len(dataLoader)

# * Test step
def TestStep(
  model: torch.nn.Module,
  dataLoader: torch.utils.data.DataLoader,
  lossfn: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  accuracyfn,
  device: torch.device = device
):

  testLoss, testAcc = 0 , 0
  model.eval()
  with torch.inference_mode():

    for batch, (x, y) in enumerate(testDataLoader):
      x, y = x.to(device), y.to(device)

      testPred = model(x)

      testLoss += lossfn(testPred, y) 
      testAcc += accuracyfn(y_true=y, y_pred=testPred.argmax(dim=1))

    testLoss /= len(dataLoader)
    testAcc /= len(dataLoader)

  print(f"Test loss: {testLoss:.5f} | Test acc: {testAcc:.5f}")

startTime = tm()
_epochs = 3

for ep in tqdm(range(_epochs)):
  print(f"Epoch: {ep}")
  # * Train model
  TrainStep(
    model=cudaFashonModel,
    dataLoader=trainDataLoader,
    lossfn=lossfn,
    optimizer=optimizer,
    accuracyfn=accuracy_fn,
    device=device
  )
  # * Test model
  TestStep(
    model=cudaFashonModel,
    dataLoader=trainDataLoader,
    lossfn=lossfn,
    optimizer=optimizer,
    accuracyfn=accuracy_fn,
    device=device
  )

endTime = tm()
trainTime(start=startTime, end=endTime, device=device)

