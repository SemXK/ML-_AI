import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor 
import torchmetrics 

from timeit import default_timer as tm
import matplotlib.pyplot as plt
from helper_function import accuracy_fn
from tqdm.auto import tqdm
import random
import pandas as padding
from mlxtend.plotting import plot_confusion_matrix
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
class_names = trainData.classes
# * Convolutional Neural Network (CNN)
class CNNFashonMNISTModel(nn.Module):
  def __init__(
    self,
    inputShape: int,
    hiddenUnits: int,
    outputShape: int,
  ):
    super().__init__(),
    self.convBlock1 = nn.Sequential(
      nn.Conv2d(in_channels=inputShape,
        out_channels=hiddenUnits,
        kernel_size=3,
        stride=1,
        padding=1,
      ),
      nn.ReLU(),
      nn.Conv2d(in_channels=hiddenUnits,
        out_channels=hiddenUnits,
        kernel_size=3,
        stride=1,
        padding=1,
      ),
      nn.ReLU(),
      nn.MaxPool2d(
        kernel_size=2
      )
    )
    self.convBlock2 = nn.Sequential(
      nn.Conv2d(in_channels=hiddenUnits,
        out_channels=hiddenUnits,
        kernel_size=3,
        stride=1,
        padding=1,
      ),
      nn.ReLU(),
      nn.Conv2d(in_channels=hiddenUnits,
        out_channels=hiddenUnits,
        kernel_size=3,
        stride=1,
        padding=1,
      ),
      nn.ReLU(),
      nn.MaxPool2d(
        kernel_size=2
      )
    )
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=hiddenUnits *7 *7, out_features=outputShape),
    )

  def forward(self, x):
    return self.classifier(self.convBlock2(self.convBlock1(x)))
    # return self.convBlock1(self.convBlock2(self.classifier(x)))

fashonModel = CNNFashonMNISTModel(1, 10 ,10).to(device)

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
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # trainLoss += loss 
    # trainAcc += accuracyfn(y, yPred.argmax(dim=1))
  # trainLoss /= len(dataLoader)
  # trainAcc /= len(dataLoader)

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
    #   testLoss += lossfn(testPred, y) 
    #   testAcc += accuracyfn(y_true=y, y_pred=testPred.argmax(dim=1))
    # testLoss /= len(dataLoader)
    # testAcc /= len(dataLoader)
  # print(f"Test loss: {testLoss:.5f} | Test acc: {testAcc:.5f}")

image = torch.randn(size=(1,  28, 28))
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
  params=fashonModel.parameters(),
  lr=.1
)
# # * nn.Conv2d() Theory
# convLayer = nn.Conv2d(
#   in_channels=image.shape[0],
#   out_channels=10,
#   kernel_size=3,
#   stride=1,
#   padding=1
# )
# convImage = convLayer(image)

# # * nn.MaxPool2d() Theory
# poolLayer = nn.MaxPool2d(
#   kernel_size=2
# )

# finalImage = poolLayer(convImage)
# print(finalImage.shape, convImage.shape, image.shape)

# * Train / Test loop
# _epochs = 3
# startTimer = tm()
# for ep in tqdm(range(_epochs)):
#   TrainStep(
#     model=fashonModel,
#     dataLoader=trainDataLoader,
#     lossfn=lossfn,
#     optimizer=optimizer,
#     accuracyfn=accuracy_fn,
#     device=device
#   )
#   TestStep(
#     model=fashonModel,
#     dataLoader=testDataLoader,
#     lossfn=lossfn,
#     optimizer=optimizer,
#     accuracyfn=accuracy_fn,
#     device=device
#   )

# endTimer = tm()
# * Load trained loop (to make it faster)
from pathlib import Path

# 1* make directory
PATH = Path("TrainedModels")
PATH.mkdir(parents=True, exist_ok=True,)

# 1* Get the path of the new file
PATH_NAME = "FashonMNIST.pth"
SAVE_PATH = PATH / PATH_NAME

fashonModel.load_state_dict(torch.load(f=SAVE_PATH))         # & Load a saved PyTorch obj
# torch.load(PATH)                                              
# torch.nn.Module.load_state_dict()                             # & load a models saved state

def makePredictions(
  model: torch.nn.Module,
  data: list,
  device: torch.device
):
  predProbs = []
  model.to(device)
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)
      predLogit = model(sample) 
      predProb = torch.softmax(predLogit.squeeze(), dim=0)
      predProbs.append(predProb.cpu())
  return torch.stack(predProbs)

testSamples = list()
testLabels = list()
for sample, label in random.sample(list(trainData), k=9):
  testSamples.append(sample)
  testLabels.append(label)


# # * Save a model
# from pathlib import Path

# # 1* make directory
# PATH = Path("TrainedModels")
# PATH.mkdir(parents=True, exist_ok=True,)

# # 1* Get the path of the new file
# PATH_NAME = "FashonMNIST.pth"
# SAVE_PATH = PATH / PATH_NAME

# # 1* Save the model in the specified path
# torch.save(obj=fashonModel.state_dict(), f=SAVE_PATH)                         # & Saves a PyTorch obj

# plt.imshow(
#   testSamples[0].squeeze(),
#   cmap="gray"
# )
# plt.show()

# # * view pred list 
# predProbs = makePredictions(
#   model=fashonModel,
#   data=testSamples,
#   device=device
# )
# predClasses = predProbs.argmax(dim=1)

# # * Visualize data
# plt.figure(figsize=(9,9))
# rows = 3
# cols = 3
# for i, sample in enumerate(testSamples):
#   plt.subplot(rows, cols, i+1)

#   plt.imshow(
#     sample.squeeze(),
#     cmap="gray"
#   )
#   predLabel = class_names[predClasses[i]]
#   trueLabel = class_names[testLabels[i]]
#   if predLabel == trueLabel:
#     plt.title(f"{predLabel},{trueLabel} ", color="g")
#   else:
#     plt.title(f"{predLabel},{trueLabel} ", color="r")


# plt.show()


# * Confusing Matrix 

# 1* Make predictions on the data set
yPreds = []
fashonModel.eval()
with torch.inference_mode():
  for x, y in tqdm(testDataLoader, desc="Making preds"):
    x, y = x.to(device), y.to(device)

    # 1* get logits
    yLogit = fashonModel(x)

    # 1* logits -> probabilities -> labels 
    yPred = torch.softmax(yLogit.squeeze(), dim=0).argmax(dim=1)

    yPreds.append(yPred.cpu())
yPredTensor = torch.cat(yPreds)

# 1* Create Confusion Matrix
confMat = torchmetrics.ConfusionMatrix(
  task='multiclass',
  num_classes=len(class_names)
)
confMatTensor = confMat(
  preds=yPredTensor,
  target=testData.targets
  )
# 1* Visualize ConfMatrix
fig, ax  = plot_confusion_matrix(
  conf_mat=confMatTensor.numpy(),
  class_names=class_names,
  figsize=(10, 7)
)
fig.show()