import torch
from torch import nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot
from helper_function import plot_predictions, plot_decision_boundary
from torchmetrics import Accuracy
# * Hyperparameters
_classes = 4
_features = 2
_seed =  42

# * Data
xBlob, yBlob = make_blobs(
  n_samples=1000,
  n_features=_features,
  centers=_classes,
  cluster_std=1.5,
  random_state=_seed
)
# # Mandatory for the multiclass
xBlob = torch.from_numpy(xBlob).type(torch.long)
yBlob = torch.from_numpy(yBlob).type(torch.long)

xTrain, xTest, yTrain, yTest = train_test_split(
  xBlob,
  yBlob,
  random_state=_seed
)
xTrain = xTrain.to("cuda")
xTest = xTest.to("cuda")
yTrain = yTrain.to("cuda")
yTest = yTest.to("cuda")
# accuracy = Accuracy()
# * Plot build

# plot.figure(figsize=(10, 7))
# plot.scatter(
#   xBlob[:, 0],
#   xBlob[:,1],
#   c=yBlob,
#   cmap=plot.cm.RdYlBu
# )
# plot.show()
# * Model definition
class MultiClassModel(nn.Module):
  def __init__(self, in_features, out_features, hidden_units=8):
    super().__init__(),
    self.mainLayer = nn.Sequential(
      nn.Linear(in_features=in_features, out_features=hidden_units),
      nn.Linear(in_features=hidden_units, out_features=hidden_units),
      nn.Linear(in_features=hidden_units, out_features=out_features),
    )
  
  def forward(self, x):
    x = x.type(torch.float) 
    return self.mainLayer(x)

BlobModel = MultiClassModel(
  in_features=2,
  out_features=_classes,
  hidden_units=8
).to("cuda")

# * Loss + optimizer
lossF = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
  params=BlobModel.parameters(),
  lr=0.1
)

# * Training loop
_epochs = 1000

for ep in range(_epochs):
  BlobModel.train()
  # 1* logits -> pred probs (softmax) - >pred labels (argmax)
  yLogits = BlobModel(xTrain).squeeze()
  yPred = torch.softmax(yLogits, dim=1).argmax(dim=1)
  loss = lossF(yLogits, yTrain)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # 1* Test mode
  BlobModel.eval()
  with torch.inference_mode():
    testLogits = BlobModel(xTest)
    testPreds = torch.softmax(testLogits, dim=1).argmax(dim=1)

    testLoss = lossF(testLogits, yTest)
  
  # 1* Test mode
  BlobModel.eval()
  with torch.inference_mode():
    yLogits = BlobModel(xTest)
  # if ep % 100 == 0:
  #   accuracy(yPred, yTest)

# plot.figure(figsize=(10, 7))
# plot.subplot(1,2,1)
# plot.title("Train")
# plot_decision_boundary(BlobModel, xTrain, yTrain)
# plot.show()

# plot.figure(figsize=(10, 7))
# plot.subplot(1,2,1)
# plot.title("Test")
# plot_decision_boundary(BlobModel, xTest, yTest)
# plot.show()