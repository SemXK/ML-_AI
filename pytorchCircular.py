import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot


class  CircularModel(nn.Module):
  def __init__(self):
    super().__init__(),
    self.mainLayer = nn.Sequential(
      nn.Linear(in_features=2, out_features=10),
      nn.Linear(in_features=10, out_features=10),
      nn.Linear(in_features=10, out_features=1),
    ).to("cuda")

  def forward(self, x):
    return self.mainLayer(x)

circular = CircularModel().to('cuda')
def accuracy_fn(yTrue, yPred):
  correct = torch.eq(yTrue, yPred)
  accuracy = (correct / len(yPred) * .01) 
  return accuracy

# * Learning loop
sample = 1000
X, Y = make_circles(
  sample,
  noise=.03,
  random_state=42
)
X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.float)

XTrain, XTest, YTrain, YTest = train_test_split(
  X,
  Y,
  test_size=.2
)
XTrain,  XTest =  XTrain.to("cuda"), XTest.to("cuda")
YTrain, YTest =  YTrain.to("cuda"), YTest.to("cuda")

lossFn = nn.BCEWithLogitsLoss()   # & nn.sequential( nn.Sigmoid(), nn.BCELoss() ), expects raw logits
optimizer = torch.optim.SGD(
  params=circular.parameters(),
  lr=.1
)

epochs = 1000

for ep in range(epochs):
  circular.train() 
  # 2* Forward pass
  yLogits = circular(XTrain.to('cuda')).squeeze()
  yPred = torch.round(torch.sigmoid(yLogits))
  loss = lossFn(yLogits, YTrain.to('cuda'))    #? Pytorch and sklearn invert the train and pred datas
  acc = accuracy_fn(YTrain.to('cuda'), yPred)


  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  # 2* Test
  circular.eval()
  with torch.inference_mode():
    testLogits = circular(XTest.to('cuda')).squeeze()  
    testPred = torch.round(torch.sigmoid(testLogits))
    # 3* Calculate loss
    testAcc = accuracy_fn(YTest.to('cuda'), testPred)

    if ep  % 100 == 0:
      print(f"Epoch: {ep} | Loss: {loss:.5f} | Acc: {acc} | Test loss: {testAcc} ")


