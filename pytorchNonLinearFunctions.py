import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plot

samples = 1000
X, Y = make_circles(
  samples,
  noise=.03,
  random_state=42,
)

# plot.scatter(X[:, 0], Y[: , 1], c=Y, cmap=plot.cm.RdYlBu)
# plot.show()

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(Y).type(torch.float)

# * Params Analizers
def accuracy_fn(yTrue, yPred):
  correct = torch.eq(yTrue, yPred)
  accuracy = (correct / len(yPred) * .01) 
  return accuracy

# * Data split
xTrain, xTest, yTrain, yTest = train_test_split(
  X,
  Y,
  test_size=.2, 
  random_state=42
)
XTrain, XTest = xTrain.to("cuda"), xTest.to("cuda")
YTrain, YTest = yTrain.to("cuda"), yTest.to("cuda")

class CircularModel(nn.Module):
  def __init__(self):
    super().__init__(),
    self.mainLayer = nn.Sequential(
      nn.Linear(in_features=2, out_features=10),
      nn.ReLU(),
      nn.Linear(in_features=10, out_features=10),
      # nn.ReLU(),
      nn.Linear(in_features=10, out_features=1),

    )

  def forward(self, x):
    return self.mainLayer(x)

circularNonLinear = CircularModel().to("cuda:0")

# * Model learning (Binary problem)
untrainedPreds = circularNonLinear(XTest.to("cuda"))
lossFn = nn.BCELoss()     # & expects prediction probabilities

lossFn = nn.BCEWithLogitsLoss()   # & nn.sequential( nn.Sigmoid(), nn.BCELoss() ), expects raw logits
optimizer = torch.optim.SGD(
  params=circularNonLinear.parameters(),
  lr=.1
)
epochs = 1000
# * Training loop
for ep in range(epochs):
  circularNonLinear.train() 
  # 2* Forward pass
  yLogits = circularNonLinear(XTrain).squeeze()
  yPred = torch.round(torch.sigmoid(yLogits))
  loss = lossFn(yLogits, YTrain)    #? Pytorch and sklearn invert the train and pred datas

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
# * Test
circularNonLinear.eval()
with torch.inference_mode():
  yPreds = torch.round(torch.sigmoid(circularNonLinear(XTest))).squeeze()
  print(yPred[:10])
  print(yTest[:10])


from helper_function import plot_predictions, plot_decision_boundary
plot.figure(figsize=(12,6))
plot.subplot(1, 2, 1)
plot.title("Train")
plot_decision_boundary(circularNonLinear, XTrain, YTrain)
plot.show()

# plot.subplot(1, 2, 1)
# plot.title("Test")
# plot_decision_boundary(circularNonLinear, XTest, YTest)
# plot.show()