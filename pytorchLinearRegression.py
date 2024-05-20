import torch
from torch import nn # & neural network
import matplotlib.pyplot as plt # & plotting

# * initial data

weight = .7
bias = .3
start = 0
end = 1
step = .02
X = torch.arange(start, end, step).unsqueeze(dim=-1)
y = weight * X + bias


# * Data splitting
trainSplit = int(0.8 * len(X))


XTrain, YTrain = X[:trainSplit], y[:trainSplit]
XTest, YTest = X[trainSplit:], y[trainSplit:]

# * Shift to GPU
# XTrain.to("cuda:0")
# YTrain.to("cuda:0")
# XTest.to("cuda:0")
# YTest.to("cuda:0")

# * Plot data
def plotPredictions(trainData = XTrain, trainLabels = YTrain, testData = XTest, testLabels = YTest, predictions=None):

  plt.figure(figsize=(10, 7))
  plt.scatter(trainData, trainLabels, c="b", s=4, label="Training Data")
  plt.scatter(testData, testLabels, c="r", s=4, label="Testing Data")

  if predictions != None :
    plt.scatter(testData, predictions, c="g", s=4, label="Predictions")

  plt.legend(prop={"size": 14 })
  plt.show()

# plotPredictions()


# * Model Creation
class LinearRegressionV2(nn.Module): 
  
  def __init__(self):
    super().__init__() 
    # 1* linear parameters  ( y = ax + b )
    self.linearLayer = nn.Linear(
      in_features=1,                   
      out_features=1              
    )

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.linearLayer(x)

Model = LinearRegressionV2()
# Model.to("cuda:0")
# * Model training loop

lossFn = nn.L1Loss()               
optimizer = torch.optim.SGD(
  params=Model.parameters(),  
  lr=0.01                        
)       


epochs = 2000           

for epoch in range(epochs):
  # 1* Forward pass
  yPred = Model(XTrain)

  # 1* Calculate loss
  loss = lossFn(yPred, YTrain)

  # 1* Optimize
  optimizer.zero_grad()

  # 1* Backpropagation
  loss.backward()

  # 1* optimizer step (chose if to increase or not)
  optimizer.step()

  # 1* Testing
  Model.eval()
  with torch.inference_mode():
    testPred = Model(XTest)
    testLoss = lossFn(testPred, YTest)
  
  # if epoch % 10 == 0:
  #   print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {testLoss}")
  # 1* Visualize trained data
plotPredictions(predictions=testPred)

  # 1* Save / Load trained model