import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def trainStep(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  lossfn: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.device
):
  model.train()
  trainLoss, trainAcc = 0, 0

  for batch, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)

    yPred = model(x)
    loss = lossfn(yPred, y)
    trainLoss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    yPredClass = torch.argmax(torch.softmax(yPred, dim=1), dim=1)
    trainAcc += (yPredClass == y).sum().item()/len(yPred)

  # stats
  trainLoss /= len(dataloader)
  trainAcc /= len(dataloader)
  return trainLoss, trainAcc

def testStep(
  model: torch.nn.Module,
  dataloader: torch.utils.data.DataLoader,
  lossfn: torch.nn.Module,
  device: torch.device
):
  model.eval()
  testLoss, testAcc = 0, 0

  with torch.inference_mode():
    for batch, (x, y) in enumerate(dataloader):
      x, y = x.to(device), y.to(device)

      testPredLogits = model(x)

      loss = lossfn(testPredLogits, y)
      testLoss += loss.item()
      testPredLabels = testPredLogits.argmax(dim=1)
      testAcc += (testPredLabels == y).sum().item()/len(testPredLabels)

    # stats
    testLoss /= len(dataloader)
    testAcc /= len(dataloader)
    return testLoss, testAcc

def train(
  model: torch.nn.Module,
  trainDataloader: torch.utils.data.DataLoader,
  testDataloader: torch.utils.data.DataLoader,
  lossfn: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  epochs:int,
  device: torch.device,
):
  results = {
    "trainLoss":[],
    "trainAcc":[],
    "testLoss":[],
    "testAcc":[]
  }

  # #train loop
  for ep in tqdm(range(epochs)):
    trainLoss, trainAcc = trainStep(
      model=model,
      dataloader=trainDataloader,
      lossfn=lossfn,
      optimizer=optimizer,
      device=device
    )
    testLoss, testAcc = testStep(
      model=model,
      dataloader=testDataloader,
      lossfn=lossfn,
      device=device
    )

    # stats
    results["trainLoss"].append(trainLoss)
    results["trainAcc"].append(trainAcc)
    results["testLoss"].append(testLoss)
    results["testAcc"].append(testAcc)
  return results