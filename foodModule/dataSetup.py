import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

WORKERS = os.cpu_count()


# create data loaders
def createDataLoaders(
  trainDir: str,
  testDir: str,
  transform: transforms.Compose=None,
  batchSize: int=32,
  numWorkers: int=WORKERS
):
  trainData = datasets.ImageFolder(trainDir, transform=transform)
  testData = datasets.ImageFolder(testDir, transform=transform)
  classNames = trainData.classes

  trainDataLoader = DataLoader(
    trainData,
    batch_size=batchSize,
    shuffle=True,
    num_workers=numWorkers,
    pin_memory=True
  )
  
  testDataLoader = DataLoader(
    testData,
    batch_size=batchSize,
    shuffle=True,
    num_workers=numWorkers,
    pin_memory=True
  )
  return trainDataLoader, testDataLoader, classNames