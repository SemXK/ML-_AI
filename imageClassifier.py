# * pyTorch
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, List
import torchinfo
from torchinfo import summary as torchSum
import torchvision
# * File imports
import requests
import zipfile
import pathlib
from pathlib import Path
import os

# * Visualization
import random
from PIL import Image
from tqdm.auto import tqdm

# * Device - agnostic
device = "cuda" if torch.cuda.is_available() else "cpu"

# * Transform data into tensors
dataTransform = transforms.Compose([
  transforms.Resize(size=(64, 64)),
  transforms.RandomHorizontalFlip(p=.5),
  transforms.ToTensor()
]) 

# * Load Datasets
dataPath = Path("data/")
imagePath = dataPath / "Food"
trainDir = imagePath / "train"
testDir = imagePath / "test"


trainData = datasets.ImageFolder(
  root=trainDir,
  transform=dataTransform,
  target_transform=None
)
testData = datasets.ImageFolder(
  root=testDir,
  transform=dataTransform,
)
classNames = trainData.classes
classDict = trainData.class_to_idx

BATCH_SIZE = 32
trainDataLoader = DataLoader(
  dataset=trainData,
  batch_size=BATCH_SIZE,
  num_workers=os.cpu_count(),
  shuffle=True
)
testDataLoader = DataLoader(
  dataset=testData,
  batch_size=BATCH_SIZE,
  num_workers=os.cpu_count(),
  shuffle=False
)
# * ML Model (using train/test data from previous sections)
class TinyVGG(nn.Module):

  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(
          in_channels=input_shape, 
          out_channels=hidden_units, 
          kernel_size=3, # how big is the square that's going over the image?
          stride=1, # default
          padding=1
        ), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
        nn.ReLU(),
        nn.Conv2d(
          in_channels=hidden_units, 
          out_channels=hidden_units,
          kernel_size=3,
          stride=1,
          padding=1
        ),
        nn.ReLU(),
        nn.MaxPool2d(
          kernel_size=2,
          stride=2
        ) # default stride value is same as kernel_size
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        # Where did this in_features shape come from? 
        # It's because each layer of our network compresses and changes the shape of our inputs data.
        nn.Linear(in_features=hidden_units*16*16,
                  out_features=output_shape)
    )
  
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      # print(x.shape)
      x = self.conv_block_2(x)
      # print(x.shape)
      x = self.classifier(x)
      # print(x.shape)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


# * Train Test steps loop
def train_step(
    model: torch.nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
  ):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
    


# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
          model=model,
          dataloader=train_dataloader,
          loss_fn=loss_fn,
          optimizer=optimizer
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn
            )
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results
# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Recreate an instance of TinyVGG
model_0 = TinyVGG(input_shape=3, # number of color channels (3 for RGB) 
                  hidden_units=10, 
                  output_shape=len(trainData.classes)).to(device)

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_0 
if __name__ == '__main__':
  model_0_results = train(
    model=model_0, 
    train_dataloader=trainDataLoader,
    test_dataloader=testDataLoader,
    optimizer=optimizer,
    loss_fn=loss_fn, 
    epochs=NUM_EPOCHS
    )

  # End the timer and print out how long it took
  end_time = timer()
  print(f"Total training time: {end_time-start_time:.3f} seconds")