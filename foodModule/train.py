import os 
import torch
import requests
import zipfile
from pathlib import Path
import torchvision
from torchvision import transforms
from torchinfo import summary

# # modules
import dataSetup
import modelBuilder
import engine
import utils
from matplotlib import pyplot as plt
# # device agnostic code
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# # super params
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.01

# # transformers
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])
manualTransform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[.485, .456, .406],
    std=[.229, .224, .225]
  )
])
# dirs
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

dataPath = Path("data/")
imagePath = dataPath / "pizza_steak_sushi"

if not imagePath.is_dir():
  imagePath.mkdir(parents=True, exist_ok=True)
  # # 1 download data

  with open(dataPath / "pizza_steak_sushi.zip", "wb") as f:
    req = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    f.write(req.content)
    
  # # 1 unzip data
  with zipfile.ZipFile(dataPath / "pizza_steak_sushi.zip", "r") as zipRef:
    zipRef.extractall(imagePath)
  os.remove(dataPath / "pizza_steak_sushi.zip")
trainDir = imagePath / "train"
testDir = imagePath / "test"

# # external transforms
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
autoTransform= weights.transforms()  
externalModel = torchvision.models.efficientnet_b0(weights=weights).to(device)

# # data
trainDataLoader, testDataLoader, classNames = dataSetup.createDataLoaders(
  trainDir=train_dir,
  testDir=test_dir,
  transform=autoTransform,
  batchSize=BATCH_SIZE
)

# # adapt the imported model to match the wanted model
for param in externalModel.features.parameters():
  param.requires_grad = False

externalModel.classifier = torch.nn.Sequential(
  torch.nn.Dropout(p=0.2, inplace=True), 
  torch.nn.Linear(
    in_features=1280, 
    out_features=len(classNames), 
    bias=True)
).to(device)

# # Sum = summary(
# #   model=externalModel, 
# #   input_size=(32, 3, 224, 224), # # make sure this is "input_size", not "input_shape"
# #   col_names=["input_size", "output_size", "num_params", "trainable"],
# #   col_width=20,
# #   row_settings=["var_names"]
# # )
# # print(Sum)

# import sys
# sys.exit()
# # model
# model = modelBuilder.TinyVGG(
#   inputShape=3, hiddenUnits=HIDDEN_UNITS, outputShape=len(classNames)
# ).to(device)

lossfn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
  params=externalModel.parameters(),
  lr=LEARNING_RATE
)
# # train
from timeit import default_timer as timer 
if __name__ == '__main__':
  start_time = timer()

  # result = engine.train(
  #   model=externalModel,
  #   trainDataloader=trainDataLoader,
  #   testDataloader=testDataLoader,
  #   lossfn=lossfn,
  #   optimizer=optimizer,
  #   device=device,
  #   epochs=NUM_EPOCHS
  # )

  # & Saves a PyTorch obj
  # PATH.mkdir(parents=True, exist_ok=True,)
  # PATH = Path("TrainedModels")
  # PATH_NAME = "ExportedModelV1.pth"
  # SAVE_PATH = PATH / PATH_NAME
  # torch.save(obj=externalModel.state_dict(), f=SAVE_PATH)                         
  # & Load model
  # externalModel.load_state_dict(torch.load(f=SAVE_PATH)) 

  # Prediction & Plot
  from typing import List, Tuple

  from PIL import Image
  def predPlotImge(
    model:torch.nn.Module,
    imagePath: str,
    classNames: List[str],
    imageSize: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = manualTransform,
    device: torch.device=device
  ):
    img = Image.open(imagePath)

    model.to(device)
    model.eval()
    with torch.inference_mode():
      transformedImage = transform(img).unsqueeze(dim=0)

      targetImagePred = model(transformedImage.to(device))
    
    targetImageProbs = torch.softmax(targetImagePred, dim=1)
    targetImageLabel = torch.argmax(targetImageProbs, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {classNames[targetImageLabel]} | Prob: {targetImageProbs.max():.3f}")
    plt.axis(False)
    plt.show()

  predPlotImge(
    model=externalModel,
    imagePath="data\\pizza_steak_sushi\\steak.jpg",
    classNames=classNames,
  )

