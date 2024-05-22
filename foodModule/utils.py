import torch 
from pathlib import Path
def saveModel(
  model:torch.nn.Module,
  targetDir: str,
  modelName: str,
):

  targetDirPath = Path(targetDir)
  if not targetDirPath.is_dir():
    targetDirPath.mkdir(
      parents=True,
      exist_ok=True
    )

  modelSavePath = targetDirPath / modelName

  torch.save(
    obj=model.state_dict(),
    f=modelSavePath
  )