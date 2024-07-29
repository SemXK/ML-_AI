from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
from matplotlib import pyplot as plt


# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open('download.jpg')

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()

plt.figure()
plt.imshow(image)
plt.title(f"Pred: {model.config.id2label[predicted_class_idx]}")
plt.axis(False)
plt.show()
# print("Predicted class:", model.config.id2label[predicted_class_idx])
