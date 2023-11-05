# VGG16 Pretrained Model to do Transfer Learning
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt

# Load Model
torch_vgg16 = models.vgg16_bn(weights='VGG16_BN_Weights.IMAGENET1K_V1')
torch_vgg16.eval()

# Define function to load iamge
def load_image(image_name):
    # The VGG model expects a colored image of size 224x224
    image_path = './data/'+ image_name + '.jpg'
    image = cv2.cvtColor(src=cv2.imread(image_path), code=cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    
    # Normalize Image for VGG16 Model
    preprocess = transforms.Compose([
        transforms.ToPILImage(),                 # Convert np array to PILImage
        transforms.Resize(256),                  # Resize to a standard size
        transforms.CenterCrop(224),              # Crop the center
        transforms.ToTensor(),                   # Convert to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image = preprocess(image)
    image = np.expand_dims(image, axis=0)
    return image

image1 = load_image(image_name='01')


# Load classes
imagenet_classes = np.loadtxt(fname="./data/imagenet_2012.txt", dtype=str, delimiter="\n")

# Evaluate the model for a single image
with torch.no_grad():
    print("Make prediction for a single image")
    predictions = torch_vgg16(torch.tensor(image1))
    print(predictions.shape)
    
    result_index = np.argmax(predictions)
    print("Result:", result_index, imagenet_classes[result_index])
    
    
   