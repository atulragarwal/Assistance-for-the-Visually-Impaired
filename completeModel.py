import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


import numpy as np
# import pandas as pd
# import torchvision.transforms.functional as TF
import torch
# import torch.nn as nn
# import torch.optim as optim
import torchvision
# from skimage.io import imread_collection
# from skimage import data, img_as_float, io, exposure
from customData import CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from tqdm import tqdm
# from noteModel import NeuralNetwork
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from gtts import gTTS


compiledNet = torch.load("compiledModel.pth")
groceryNet = torch.load("groceryModel.pth")
obstacleNet = torch.load("obstacleModel.pth")
notesNet = torch.load("notesModel.pth")
image = Image.open("Compiled\Obstacles\door8.jpg")
transform = transforms.Compose([
    transforms.PILToTensor()
])
  

transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = transform(image)
img_tensor = img_tensor.unsqueeze(0)



outputs = compiledNet(img_tensor)
# print(output)
_, predicted = torch.max(outputs.data, 1)
print(predicted)
print(np.asarray(predicted)[0])
if(np.asarray(predicted)[0] == 0):
    newOutput = notesNet(img_tensor)
    # print(output)
    _, predictedNew = torch.max(newOutput.data, 1)
    print(predictedNew)
    temp = np.asarray(predictedNew)[0]
    print(np.asarray(predictedNew)[0])
    if(temp == 0):
        txt = "You have a 10 Rupee Note"
    if(temp == 1):
        txt = "You have a 10 Rupee Note"
    if(temp == 2):
        txt = "You have a 20 Rupee Note"
    if(temp == 3):
        txt = "You have a 50 Rupee Note"
    if(temp == 4):
        txt = "You have a 50 Rupee Note"
    if(temp == 5):
        txt = "You have a 500 Rupee Note"
    if(temp == 6):
        txt = "You have a 100 Rupee Note"
    if(temp == 7):
        txt = "You have a 100 Rupee Note"
    if(temp == 8):
        txt = "You have a 200 Rupee Note"
    if(temp == 9):
        txt = "You have a 2000 Rupee Note"
    myobj = gTTS(text=txt, lang="en", slow=False)
    myobj.save("welcome.mp3")
    os.system("start welcome.mp3")

if(np.asarray(predicted)[0] == 1):
    newOutput = groceryNet(img_tensor)
    # print(output)
    _, predictedNew = torch.max(newOutput.data, 1)
    print(predictedNew)
    temp = np.asarray(predictedNew)[0]
    print(np.asarray(predictedNew)[0])
    if(temp == 0):
        txt = "You have a Avocado"
    if(temp == 1):
        txt = "You have a Banana"
    if(temp == 2):
        txt = "You have a Kiwi"
    if(temp == 3):
        txt = "You have a Lemon"
    if(temp == 4):
        txt = "You have a Lime"
    if(temp == 5):
        txt = "You have a Mango"
    if(temp == 6):
        txt = "You have a Nectarine"
    if(temp == 7):
        txt = "You have a Orange"
    if(temp == 8):
        txt = "You have a Papaya"
    if(temp == 9):
        txt = "You have a Passion Fruit"
    if(temp == 10):
        txt = "You have a Peach"
    if(temp == 11):
        txt = "You have a Pineapple"
    if(temp == 12):
        txt = "You have a Plum"
    if(temp == 13):
        txt = "You have a Pomegranate"
    if(temp == 14):
        txt = "You have a Red Grape Fruit"
    myobj = gTTS(text=txt, lang="en", slow=False)
    myobj.save("welcome.mp3")
    os.system("start welcome.mp3")

if(np.asarray(predicted)[0] == 2):
    newOutput = obstacleNet(img_tensor)
    # print(output)
    _, predictedNew = torch.max(newOutput.data, 1)
    print(predictedNew)
    temp = np.asarray(predictedNew)[0] 
    print(np.asarray(predictedNew)[0])
    if(temp == 0):
        txt = "You are at an ATM"
    if(temp == 1):
        txt = "There is a bench"
    if(temp == 2):
        txt = "Caution, There is a bus in front of you"
    if(temp == 3):
        txt = "Caution. You may face harm"
    if(temp == 4):
        txt = "There is a church in front of you"
    if(temp == 5):
        txt = "Caution. You May Face Harm"
    if(temp == 6):
        txt = "Caution, There is a door in front of you. You May Face Harm"
    if(temp == 7):
        txt = "Caution. You May Face Harm"
    if(temp == 8):
        txt = "Caution, Lift in front of you. You May Face Harm"
    if(temp == 9):
        txt = "Caution. You May Face Harm"
    if(temp == 10):
        txt = "Caution, Staires. You May Face Harm"
    if(temp == 11):
        txt = "Caution, Washroom. You May Face Harm"
    if(temp == 12):
        txt = "Caution, wetfloor. You May Face Harm"
    #txt = "Caution. You May Face Harm"
    myobj = gTTS(text=txt, lang="en", slow=False)
    myobj.save("welcome.mp3")
    os.system("start welcome.mp3")


