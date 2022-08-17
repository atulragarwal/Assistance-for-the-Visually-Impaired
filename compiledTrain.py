import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from skimage.io import imread_collection
from skimage import data, img_as_float, io, exposure
from customData import CustomImageDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from noteModel import NeuralNetwork
from PIL import Image

batchSize = 1
epochs =5
inChannels = 3
imgHeight = 256
imgWidth = 256
noClasses = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# net = NeuralNetwork().to(device)
# print(model)



# pathTenNew = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Ten New/")]
# pathTen = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Ten Old/")]
# pathFiftyNew = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Fifety New/")]
# pathFiftyOld = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Fifety Old/")]
# path5Hundred = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Five Hundred/")]
# path100New = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Hundred New/")]
# path100Old = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Hundred Old/")]
# pathTwenty = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Twenty/")]
# path2Hundred = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Two Hundred/")]
# path2K = [f for f in os.listdir("Thai and Indian Currency Dataset256x256\Indian Currencies\Two Thousand/")]

completePath = "Compiled\All"
# categories = ["Avocado", "Banana", "Kiwi", "Lemon", "Lime", "Mango", "Nectarine", "Orange", "Papaya", "Passion-Fruit", "Peach", "Pineapple", "Plum", "Pomegranate", "Red-Grapefruit"]

# completeList = []
# for i in range(len(categories)):
#     string = completePath+categories[i]
#     for names in os.listdir(string):
#         temp = [names, i]
#         completeList.append(temp)

dataset = CustomImageDataset(annotations_file = "compiledLabel.csv", img_dir = completePath, transform = transforms.ToTensor())
# transform = transforms.ToTensor()

print(len(dataset))

trainSet, testSet = torch.utils.data.random_split(dataset, [270, 39])
train_dataloader = DataLoader(trainSet, batch_size=1, shuffle=True)
test_dataloader = DataLoader(testSet, batch_size=1, shuffle=True)

net = torchvision.models.googlenet(pretrained = True)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times
    loop = tqdm(train_dataloader)
    running_loss = 0.0
    for i, data in enumerate(loop, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        # print(inputs.shape)
        # forward + backward + optimize
        outputs = net(inputs)
        # print(outputs.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')
torch.save(net, "compiledModel.pth")


# dataiter = iter(test_dataloader)
# images, labels = dataiter.next()
# # print images
# # imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 109 test images: {100 * correct // total} %')




# l = ["FifetyOld.jpg", "FifetyNew.jpg", "FiveHundred.jpg", "HundredNew.jpg", "HundredOld.jpg", "TenNew.jpg", "TenOld.jpg", "Thousand.jpg", "THundred.jpg", "Twenty.jpg"]
# transform1 = transforms.Compose([
#         transforms.PILToTensor()
#     ])
# for i in l:
#     image = Image.open("New Folder/"+i)
#     img_tensor = transform1(image)
#     img_tensor = img_tensor.unsqueeze(0)

# # print(img_tensor.shape)
#     outputs = net(img_tensor)
# # print(output)
#     _, predicted = torch.max(outputs.data, 1)
#     print(predicted)
