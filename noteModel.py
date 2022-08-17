import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        #256 256 3
        #254 254 8
        #127 127 8
        #125 125 16
        #62 62 16
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        #Shape= (4,12,256,256)
        self.bn1=nn.BatchNorm2d(num_features=12)
        #Shape= (256,12,256,256)
        self.relu1=nn.ReLU()
        #Shape= (256,12,256,256)
        
        self.pool=nn.MaxPool2d(kernel_size=2)
        #Reduce the image size be factor 2
        #Shape= (256,12,128,128)
        
        
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        #Shape= (256,20,128,128)
        self.relu2=nn.ReLU()
        #Shape= (256,20,128,128)
        
        
        
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        #Shape= (256,32,128,128)
        self.bn3=nn.BatchNorm2d(num_features=32)
        #Shape= (256,32,128,128)
        self.relu3=nn.ReLU()
        #Shape= (256,32,128,128)
        
        
        self.fc=nn.Linear(in_features=128 * 128 * 32,out_features=10)
            # nn.ReLU(),
            # nn.Linear(32,10),

    def forward(self, x):
        # x = self.flatten(x)
        # logits = self.linear_relu_stack(x)
        # logits = logits.view(-1, 62)

        # return logits
        output=self.conv1(x)
        output=self.bn1(output)
        output=self.relu1(output)
            
        output=self.pool(output)
            
        output=self.conv2(output)
        output=self.relu2(output)
            
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)
        output=output.view(-1,32*128*128)
            
            
        output=self.fc(output)
            
        return output


