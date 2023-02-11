'''
Implementation of AlexNet as described in the paper. 
The local response normalization layers were omitted due to reports of its impact being minimal.

Paper: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.max = nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()
        
        self.c1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4)
        self.c2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=1, padding=2)

        self.c3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=1, padding=1) 
        self.c4 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=1, padding=1)
        
        self.c5 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=1, padding=1)

        # Initialize the weights of the Conv2d layers using a zero-mean Gaussian distribution
        i = 1
        for m in self.modules():
          if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if i < 2:
                nn.init.constant_(m.bias, 0)
            else:
                nn.init.constant_(m.bias, 1)
            i += 1
            
    def forward(self, x):
        x = self.c1(x)
        x = self.max(self.relu(x))

        x = self.c2(x)
        x = self.max(self.relu(x))

        x = self.c3(x)
        x = self.c4(self.relu(x))
        x = self.c5(self.relu(x))
        
        x = self.max(self.relu(x))
        x = self.dropout(x)
        return x
    
    
class FullyConnected(nn.Module):
    def __init__(self, num_classes=1000):
        super(FullyConnected, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(self.relu(x))
        x = self.fc3(self.relu(x))
        return self.softmax(x)
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.fe = FeatureExtractor()
        self.fc = FullyConnected(num_classes=num_classes)

    def forward(self, x):
        x = self.fe(x)
        x = self.fc(x.view(-1, 256 * 6 * 6))
        return x
