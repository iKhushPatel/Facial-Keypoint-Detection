## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # Covolutional Layers
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2)

        # Maxpooling Layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 36864, out_features = 1000) 
        self.fc2 = nn.Linear(in_features = 1000,    out_features = 1000)
        self.fc3 = nn.Linear(in_features = 1000,    out_features = 136) 

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.6)


    def forward(self, x):

        x = self.drop1(self.pool(F.relu(self.conv1(x))))
        x = self.drop2(self.pool(F.relu(self.conv2(x))))
        x = self.drop3(self.pool(F.relu(self.conv3(x))))
        x = self.drop4(self.pool(F.relu(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = self.drop5(F.relu(self.fc1(x)))        
        x = self.drop6(F.relu(self.fc2(x)))
        
        x = self.fc3(x)
        
        return x