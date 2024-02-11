import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout1 = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16 , out_channels=32, kernel_size=(3,3), stride=(1,1),padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout2 = nn.Dropout(p=0.1)
        self.bn2 = nn.BatchNorm2d(32) 

        self.conv3 = nn.Conv2d(in_channels=32 , out_channels=64, kernel_size=(3,3), stride=(1,1),padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout3 = nn.Dropout(p=0.1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64 , out_channels=128, kernel_size=(3,3), stride=(1,1),padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.dropout4 = nn.Dropout(p=0.1)
        self.bn4 = nn.BatchNorm2d(128) 

        self.fc1 = nn.Linear(128*7*7,512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout5 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(512,num_classes)

        
    def forward(self, x):
        x = self.dropout1(self.pool1(self.bn1(F.relu(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.bn2(F.relu(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.bn3(F.relu(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.bn4(F.relu(self.conv4(x)))))
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.dropout5(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x