import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv6 = nn.Conv2d(16, 32, 3)

        # Replace conv7 with a 1x1 kernel instead of a 3x3 kernel
        self.conv7 = nn.Conv2d(32, 10, 1)  # 1x1 kernel

        # Define dropout with 10% probability
        self.dropout = nn.Dropout(p=0.15)  # 15% dropout rate

        # Define Batch Normalization layers
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for the output of conv1
        self.bn2 = nn.BatchNorm2d(16)  # BatchNorm for the output of conv2
        self.bn3 = nn.BatchNorm2d(32)  # BatchNorm for the output of conv3
        self.bn4 = nn.BatchNorm2d(16)  # BatchNorm for the output of conv4
        self.bn5 = nn.BatchNorm2d(16) # BatchNorm for the output of conv5
        self.bn6 = nn.BatchNorm2d(32) # BatchNorm for the output of conv6

        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)  # Reduces each feature map to 1x1

    def forward(self, x):
        # First convolution block with BatchNorm, ReLU, and Dropout
        x = self.pool1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))  # Conv1 -> BN -> ReLU -> Conv2 -> BN -> ReLU -> Pool1
        x = self.dropout(x)  # Apply dropout after activation

        # Second convolution block with BatchNorm, ReLU, and Dropout
        x = self.pool2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))  # Conv3 -> BN -> ReLU -> Conv4 -> BN -> ReLU -> Pool2
        x = self.dropout(x)  # Apply dropout after activation

        # Third convolution block with BatchNorm, ReLU, and Dropout
        x = F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x))))))  # Conv5 -> BN -> ReLU -> Conv6 -> BN -> ReLU
        x = self.dropout(x)  # Apply dropout after activation

        # Apply Global Average Pooling (GAP) to reduce each feature map to 1x1
        x = self.gap(x)  # Now the size of x is (batch_size, 128, 1, 1)

        # Flatten the output to feed into the final fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 128)

        # Final convolution layer (now with 1x1 kernel)
        # x = self.conv7(x.unsqueeze(-1).unsqueeze(-1))  # Convert the flattened vector into a form for conv7 (needs to be (batch_size, 128, 1, 1))

        # Return log softmax for classification output
        return F.log_softmax(x, dim=1)  # Log softmax for classification output