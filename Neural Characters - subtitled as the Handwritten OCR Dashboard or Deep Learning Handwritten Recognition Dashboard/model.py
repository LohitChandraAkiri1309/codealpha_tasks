import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, num_classes=10):
        """
        Lightweight Convolutional Neural Network (CNN) for character recognition.
        Works for 28x28 grayscale inputs.
        """
        super(CharCNN, self).__init__()
        
        # Convolutional Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Convolutional Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Max Pooling: reduces dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully Connected Layers
        # After two MaxPool2d operations, the 28x28 image size becomes:
        # 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: [Batch, 1, 28, 28]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten for dense layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output Logits
        x = self.fc2(x)
        return x
