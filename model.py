import torch
import torch.nn as nn
import torch.nn.functional as F
from config import embedding_size
from config import num_classes


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(FaceRecognitionModel, self).__init__()
        
        # Feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Embedding layer
        self.fc1 = nn.Linear(128 * 20 * 20, embedding_size)
        
        # Classification layer
        if num_classes:
            self.fc2 = nn.Linear(embedding_size, num_classes)
        self.num_classes = num_classes
        
    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Embedding
        embedding = F.relu(self.fc1(x))
        
        # Classification if num_classes is set
        if self.num_classes:
            x = self.fc2(embedding)
            return x, embedding
        return embedding