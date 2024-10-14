import torch
import torch.nn as nn
from torchvision import models
import timm


class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_tasks=6):
        super(MultiTaskEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Free/Unfreeze all layers
        for param in efficientnet.parameters():
            param.requires_grad = False
        
        # Backbone (remove final classification layer)
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Heads for binary classification for each task
        self.heads = nn.ModuleList([nn.Linear(efficientnet.classifier[1].in_features, 1) for _ in range(num_tasks)])
    
    def forward(self, x):
        # Extract features using the EfficientNet backbone
        features = self.backbone(x)
        features = features.flatten(start_dim=1)
        
        # Apply each head to the same features
        outputs = [head(features) for head in self.heads]
        
        return outputs

    def get_embedding(self, x):
        """Forward pass through the backbone to get embeddings (without heads)."""
        with torch.no_grad():
            features = self.backbone(x)
            features = features.flatten(start_dim=1)
        return features

class MultiTaskResNet18(nn.Module):
    def __init__(self, num_tasks=6, retrain = False):
        super(MultiTaskResNet18, self).__init__()
        
        # Load pre-trained ResNet
        resnet = models.resnet18(weights='DEFAULT')
        
        # Remove the fully connected layer from ResNet18 (replace with identity)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Define separate heads for each task
        self.heads = nn.ModuleList([nn.Linear(resnet.fc.in_features, 1) for _ in range(num_tasks)])
        
        # Optionally freeze the backbone layers
        if not retrain:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Forward pass through the shared backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten features
        
        # Pass through each head for binary classification
        outputs = [torch.sigmoid(head(x)) for head in self.heads]
        
        return outputs

    # def get_embedding(self, x):
    #     """Forward pass through the backbone to get embeddings (without heads)."""
    #     with torch.no_grad():
    #         features = self.backbone(x)
    #         features = features.flatten(start_dim=1)
    #     return features

class MultiTaskResNet50(nn.Module):
    def __init__(self, num_tasks=6, retrain = False):
        super(MultiTaskResNet50, self).__init__()
        
        # Load pre-trained ResNet
        resnet = models.resnet50(weights='DEFAULT')
        
        # Remove the fully connected layer from ResNet50 (replace with identity)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Define separate heads for each task
        self.heads = nn.ModuleList([nn.Linear(resnet.fc.in_features, 1) for _ in range(num_tasks)])
        
        # Optionally freeze the backbone layers
        if not retrain:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Forward pass through the shared backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten features
        
        # Pass through each head for binary classification
        outputs = [torch.sigmoid(head(x)) for head in self.heads]
        
        return outputs
    
    