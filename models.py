import torch
import torch.nn as nn
from torchvision import models
import timm

class MultiTaskResNet18(nn.Module):
    def __init__(self, num_tasks=6, retrain = True):
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
    def __init__(self, num_tasks=6, retrain = True):
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
    

class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_tasks=6, retrain=True):
        super(MultiTaskEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet
        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        
        # Remove the fully connected layer from EfficientNet (replace with identity)
        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Define separate heads for each task
        self.heads = nn.ModuleList([nn.Linear(efficientnet.classifier[1].in_features, 1) for _ in range(num_tasks)])
        
        # Optionally freeze the backbone layers
        if not retrain:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Forward pass through the shared backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten features
        
        # Pass through each head for binary classification
        outputs = [head(x) for head in self.heads]
        
        return outputs


class UNIMultitask(nn.Module):
    def __init__(self, num_tasks, output_dim=1):
        super(UNIMultitask, self).__init__()
        # load pretrained model UNI, and then retrain block (23) onward
        base_model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

        for name, param in base_model.named_parameters():
            if "blocks" in name:
                block_idx = int(name.split(".")[1])
                if block_idx < 23:  # Freeze all blocks except the last one
                    param.requires_grad = False
            elif name in ["patch_embed.proj.weight", "patch_embed.proj.bias", "pos_drop.weight", "pos_drop.bias"]:
                param.requires_grad = False
            else:
                param.requires_grad = True  # Unfreeze "norm", "fc_norm", "head_drop", and "head"

        # Replace the original head with an identity layer (for feature extraction)
        base_model.head = nn.Identity()

        self.base_model = base_model
        self.num_tasks = num_tasks
        
        # Add multiple classification heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_model.num_features, 256),  # Intermediate hidden layer
                nn.ReLU(),
                nn.Linear(256, output_dim),
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        # Pass input through the base model up to the feature extractor
        features = self.base_model(x)
        
        # Pass extracted features through each head
        outputs = [head(features) for head in self.heads]
        return outputs
