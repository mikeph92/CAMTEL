import torch
import torch.nn as nn
import timm

class UNIMultitask(nn.Module):
    def __init__(self, num_tasks, output_dim=1):
        super(UNIMultitask, self).__init__()
        # Load UNI model from MahmoodLab
        base_model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        # Freeze early layers for efficiency
        for name, param in base_model.named_parameters():
            if "blocks" in name:
                block_idx = int(name.split(".")[1])
                if block_idx < 23:
                    param.requires_grad = False
            elif name in ["patch_embed.proj.weight", "patch_embed.proj.bias", "pos_drop.weight", "pos_drop.bias"]:
                param.requires_grad = False
            else:
                param.requires_grad = True
        base_model.head = nn.Identity()
        self.base_model = base_model
        self.num_tasks = num_tasks
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_model.num_features, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
            ) for _ in range(num_tasks)
        ])
        # Attention module for dynamic head selection
        self.attention = nn.Sequential(
            nn.Linear(base_model.num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_tasks),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.base_model(x)
        cluster_probs = self.attention(features)  # [batch_size, num_tasks]
        outputs = [head(features) for head in self.heads]  # List of [batch_size, output_dim]
        return outputs, cluster_probs

    