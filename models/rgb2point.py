import timm
import torch
import torch.nn as nn

class RGB2Point(nn.Module):
    def __init__(self):
        super(RGB2Point, self).__init__()
        
        # Load pretrained ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.head = nn.Identity()

        # Contextual Feature Integrator (CFI)
        self.cfi = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024)
        )

        # Geometric Projection Module (GPM)
        self.gpm = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 3 * 1024)  # 1024 points, each with x,y,z
        )

    def forward(self, x):
        features = self.vit(x)
        features = self.cfi(features)
        output = self.gpm(features)
        output = output.view(x.size(0), 1024, 3)
        return output
