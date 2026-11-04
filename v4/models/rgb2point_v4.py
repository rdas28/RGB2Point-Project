import torch
import torch.nn as nn
import timm


class RGB2PointV4(nn.Module):
    def __init__(
        self,
        num_points=1024,
        hidden_dim=256,
        num_heads=4,
        num_decoder_layers=2,
        pretrained=True,
    ):
        super().__init__()

        self.num_points = num_points

        # Pretrained image encoder
        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
        )

        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        # Project ViT features from 768 to 256
        self.image_projection = nn.Linear(
            768, hidden_dim
        )

        # Learnable point queries
        self.point_queries = nn.Parameter(
            torch.randn(1, num_points, hidden_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.point_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

    def train(self, mode=True):
        super().train(mode)
        # Keep frozen encoder in evaluation mode
        self.encoder.eval()
        return self

    def forward(self, images):
        batch_size = images.shape[0]

        # Frozen ViT feature extraction
        with torch.no_grad():
            features = self.encoder.forward_features(images)

        # Remove CLS token: 197 -> 196
        patch_features = features[:, 1:, :]

        memory = self.image_projection(patch_features)

        queries = self.point_queries.expand(
            batch_size, -1, -1
        )

        decoded = self.decoder(
            tgt=queries,
            memory=memory,
        )

        points = self.point_head(decoded)

        return points
