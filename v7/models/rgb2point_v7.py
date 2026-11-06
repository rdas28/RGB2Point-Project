import torch
import torch.nn as nn
import timm


class RGB2PointV7(nn.Module):

    def __init__(
        self,
        num_coarse=256,
        expansion=4,
        hidden_dim=256,
        num_heads=4,
        num_decoder_layers=2,
        pretrained=True
    ):
        super().__init__()

        self.num_coarse = num_coarse
        self.expansion = expansion
        self.num_points = num_coarse * expansion

        self.encoder = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

        self.image_projection = nn.Linear(
            768,
            hidden_dim
        )

        self.coarse_queries = nn.Parameter(
            torch.randn(1, num_coarse, hidden_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        self.coarse_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

        self.expansion_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, expansion * 3)
        )

        self.offset_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images, return_coarse=False):
        batch_size = images.shape[0]

        with torch.no_grad():
            features = self.encoder.forward_features(images)

        patch_features = features[:, 1:, :]

        memory = self.image_projection(patch_features)

        queries = self.coarse_queries.expand(
            batch_size, -1, -1
        )

        decoded = self.decoder(
            tgt=queries,
            memory=memory
        )

        coarse = self.coarse_head(decoded)

        offsets = self.expansion_head(decoded)

        offsets = offsets.reshape(
            batch_size,
            self.num_coarse,
            self.expansion,
            3
        )

        offsets = torch.tanh(offsets)

        scale = 0.1 * torch.sigmoid(
            self.offset_scale
        ) / torch.sigmoid(
            torch.tensor(
                0.1,
                device=images.device
            )
        )

        fine = (
            coarse.unsqueeze(2)
            + scale * offsets
        )

        fine = fine.reshape(
            batch_size,
            self.num_points,
            3
        )

        if return_coarse:
            return fine, coarse

        return fine
