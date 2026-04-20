import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=2., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop_rate, batch_first=True)
        self.drop_path = nn.Dropout(drop_path_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SkeletonTransformer(nn.Module):
    def __init__(self, input_size=48, embed_dim=128, num_heads=8, num_layers=4, num_classes=12, dropout=0.2):
        super(SkeletonTransformer, self).__init__()

        self.input_proj = nn.Linear(input_size, embed_dim)  # Project input to embedding size

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, drop_rate=dropout, attn_drop_rate=dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, skeleton_data):
        x = self.input_proj(skeleton_data)  # Convert input to embeddings

        # Pass through Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)

        # Take the last time step for classification
        x = x[:, -1, :]

        output = self.fc(x)
        return output