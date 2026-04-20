import torch
from torch import nn

# Positional / Time Embedding
class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        x = x.float()  # safer for timestep tensors
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Skeleton adjacency
BONE_EDGES_16 = [
    (0, 1),                    # Head -> Neck
    (1, 2), (2, 3), (3, 4),    # Left Arm
    (1, 5), (5, 6), (6, 7),    # Right Arm
    (1, 8), (8, 9),            # Spine
    (9, 10), (10, 11), (11, 12),   # Left Leg
    (9, 13), (13, 14), (14, 15),   # Right Leg
]


def build_bone_attn_mask(num_nodes, edges, device, include_self=True, hops=1):
    """
    Returns additive attn_mask for nn.MultiheadAttention:
      allowed -> 0
      disallowed -> -inf
    """
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)

    for i, j in edges:
        if 0 <= i < num_nodes and 0 <= j < num_nodes:
            adj[i, j] = True
            adj[j, i] = True

    if include_self:
        adj.fill_diagonal_(True)

    # Expand to k-hop neighborhood if hops > 1
    if hops > 1:
        a = adj.float()
        reach = adj.clone()
        cur = adj.float()
        for _ in range(hops - 1):
            cur = (cur @ a > 0).float()
            reach = reach | cur.bool()
        adj = reach

    mask = torch.zeros(num_nodes, num_nodes, device=device)
    mask[~adj] = float("-inf")
    return mask


# Spatial block with optional adjacency mask
class MaskedSpatialBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


# Graph Denoiser (masked spatial + temporal attention)
class GraphDenoiserMasked(nn.Module):
    """
    Input/Output: [B, 90, 48] (T=90, 16 joints * 3 coords = 48)
    """
    def __init__(
        self,
        time_emb_dim=12,
        context_dim=512,
        num_classes=2,
        d_model=128,
        heads=4,
        depth=2,
        dropout=0.1,
        hops=1,
    ):
        super().__init__()

        self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.in_proj = nn.Linear(3, d_model)

        self.tc_proj = nn.Sequential(
            nn.Linear(time_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ctx_to_film = nn.Sequential(
            nn.Linear(context_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

        self.spatial_blocks = nn.ModuleList([
            MaskedSpatialBlock(d_model, heads, dropout) for _ in range(depth)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.out_proj = nn.Linear(d_model, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.hops = hops
        self.register_buffer("spatial_attn_mask", None, persistent=False)

    def _ensure_mask(self, N, device):
        if self.spatial_attn_mask is not None and self.spatial_attn_mask.shape[0] == N:
            return
        if N != 16:
            self.spatial_attn_mask = None
            return
        self.spatial_attn_mask = build_bone_attn_mask(
            num_nodes=16,
            edges=BONE_EDGES_16,
            device=device,
            include_self=True,
            hops=self.hops,
        )

    def forward(self, x, context, time, sensor_pred):
        B, T, Fdim = x.shape
        if Fdim % 3 != 0:
            raise ValueError(f"GraphDenoiserMasked expected Fdim divisible by 3, got {Fdim}")

        N = Fdim // 3
        x_nodes = x.view(B, T, N, 3)

        time_emb = self.time_mlp(time.to(x.device))
        class_emb = self.class_emb(sensor_pred.to(x.device))
        tc = self.tc_proj(time_emb + class_emb)

        film = self.ctx_to_film(context.to(x.device))
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1)

        h = self.in_proj(x_nodes)
        h = h + tc.view(B, 1, 1, -1)
        h = h * (1.0 + torch.tanh(gamma)) + beta

        self._ensure_mask(N, x.device)

        # Spatial attention over joints (per frame)
        h_sp = h.contiguous().view(B * T, N, -1)
        for blk in self.spatial_blocks:
            h_sp = blk(h_sp, attn_mask=self.spatial_attn_mask)
        h = h_sp.view(B, T, N, -1)

        # Temporal attention over frames (per joint)
        h_tm = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        h_tm = self.temporal_encoder(h_tm)
        h = h_tm.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()

        delta = self.out_proj(h)
        x_out = x_nodes + delta
        return x_out.view(B, T, Fdim)


class GraphEncoder(nn.Module):
    def __init__(self, d_model=128, heads=4, depth=1, dropout=0.1, hops=1):
        super().__init__()
        self.in_proj = nn.Linear(3, d_model)

        self.spatial_blocks = nn.ModuleList([
            MaskedSpatialBlock(d_model, heads, dropout) for _ in range(depth)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.out_proj = nn.Linear(d_model, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.hops = hops
        self.register_buffer("attn_mask", None, persistent=False)

    def _ensure_mask(self, N, device):
        if self.attn_mask is not None and self.attn_mask.shape[0] == N:
            return
        if N != 16:
            self.attn_mask = None
            return
        self.attn_mask = build_bone_attn_mask(
            num_nodes=16,
            edges=BONE_EDGES_16,
            device=device,
            include_self=True,
            hops=self.hops
        )

    def forward(self, x):
        B, T, Fdim = x.shape
        if Fdim % 3 != 0:
            raise ValueError(f"GraphEncoder expected Fdim divisible by 3, got {Fdim}")

        N = Fdim // 3
        x_nodes = x.view(B, T, N, 3)
        h = self.in_proj(x_nodes)

        self._ensure_mask(N, x.device)

        h_sp = h.contiguous().view(B * T, N, -1)
        for blk in self.spatial_blocks:
            h_sp = blk(h_sp, attn_mask=self.attn_mask)
        h = h_sp.view(B, T, N, -1)

        h_tm = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        h_tm = self.temporal_encoder(h_tm)
        h = h_tm.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()

        delta = self.out_proj(h)
        x_out = x_nodes + delta
        return x_out.view(B, T, Fdim)


class GraphDecoder(nn.Module):
    def __init__(self, d_model=128, heads=4, depth=1, dropout=0.1, hops=1):
        super().__init__()
        self.in_proj = nn.Linear(3, d_model)

        self.spatial_blocks = nn.ModuleList([
            MaskedSpatialBlock(d_model, heads, dropout) for _ in range(depth)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.out_proj = nn.Linear(d_model, 3)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.hops = hops
        self.register_buffer("attn_mask", None, persistent=False)

    def _ensure_mask(self, N, device):
        if self.attn_mask is not None and self.attn_mask.shape[0] == N:
            return
        if N != 16:
            self.attn_mask = None
            return
        self.attn_mask = build_bone_attn_mask(
            num_nodes=16,
            edges=BONE_EDGES_16,
            device=device,
            include_self=True,
            hops=self.hops
        )

    def forward(self, z):
        B, T, Fdim = z.shape
        if Fdim % 3 != 0:
            raise ValueError(f"GraphDecoder expected Fdim divisible by 3, got {Fdim}")

        N = Fdim // 3
        z_nodes = z.view(B, T, N, 3)
        h = self.in_proj(z_nodes)

        self._ensure_mask(N, z.device)

        h_sp = h.contiguous().view(B * T, N, -1)
        for blk in self.spatial_blocks:
            h_sp = blk(h_sp, attn_mask=self.attn_mask)
        h = h_sp.view(B, T, N, -1)

        h_tm = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        h_tm = self.temporal_encoder(h_tm)
        h = h_tm.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous()

        delta = self.out_proj(h)
        x_out = z_nodes + delta
        return x_out.view(B, T, Fdim)