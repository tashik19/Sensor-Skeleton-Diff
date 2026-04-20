import torch
from torch import nn


# ── Positional / time embedding ───────────────────────────────────────────────

class JointPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for diffusion timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device   = x.device
        half_dim = self.dim // 2
        scale    = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        freq     = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb      = x[:, None] * freq[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ── Sensor context fusion (cross-attention bottleneck) ────────────────────────

class BoneContextFusion(nn.Module):
    def __init__(self, latent_dim, context_dim, sequence_length, num_heads=4):
        super().__init__()
        self.joint_proj   = nn.Linear(latent_dim, latent_dim // 2)
        self.context_proj = nn.Linear(context_dim, (latent_dim // 2) * sequence_length)
        self.attn  = nn.MultiheadAttention(
            embed_dim=latent_dim // 2, num_heads=num_heads, batch_first=True
        )
        self.norm   = nn.LayerNorm(latent_dim // 2)
        self.refine = nn.Sequential(
            nn.Linear(latent_dim // 2, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim // 2),
        )

    def forward(self, joint_feat, sensor_ctx):
        joint_feat = self.joint_proj(joint_feat)
        sensor_ctx = self.context_proj(sensor_ctx)
        sensor_ctx = sensor_ctx.view(joint_feat.size(0), joint_feat.size(1), -1)
        attn_out, _ = self.attn(joint_feat, sensor_ctx, sensor_ctx)
        joint_feat  = self.norm(attn_out + joint_feat)
        return joint_feat + self.refine(joint_feat)


# ── Building blocks ───────────────────────────────────────────────────────────

class JointConnectivityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv      = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn        = nn.BatchNorm1d(out_channels)
        self.act       = nn.ReLU()

    def forward(self, x, time_emb):
        t = self.time_proj(time_emb).unsqueeze(-1)
        return self.act(self.bn(self.conv(x) + t))


class TemporalJointBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
        )

    def forward(self, x):
        x, _ = self.lstm(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class BoneDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class BoneUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.deconv(x)


# ── Main denoiser (SSDL-style UNet with renamed components) ───────────────────
# NOTE: graph_modules.py contains GraphDenoiserMasked / GraphEncoder / GraphDecoder
# which provide graph-attention processing. BoneAttentionDenoiser uses the same
# UNet backbone augmented with bone-aware naming and cross-attention fusion.

class BoneAttentionDenoiser(nn.Module):
    """
    1-D UNet denoiser conditioned on sensor context (cross-attention) and
    diffusion timestep + class label.

    Input / Output shape: [B, skeleton_dim, 48]
    where skeleton_dim == window_size (temporal frames treated as channels)
    and 48 == 16 joints × 3 coords.
    """

    def __init__(
        self,
        skeleton_dim=32,        # window_size (default 32)
        time_emb_dim=12,
        context_dim=256,
        hidden_dim=128,
        num_classes=12,
    ):
        super().__init__()

        self.class_emb  = nn.Embedding(num_classes, time_emb_dim)
        self.time_embed = nn.Sequential(
            JointPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_proj = nn.Conv1d(skeleton_dim, hidden_dim, kernel_size=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_temporal_1 = TemporalJointBlock(hidden_dim,      hidden_dim)
        self.enc_spatial_1  = JointConnectivityBlock(hidden_dim * 2, hidden_dim * 2, time_emb_dim)
        self.enc_down_1     = BoneDownsample(hidden_dim * 2, hidden_dim * 4)

        self.enc_temporal_2 = TemporalJointBlock(hidden_dim * 4,  hidden_dim * 4)
        self.enc_spatial_2  = JointConnectivityBlock(hidden_dim * 8, hidden_dim * 4, time_emb_dim)
        self.enc_down_2     = BoneDownsample(hidden_dim * 4, hidden_dim * 8)

        self.enc_temporal_3 = TemporalJointBlock(hidden_dim * 8,  hidden_dim * 8)
        self.enc_spatial_3  = JointConnectivityBlock(hidden_dim * 16, hidden_dim * 16, time_emb_dim)

        # ── Bottleneck: sensor context fusion via cross-attention ─────────────
        # sequence_length = joint_features // 2 // 2 = 48//4 = 12
        # The downsampling is on the joint-feature dimension (L=48), NOT window_size.
        # window_size is the channel dimension (C) in Conv1d, so it never gets halved.
        # This is 12 for all window_size values.
        self.bone_context_fusion = BoneContextFusion(
            latent_dim=hidden_dim * 16,
            context_dim=context_dim,
            sequence_length=12,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_spatial_1  = JointConnectivityBlock(hidden_dim * 8, hidden_dim * 4, time_emb_dim)
        self.dec_temporal_1 = TemporalJointBlock(hidden_dim * 4,  hidden_dim * 4)
        self.dec_up_1       = BoneUpsample(hidden_dim * 8, hidden_dim * 4)

        self.dec_spatial_2  = JointConnectivityBlock(hidden_dim * 4, hidden_dim * 2, time_emb_dim)
        self.dec_temporal_2 = TemporalJointBlock(hidden_dim * 2,  hidden_dim * 2)
        self.dec_up_2       = BoneUpsample(hidden_dim * 4, hidden_dim * 2)

        self.dec_spatial_3  = JointConnectivityBlock(hidden_dim * 2, hidden_dim, time_emb_dim)
        # BiLSTM(hidden_dim, skeleton_dim) → skeleton_dim * 2 output channels
        self.dec_temporal_3 = TemporalJointBlock(hidden_dim, skeleton_dim)

        # FIX: output channels = skeleton_dim * 2 (bidirectional LSTM doubles channels)
        # Old code hardcoded 180 (= 90*2) which broke when skeleton_dim != 90
        self.output_proj = nn.Conv1d(skeleton_dim * 2, skeleton_dim, kernel_size=1)

    def forward(self, x, context, time, sensor_pred):
        x           = x.to(context.device)
        time        = time.to(context.device)
        sensor_pred = sensor_pred.to(context.device)

        time_emb = self.time_embed(time) + self.class_emb(sensor_pred)
        x = self.input_proj(x)

        x = self.enc_temporal_1(x)
        x = self.enc_spatial_1(x, time_emb)
        x = self.enc_down_1(x)

        x = self.enc_temporal_2(x)
        x = self.enc_spatial_2(x, time_emb)
        x = self.enc_down_2(x)

        x = self.enc_temporal_3(x)
        x = self.enc_spatial_3(x, time_emb)

        # Bottleneck cross-attention with sensor context
        x = self.bone_context_fusion(x.permute(0, 2, 1), context).permute(0, 2, 1)

        x = self.dec_spatial_1(x, time_emb)
        x = self.dec_temporal_1(x)
        x = self.dec_up_1(x)

        x = self.dec_spatial_2(x, time_emb)
        x = self.dec_temporal_2(x)
        x = self.dec_up_2(x)

        x = self.dec_spatial_3(x, time_emb)
        x = self.dec_temporal_3(x)

        return self.output_proj(x)


# ── Diffusion1D wrapper ───────────────────────────────────────────────────────

class Diffusion1D(nn.Module):
    """
    Full diffusion model wrapping BoneAttentionDenoiser.
    Input / Output: [B, window_size, 48]  (same shape as skeleton windows)
    """

    def __init__(self, skeleton_dim=32, num_classes=12):
        super().__init__()

        self.bone_denoiser = BoneAttentionDenoiser(
            skeleton_dim=skeleton_dim,
            time_emb_dim=12,
            context_dim=256,
            hidden_dim=128,
            num_classes=num_classes,
        )

        self.output_proj = nn.Conv1d(skeleton_dim, skeleton_dim, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, context, time, sensor_pred):
        x           = x.to(context.device)
        time        = time.to(context.device)
        sensor_pred = sensor_pred.to(context.device)
        out = self.bone_denoiser(x, context, time, sensor_pred)
        return self.output_proj(out)
