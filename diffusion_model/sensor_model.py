import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads  = heads
        self.scale  = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, l = x.shape
        x   = x.permute(0, 2, 1)                                    # (b, l, c)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, l, self.heads, -1).transpose(1, 2), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out  = torch.matmul(attn, v).transpose(1, 2).reshape(b, l, -1)
        return self.to_out(out).permute(0, 2, 1)                     # (b, c, l)


class TemporalAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = TemporalAttention(dim)
        self.norm      = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)


class SensorConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        conv_channels,
        kernel_size,
        dropout=0.5,
        window_size=90,          # ← must match training window_size
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.conv1d = nn.Conv1d(
            input_size, conv_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn      = nn.BatchNorm1d(conv_channels)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.temporal_attention = TemporalAttentionBlock(conv_channels)

        self.lstm = nn.LSTM(
            conv_channels, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size * 2)

        # residual_projection: [B, window_size] → [B, hidden*2]
        # window_size must match the T dimension of the input sensor windows
        self.residual_projection = nn.Linear(window_size, hidden_size * 2)

    def forward(self, x):
        residual = x                             # [B, T, 3]

        x = x.permute(0, 2, 1)                  # [B, 3, T]
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.temporal_attention(x)
        x = x.permute(0, 2, 1)                  # [B, T, conv_channels]

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out    = self.fc(out[:, -1, :])          # [B, hidden*2]

        # Residual: take the last feature axis (z-axis) across all timesteps
        residual = residual[:, :, -1]            # [B, T]
        residual = self.residual_projection(residual)
        out = out + residual
        return out


class CombinedLSTMClassifier(nn.Module):
    def __init__(
        self,
        sensor_input_size,
        hidden_size,
        num_layers,
        num_classes,
        conv_channels,
        kernel_size,
        dropout=0.5,
        num_heads=8,
        window_size=90,          # ← passed through to SensorConvLSTMClassifier
    ):
        super().__init__()
        self.sensor1_conv_lstm = SensorConvLSTMClassifier(
            sensor_input_size, hidden_size, num_layers,
            conv_channels, kernel_size, dropout, window_size=window_size,
        )
        self.sensor2_conv_lstm = SensorConvLSTMClassifier(
            sensor_input_size, hidden_size, num_layers,
            conv_channels, kernel_size, dropout, window_size=window_size,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=num_heads, dropout=dropout
        )
        self.fc      = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sensor1_data, sensor2_data, return_attn_output=False):
        s1 = self.sensor1_conv_lstm(sensor1_data).unsqueeze(1)   # [B, 1, hidden*2]
        s2 = self.sensor2_conv_lstm(sensor2_data).unsqueeze(1)

        attn_out, _ = self.attention(s1, s2, s2)
        attn_out     = self.dropout(attn_out.squeeze(1))          # [B, hidden*2]

        return self.fc(attn_out), attn_out
