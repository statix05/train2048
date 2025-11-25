import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = F.relu(self.conv1(x))
        attention = torch.sigmoid(self.conv2(attention))
        return x * attention

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int, size: int = 4):
        super().__init__()
        pe = torch.zeros(channels, size, size)
        for i in range(size):
            for j in range(size):
                for c in range(channels):
                    if c % 2 == 0:
                        pe[c, i, j] = np.sin(i / (10000 ** (c / channels)))
                    else:
                        pe[c, i, j] = np.cos(j / (10000 ** ((c-1) / channels)))
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe.unsqueeze(0)

# --- Models ---

class SimpleDQN(nn.Module):
    """Classic MLP architecture"""
    def __init__(self, board_size: int = 4, n_features: int = 9, n_actions: int = 4):
        super().__init__()
        input_size = (board_size * board_size) + n_features
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # state: [batch, 4, 4] -> [batch, 16]
        batch_size = state.size(0)
        flat_state = state.reshape(batch_size, -1)
        combined = torch.cat([flat_state, features], dim=1)
        return self.net(combined)

class ConvDQN(nn.Module):
    """Standard CNN without fancy attention or dueling"""
    def __init__(self, board_size: int = 4, n_features: int = 9, n_actions: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.fc1 = nn.Linear(128 * 6 * 6 + n_features, 256) # Actual size: 4x4 -> pad 1 -> 5x5 -> pad 1 -> 6x6
        # Actual size check: 4x4 -> pad 1 -> 6x6 -> kernel 2 -> 5x5
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = state.unsqueeze(1) # Add channel
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        combined = torch.cat([x, features], dim=1)
        x = F.relu(self.fc1(combined))
        return self.fc2(x)

class DuelingDQN(nn.Module):
    """The original advanced architecture (updated with GELU and larger layers)"""
    def __init__(self, board_size: int = 4, n_features: int = 9, n_actions: int = 4):
        super().__init__()
        self.tile_embedding = nn.Conv2d(1, 64, 1)
        self.pos_encoding = PositionalEncoding2D(64, board_size)
        self.conv_block1 = ConvBlock(64, 128)
        self.attention1 = SpatialAttention(128)
        self.conv_block2 = ConvBlock(128, 256)
        self.attention2 = SpatialAttention(256)
        self.conv_block3 = ConvBlock(256, 256)
        self.attention3 = SpatialAttention(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = state.unsqueeze(1)
        x = F.relu(self.tile_embedding(x))
        x = self.pos_encoding(x)
        x = self.attention1(self.conv_block1(x))
        x = self.attention2(self.conv_block2(x))
        x = self.attention3(self.conv_block3(x))
        x = self.global_pool(x).view(state.size(0), -1)
        feat = self.feature_encoder(features)
        combined = torch.cat([x, feat], dim=1)
        val = self.value_stream(combined)
        adv = self.advantage_stream(combined)
        return val + adv - adv.mean(dim=1, keepdim=True)

class HybridDQN(nn.Module):
    """Experimental Hybrid: Multi-Scale Conv + Dense"""
    def __init__(self, board_size: int = 4, n_features: int = 9, n_actions: int = 4):
        super().__init__()
        # Path A: Global view (Dense)
        self.flat_path = nn.Sequential(
            nn.Linear(board_size*board_size, 128),
            nn.ReLU()
        )
        # Path B: Local patterns (Conv)
        self.conv_path = nn.Sequential(
            nn.Conv2d(1, 32, 2), # 3x3 output
            nn.ReLU(),
            nn.Conv2d(32, 64, 2), # 2x2 output
            nn.ReLU()
        )
        
        combined_size = 128 + (64 * 2 * 2) + n_features
        self.head = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        batch = state.size(0)
        # Path A
        flat = state.reshape(batch, -1)
        out_flat = self.flat_path(flat)
        # Path B
        img = state.unsqueeze(1)
        out_conv = self.conv_path(img).reshape(batch, -1)
        
        combined = torch.cat([out_flat, out_conv, features], dim=1)
        return self.head(combined)
