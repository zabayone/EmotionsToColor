import torch
import torch.nn as nn
from config import EMBED_DIM, HIDDEN_DIM, N_COLORS, COLOR_DIM


class Text2PaletteModel(nn.Module):
    """
    Maps a CLIP text embedding to a 5-color palette in Oklab space.

    Architecture:
        - Shared encoder: two-layer MLP with LayerNorm and Dropout
        - Five independent color heads, one per output color
    Output shape: (B, N_COLORS, COLOR_DIM)  →  (B, 5, 3)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIM),
            nn.Dropout(0.1),
        )
        self.color_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_DIM, 64),
                nn.GELU(),
                nn.Linear(64, COLOR_DIM),
            )
            for _ in range(N_COLORS)
        ])

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        h = self.encoder(text_emb)
        colors = []
        for head in self.color_heads:
            raw = head(h)
            L  = torch.sigmoid(raw[:, 0:1])       # L  ∈ [0, 1]
            ab = torch.tanh(raw[:, 1:]) * 0.5     # a,b ∈ [-0.5, 0.5]
            colors.append(torch.cat([L, ab], dim=-1))
        return torch.stack(colors, dim=1)          # (B, 5, 3)
