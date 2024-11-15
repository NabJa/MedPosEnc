import torch
from einops import rearrange
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_layers: int = 0) -> None:
        super().__init__()

        layers = []
        if n_layers < 2:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        else:
            _out = in_features // 2
            layers.append(nn.Linear(in_features, _out))
            layers.append(nn.ReLU())
            for i in range(1, n_layers - 1):
                _in = in_features // (i * 2)
                _out = in_features // ((i + 1) * 2)
                layers.append(nn.Linear(_in, _out))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(_out, out_features))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BoxConstraintLayer(nn.Module):
    """Ensure that boxes are in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, y1, x2, y2 = torch.chunk(x, 4, dim=-1)
        x2 = torch.clamp(x2, min=x1 + 1e-3, max=torch.tensor(1).to(x))
        y2 = torch.clamp(y2, min=y1 + 1e-3, max=torch.tensor(1).to(x))
        return torch.cat([x1, y1, x2, y2], dim=-1)
