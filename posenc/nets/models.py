from typing import Sequence, Tuple

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.nets import UNETR
from monai.networks.nets import ViT as ViTMonai
from monai.utils import ensure_tuple_rep, is_sqrt
from torch import nn

from posenc.enums import PatchEmbeddingType, PosEncType
from posenc.nets.blocks import BoxConstraintLayer, Transformer
from posenc.nets.positional_encodings import PatchEmbeddingBlock, PositionalEmbedding


def pair(t, n=2):
    return t if isinstance(t, tuple) else [t for _ in range(n)]


class UNETRPOS(nn.Module):
    def __init__(
        self,
        posenc: PosEncType = PosEncType.SINCOS,
        scale: float = 1.0,
        temperature: int = 10000,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        out_channels: int = 4,
        crop_size: Tuple[int, int, int] = (96, 112, 96),
    ):
        super().__init__()
        self.unetr = UNETR(
            in_channels=4,
            out_channels=out_channels,
            img_size=crop_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
        )

        self.unetr.vit.patch_embedding = PatchEmbeddingBlock(
            pos_embed_type=posenc,
            img_size=crop_size,
            patch_size=16,
            hidden_size=768,
            num_heads=num_heads,
            in_channels=4,
            spatial_dims=3,
            scale=scale,
            temperature=temperature,
        )

    def forward(self, x):
        return self.unetr(x)


class ViT(nn.Module):
    def __init__(
        self,
        posenc: PosEncType,
        img_size: Sequence[int] | int,
        patch_embed_type: PatchEmbeddingType = PatchEmbeddingType.CONV,
        num_classes: int = 2,
        scale: int = 1,
        temperature: int = 10000,
        patch_size: Sequence[int] | int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.posenc = posenc

        self.vit = ViTMonai(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            proj_type="conv",
            pos_embed_type="learnable",
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=2,
            post_activation="none",
            qkv_bias=False,
            save_attn=False,
        )

        self.vit.patch_embedding = PatchEmbeddingBlock(
            pos_embed_type=posenc,
            patch_embed_type=patch_embed_type,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            in_channels=1,
            spatial_dims=2,
            scale=scale,
            temperature=temperature,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)[0]


class ViTLucid(nn.Module):
    def __init__(
        self,
        posenc: PosEncType,
        img_size: Sequence[int] | int = 224,
        patch_embed_type: PatchEmbeddingType = PatchEmbeddingType.CONV,
        num_classes: int = 2,
        scale: int = 1,
        temperature: int = 10000,
        patch_size: Sequence[int] | int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.posenc = posenc

        spatial_dims = 2
        channels = 1

        image_height, image_width = pair(img_size, spatial_dims)
        patch_height, patch_width = pair(patch_size, spatial_dims)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.pos_embedding = PositionalEmbedding(
            posenc,
            img_size,
            patch_size,
            hidden_size,
            spatial_dims,
            n_tokens=1,
            scale=scale,
            temperature=temperature,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.dropout = nn.Dropout(dropout_rate)

        self.transformer = Transformer(
            hidden_size, num_layers, num_heads, 64, mlp_dim, dropout_rate
        )

        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b = x.shape[0]

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embedding(x)

        x = self.dropout(x)

        x = self.transformer(x)

        return self.cls_head(x[:, 0])


class ViTDetection(nn.Module):
    """This model is only for 2D images with a single channel. The targets should be 8 classes and 4 bbox coordinates."""

    def __init__(
        self,
        posenc,
        image_size=256,
        patch_size=16,
        dim=512,
        depth=4,
        heads=8,
        mlp_dim=256,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.1,
        scale=1.0,
        temperature=10_000,
    ):

        # This model is for 2D only!
        spatial_dims = 2
        num_classes = 8
        bbox_coordinates = 4
        channels = 1
        n_tokens = 2  # One for classes and one for bboxes

        super().__init__()
        image_height, image_width = pair(image_size, spatial_dims)
        patch_height, patch_width = pair(patch_size, spatial_dims)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = PositionalEmbedding(
            posenc,
            image_size,
            patch_size,
            dim,
            spatial_dims,
            n_tokens=n_tokens,
            scale=scale,
            temperature=temperature,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.reg_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.cls_head = nn.Linear(dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(dim, bbox_coordinates), nn.Sigmoid(), BoxConstraintLayer()
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b = x.shape[0]

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        reg_tokens = repeat(self.reg_token, "1 1 d -> b 1 d", b=b)

        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        x = self.pos_embedding(x)

        x = self.dropout(x)

        x = self.transformer(x)

        cls_out = self.cls_head(x[:, 0])
        reg_out = self.reg_head(x[:, 1])

        return cls_out, reg_out


class VideoViT(nn.Module):
    """This VideoViT model is ViT that handles a sequence of 2D images."""

    def __init__(
        self,
        posenc: PosEncType,
        n_frames: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        scale: float = 1.0,
        temperature: int = 10000,
        cls_head: bool = False,
    ) -> None:

        super().__init__()

        self.cls_head = cls_head

        if not is_sqrt(patch_size):
            raise ValueError(f"patch_size should be square number, got {patch_size}.")

        # Ensure tuples
        self.patch_size = ensure_tuple_rep(patch_size, 2)
        self.img_size = ensure_tuple_rep(img_size, 2)

        # Define spatial grid size. MUST BE SQUARE
        self.spatial_grid_size = img_size // patch_size

        # Check patch size validity
        self.spatial_dims = 2
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(
                    f"patch_size={patch_size} should be divisible by img_size={img_size}."
                )

        # Patch embedding block
        self.patch_embedding_block = PatchEmbeddingBlock(
            pos_embed_type=posenc,
            in_channels=1,
            img_size=[img_size, img_size, n_frames],
            patch_size=[patch_size, patch_size, 1],
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=3,  # 3d for temporal positional embedding!
            patch_embed_type=PatchEmbeddingType.VIDEO,
            scale=scale,
            temperature=temperature,
        )

        # Build Transformer
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        if self.cls_head:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.classification_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as [batch_size, frames, channels, sp_size, sp_size].
        """
        x = self.patch_embedding_block(x)

        if self.cls_head:
            # Add cls_token to input
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Run transformer
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        if self.cls_head:
            # Run classification head
            x = self.classification_head(x[:, 0])

        return x, hidden_states_out


class Vid2Vid(nn.Module):
    def __init__(
        self,
        posenc=PosEncType.LEARNABLE,
        mlp_dim=192,
        num_heads=3,
        num_layers=2,
        hidden_size=768,
        patch_size=16,
        n_frames=16,
        img_size=112,
        dim_head=64,
        dropout_rate=0.0,
        scale=1.0,
        temperature=10000,
    ):
        super().__init__()

        # Some constants
        self.num_pixels_per_patch = patch_size**2
        self.grid_size = img_size // patch_size
        self.num_patches = n_frames * (img_size // patch_size) ** 2

        # Generate patches of the video
        self.generate_patches = PatchEmbeddingBlock(
            in_channels=1,
            spatial_dims=3,
            img_size=[n_frames, 112, 112],
            patch_size=[1, patch_size, patch_size],
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed_type=posenc,
            patch_embed_type=PatchEmbeddingType.VIDEO,
            dropout_rate=0.0,
            scale=scale,
            temperature=temperature,
        )

        # Transformer blocks
        self.transformer = Transformer(
            hidden_size,
            num_layers,
            num_heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout_rate,
        )

        # Regenerate video from patch tokens
        self.pixel_predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.num_pixels_per_patch),
            Rearrange(
                "b (f grid_h grid_w) (patch_h patch_w) -> b f () (grid_h patch_h) (grid_w patch_w)",
                f=n_frames,
                patch_h=patch_size,
                patch_w=patch_size,
                grid_w=self.grid_size,
                grid_h=self.grid_size,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        tokens = self.generate_patches(x)
        tokens = self.transformer(tokens)
        return self.pixel_predictor(tokens)


# TODO On module level at initialization
#     self.vit.apply(self._init_weights)

# def _init_weights(self, m):
#     """
#     Truncated normal initialization for weights and constant initialization for biases.
#     This is recommended for ViT models as shown by https://arxiv.org/abs/1803.01719.
#     """
#     if isinstance(m, nn.Linear):
#         trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
#         if isinstance(m, nn.Linear) and m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.constant_(m.bias, 0)
#         nn.init.constant_(m.weight, 1.0)
