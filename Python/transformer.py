import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Convert patches to embeddings with a conv layer
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        assert H == W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        
        # Project patches and flatten: [B, E, H/P, W/P] -> [B, E, N]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with self-attention.
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class EncoderBlock(nn.Module):
    """
    Transformer encoder block with self-attention.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Calculate query, key, value for all heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    MLP block in Transformer.
    """
    def __init__(self, in_features, hidden_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SegmentationDecoder(nn.Module):
    """
    Decoder that converts transformer outputs to segmentation masks.
    """
    def __init__(self, embed_dim, img_size, patch_size, num_classes):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        # Project back to patch space
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * num_classes)
        
    def forward(self, x):
        # x: [B, num_patches, embed_dim]
        B, N, C = x.shape
        
        # Project to patch space [B, num_patches, P*P*num_classes]
        x = self.proj(x)
        
        # Reshape to [B, num_patches, patch_size, patch_size, num_classes]
        x = x.reshape(B, N, self.patch_size, self.patch_size, -1)
        
        # Rearrange patches to form the full image
        # First reshape to [B, grid_size, grid_size, patch_size, patch_size, num_classes]
        x = x.reshape(B, self.grid_size, self.grid_size, self.patch_size, self.patch_size, -1)
        
        # Permute and reshape to [B, num_classes, img_size, img_size]
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, -1, self.img_size, self.img_size)
        
        return x


class TransformerSegmentation(nn.Module):
    """
    Full Transformer model for semantic segmentation.
    """
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3, 
        num_classes=21,
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0, 
        dropout=0.1
    ):
        super().__init__()
        
        # Create patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Add position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        
        # Encoder
        self.encoder = TransformerEncoder(embed_dim, depth, num_heads, mlp_ratio, dropout)
        
        # Decoder for segmentation
        self.decoder = SegmentationDecoder(embed_dim, img_size, patch_size, num_classes)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        x = self.encoder(x)  # [B, num_patches, embed_dim]
        
        # Decoder for segmentation
        x = self.decoder(x)  # [B, num_classes, img_size, img_size]
        
        return x
