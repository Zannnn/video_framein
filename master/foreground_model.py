import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head Attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        
        # Feed Forward Network
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_output))
        return x

# class ForegroundModel(nn.Module):
#     def __init__(self, image_size=256, patch_size=16, embed_dim=128, num_heads=4, ff_dim=512, depth=4):
#         super(ForegroundModel, self).__init__()
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_patches = (image_size // patch_size) ** 2
#         self.embed_dim = embed_dim

#         # Patch embedding: flatten patches and embed them
#         self.patch_embedding = nn.Sequential(
#             nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
#             nn.Flatten(2),  # 将 H 和 W 合并到单一维度
#         )

#         # Positional encoding
#         self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

#         # Transformer layers
#         self.transformer_layers = nn.ModuleList([
#             TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(depth)
#         ])

#         # Output: project back to image space
#         self.output_projection = nn.Linear(embed_dim, patch_size * patch_size * 3)
#         self.unpatchify = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)

class ForegroundModel(nn.Module):
    def __init__(self, image_size=256, patch_size=16, embed_dim=128, num_heads=4, ff_dim=512, depth=4):
        super(ForegroundModel, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  # 每个图像的 patch 数量
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

        # Positional encoding: 动态计算大小，支持两个图像拼接
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 2 * self.num_patches, embed_dim)  # 支持拼接后的 patch 数量
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(depth)
        ])

        # Output projection
        self.output_projection = nn.Linear(embed_dim, patch_size * patch_size * 3)
        self.unpatchify = nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)


    # def forward(self, frame1, frame3):
    #     batch_size, _, h, w = frame1.size()

    #     # Step 1: Patchify input frames
    #     patches1 = self.patch_embedding(frame1).permute(0, 2, 1)  # (batch, num_patches, embed_dim)
    #     patches3 = self.patch_embedding(frame3).permute(0, 2, 1)

    #     # Step 2: Concatenate patches and add positional encoding
    #     patches = torch.cat([patches1, patches3], dim=1)  # (batch, 2*num_patches, embed_dim)
    #     patches += self.positional_encoding[:, :patches.size(1), :]

    #     # Step 3: Pass through Transformer layers
    #     for layer in self.transformer_layers:
    #         patches = layer(patches)

    #     # Step 4: Predict middle frame patches
    #     middle_patches = self.output_projection(patches)

    #     # Step 5: Unpatchify to reconstruct the image
    #     middle_frame = self.unpatchify(middle_patches.transpose(1, 2).view(batch_size, -1, h // self.patch_size, w // self.patch_size))
    #     return middle_frame
    def forward(self, frame1, frame3):
        batch_size, _, h, w = frame1.size()

        # Step 1: Patchify input frames
        patches1 = self.patch_embedding(frame1).permute(0, 2, 1)  # (batch, num_patches, embed_dim)
        patches3 = self.patch_embedding(frame3).permute(0, 2, 1)

        # Step 2: Concatenate patches and add positional encoding
        patches = torch.cat([patches1, patches3], dim=1)  # (batch, 2*num_patches, embed_dim)
        patches += self.positional_encoding[:, :patches.size(1), :]

        # Step 3: Pass through Transformer layers
        for layer in self.transformer_layers:
            patches = layer(patches)

        # Step 4: Predict middle frame patches
        middle_patches = self.output_projection(patches)  # (batch, 2*num_patches, patch_dim)
        middle_patches = middle_patches[:, :self.num_patches, :]  # 只保留中间帧的 patch (batch, num_patches, patch_dim)

        # Step 5: Unpatchify to reconstruct the image
        patch_height, patch_width = self.patch_size, self.patch_size
        middle_patches = middle_patches.transpose(1, 2)  # 转换为 (batch, patch_dim, num_patches)
        middle_patches = middle_patches.reshape(batch_size, 3 * patch_height * patch_width, -1)
        middle_frame = self.unpatchify(middle_patches)  # 使用 nn.Fold 重建中间帧

        return middle_frame

