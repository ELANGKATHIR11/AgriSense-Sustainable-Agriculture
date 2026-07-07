# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import torch
import torch.nn as nn
import math

class PatchTST(nn.Module):
    def __init__(self, c_in=1, context_window=5, patch_len=2, stride=1, d_model=16, n_heads=2, d_ff=32, num_layers=1, target_dim=1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.c_in = c_in
        
        # Calculate number of patches
        self.num_patches = max(1, ((context_window - patch_len) // stride) + 1)
        
        # Linear projection of patches
        self.patch_proj = nn.Linear(patch_len, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head: projects flattened channels * patches * d_model to target
        self.head = nn.Linear(c_in * self.num_patches * d_model, target_dim)
        
    def forward(self, x):
        # x shape: [batch, c_in, seq_len]
        batch_size, c_in, seq_len = x.shape
        
        # Slide window across seq_len to extract patches
        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patches.append(x[:, :, i:i+self.patch_len])
            
        if not patches:
            # Fallback if seq_len is smaller than patch_len
            # Pad input
            padding = torch.zeros(batch_size, c_in, self.patch_len - seq_len, device=x.device)
            padded_x = torch.cat([x, padding], dim=-1)
            patches.append(padded_x)
            
        patches = torch.stack(patches, dim=2) # [batch, c_in, num_patches, patch_len]
        
        # Channel-independence: merge batch and channel dimensions
        num_patches = patches.shape[2]
        patches = patches.view(batch_size * c_in, num_patches, self.patch_len) # [batch * c_in, num_patches, patch_len]
        
        # Projection
        enc_out = self.patch_proj(patches) # [batch * c_in, num_patches, d_model]
        
        # Add Positional encoding (adjusting size if num_patches varies)
        if num_patches == self.num_patches:
            enc_out = enc_out + self.pos_encoder
        else:
            # Dynamically adjust position encodings
            pos = nn.functional.interpolate(self.pos_encoder.transpose(1, 2), size=num_patches, mode='linear', align_corners=False).transpose(1, 2)
            enc_out = enc_out + pos
            
        # Transformer
        enc_out = self.transformer(enc_out) # [batch * c_in, num_patches, d_model]
        
        # Restore channel dimension before flattening to maintain shape consistency
        enc_out = enc_out.view(batch_size, c_in, num_patches, self.d_model)
        
        # Flatten all channels and sequence patches
        enc_out = enc_out.view(batch_size, -1) # [batch, c_in * num_patches * d_model]
        
        # Project to target_dim
        out = self.head(enc_out) # [batch, target_dim]
        return out.squeeze(-1) # [batch]
