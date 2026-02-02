import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, List

# Check availability of ViT
try:
    from torchvision.models import ViT_B_16_Weights

    HAS_VIT = True
except ImportError:
    HAS_VIT = False


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) Encoder.
    Captures leaf texture, lesion shape, and vein damage.
    """

    def __init__(self, pretrained: bool = True):
        super(VisionEncoder, self).__init__()
        if HAS_VIT and pretrained:
            self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            # Fallback or initialization without weights
            self.model = models.vit_b_16()

        # We need the hidden states (sequences), not just the class token
        # Standard ViT output is (batch, 1000). We need to access the transformer blocks.
        # However, torchvision's ViT forward returns the class token output.
        # For a custom VLM, we ideally want the feature map.
        # We will wrap it to expose the encoder features.

        self.hidden_dim = 768  # ViT-Base hidden dimension

    def forward(self, x):
        # Allow gradients to flow for fine-tuning, or detach if frozen
        # x shape: (Batch, 3, 224, 224)

        # Extract features using the internal wrapper of torchvision ViT
        # Reshape to patches
        x = self.model._process_input(x)
        instance_token = self.model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((instance_token, x), dim=1)
        x = self.model.encoder(x)

        # x shape: (Batch, N_Patches+1, Hidden_Dim)
        return x


class LanguageDecoder(nn.Module):
    """
    Decoder-Only Transformer (Instruction-Tuned LLM compatible structure).
    Handles cure & prevention generation reasoning.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_dim: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
    ):
        super(LanguageDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, 512, hidden_dim)
        )  # Simplified pos encoding

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text_tokens, memory):
        # text_tokens: (Batch, Seq_Len)
        # memory: (Batch, Vision_Seq_Len, Hidden_Dim) -> From Vision Encoder

        x = self.embedding(text_tokens) + self.pos_encoder[:, : text_tokens.size(1), :]

        # Causal mask for the decoder (autoregressive)
        sz = text_tokens.size(1)
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1).to(x.device)

        # Transformer Decoder Forward
        # tgt = text embeddings
        # memory = vision embeddings
        out = self.decoder(tgt=x, memory=memory, tgt_mask=mask)
        return self.output_projection(out)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Transformer Layers.
    Allows language output to attend to visual features explicitly.
    """

    def __init__(self, hidden_dim: int = 768, nhead: int = 8):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value):
        # query: Language tokens
        # key_value: Vision embeddings
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        x = self.norm(query + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class AgriVLM(nn.Module):
    """
    AGRI-VLM-CARE+++: Production-grade Agricultural VLM.
    Integrates Vision Encoder, Language Decoder, and Multi-Task Heads.
    """

    def __init__(
        self, num_crops: int = 96, num_diseases: int = 200, num_weeds: int = 50
    ):
        super(AgriVLM, self).__init__()

        # 1. Vision Encoder (ViT)
        self.vision_encoder = VisionEncoder(pretrained=True)

        # 2. Language Decoder (Decoder-Only Transformer)
        # In a real scenario, this would be a pre-trained LLM like Phi-2 or Llama-2-7b
        # specific for instruction tuning. We verify architecture here.
        self.language_decoder = LanguageDecoder(hidden_dim=768)

        # 3. Fusion (Cross-Attention)
        self.fusion = CrossAttentionFusion(hidden_dim=768)

        # 4. Multi-Task Prediction Heads
        # Operating on the global visual token (CLS token from ViT)
        hidden_dim = 768

        # Head 1: Crop Classification
        self.crop_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_crops),
        )

        # Head 2: Disease Classification
        self.disease_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_diseases),
        )

        # Head 3: Weed Detection
        self.weed_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_weeds),  # Multi-label if multiple weeds possible
        )

        # Head 4: Severity Regression (0-5 scale)
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Scale to 0-1 then multiply by 5 later
        )

        # Head 5: Confidence Estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, images, text_tokens=None):
        """
        Forward pass for the VLM.
        """
        # 1. Encode Vision
        # vision_feats: (Batch, N_Patches+1=197, 768)
        vision_feats = self.vision_encoder(images)
        cls_token = vision_feats[:, 0, :]  # Global image representation

        # 2. Multi-Task Classification Heads Outputs
        crop_logits = self.crop_head(cls_token)
        disease_logits = self.disease_head(cls_token)
        weed_logits = self.weed_head(cls_token)
        severity = self.severity_head(cls_token) * 5.0  # Scale to 0-5
        confidence = self.confidence_head(cls_token)

        # 3. Language Generation (if text_tokens provided)
        # This simulates the "Instruction Tuning" phase where the model generates advice
        lm_logits = None
        if text_tokens is not None:
            # Cross Attention: Text attends to Image
            # fused_features = self.fusion(query=text_embeddings, key_value=vision_feats)
            # For simplicity in this architectural demo, passing vision_feats as memory
            lm_logits = self.language_decoder(text_tokens, memory=vision_feats)

        return {
            "crop_logits": crop_logits,
            "disease_logits": disease_logits,
            "weed_logits": weed_logits,
            "severity": severity,
            "confidence": confidence,
            "lm_logits": lm_logits,
        }


def create_agri_vlm():
    return AgriVLM()
