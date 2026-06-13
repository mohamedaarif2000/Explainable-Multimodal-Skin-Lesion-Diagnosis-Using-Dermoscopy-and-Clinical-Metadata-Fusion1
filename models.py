import torch
import torch.nn as nn
from torchvision import models

from config import DROPOUT, IMG_EMB_DIM, META_EMB_DIM, N_ATTN_HEADS, SHARED_DIM


class ImageEncoder(nn.Module):
    """EfficientNet-B3 backbone -> embedding vector."""

    def __init__(self, emb_dim=IMG_EMB_DIM, pretrained=True, dropout=DROPOUT):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 1536
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward_features(self, x):
        """Returns the pre-pooling feature map (for Grad-CAM) and the pooled vector."""
        feat_map = self.features(x)
        pooled = self.pool(feat_map).flatten(1)
        return feat_map, pooled

    def forward(self, x):
        _, pooled = self.forward_features(x)
        return self.proj(pooled)


class MetaEncoder(nn.Module):
    """Small MLP encoder for tabular clinical metadata."""

    def __init__(self, in_dim, emb_dim=META_EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ImageOnlyModel(nn.Module):
    """Image-only baseline (RQ1)."""

    def __init__(self, num_classes, emb_dim=IMG_EMB_DIM):
        super().__init__()
        self.encoder = ImageEncoder(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, img, meta=None):
        return self.classifier(self.encoder(img))


class AttentionFusionModel(nn.Module):
    """Proposed method: cross-attention multimodal fusion.

    Image and metadata embeddings are projected to a shared dimension and
    treated as a 2-token sequence; multi-head attention lets each modality
    attend to the other, producing a fused representation and interpretable
    attention weights.
    """

    def __init__(self, meta_dim, num_classes, img_emb=IMG_EMB_DIM, meta_emb=META_EMB_DIM,
                 shared_dim=SHARED_DIM, n_heads=N_ATTN_HEADS, dropout=DROPOUT):
        super().__init__()
        self.img_enc = ImageEncoder(img_emb, dropout=dropout)
        self.meta_enc = MetaEncoder(meta_dim, meta_emb, dropout=dropout)
        self.img_proj = nn.Linear(img_emb, shared_dim)
        self.meta_proj = nn.Linear(meta_emb, shared_dim)
        self.attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(shared_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * shared_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.shared_dim = shared_dim

    def project_image(self, img):
        """Image-branch projection only (used to cache embeddings for SHAP, Grad-CAM, etc.)."""
        return self.img_proj(self.img_enc(img))

    def forward_from_image_projection(self, i_proj, meta, return_attention=False):
        """Forward pass given an already-computed image projection, varying only metadata."""
        m = self.meta_proj(self.meta_enc(meta))
        B = meta.shape[0]
        i_rep = i_proj.expand(B, -1) if i_proj.shape[0] == 1 else i_proj
        tokens = torch.stack([i_rep, m], dim=1)
        attn_out, attn_weights = self.attn(tokens, tokens, tokens, need_weights=True, average_attn_weights=True)
        tokens = self.norm(tokens + attn_out)
        x = tokens.flatten(1)
        out = self.classifier(x)
        if return_attention:
            return out, attn_weights
        return out

    def forward(self, img, meta, return_attention=False):
        i = self.img_proj(self.img_enc(img))
        return self.forward_from_image_projection(i, meta, return_attention=return_attention)


class EarlyFusionModel(nn.Module):
    """Early fusion: concatenate image + metadata embeddings, then process jointly
    through several shared dense layers ("traditional feature concatenation")."""

    def __init__(self, meta_dim, num_classes, img_emb=IMG_EMB_DIM, meta_emb=META_EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.img_enc = ImageEncoder(img_emb, dropout=dropout)
        self.meta_enc = MetaEncoder(meta_dim, meta_emb, dropout=dropout)
        self.joint = nn.Sequential(
            nn.Linear(img_emb + meta_emb, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, img, meta):
        i = self.img_enc(img)
        m = self.meta_enc(meta)
        x = torch.cat([i, m], dim=1)
        return self.joint(x)


class LateFusionModel(nn.Module):
    """Late (decision-level) fusion: each modality independently produces class
    logits; the final prediction is a learned weighted combination of the two."""

    def __init__(self, meta_dim, num_classes, img_emb=IMG_EMB_DIM, meta_emb=META_EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.img_enc = ImageEncoder(img_emb, dropout=dropout)
        self.meta_enc = MetaEncoder(meta_dim, meta_emb, dropout=dropout)
        self.img_classifier = nn.Linear(img_emb, num_classes)
        self.meta_classifier = nn.Linear(meta_emb, num_classes)
        self.fusion_weight = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 initial weight

    def forward(self, img, meta):
        i_logits = self.img_classifier(self.img_enc(img))
        m_logits = self.meta_classifier(self.meta_enc(meta))
        w = torch.sigmoid(self.fusion_weight)
        return w * i_logits + (1 - w) * m_logits