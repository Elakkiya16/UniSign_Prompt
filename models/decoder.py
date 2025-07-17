import torch
import torch.nn as nn
import torch.nn.functional as F

class GlossTextDecoder(nn.Module):
    """
    GlossTextDecoder with dual-branch structure:
    - Gloss branch for sign gloss prediction
    - Text branch for natural language translation
    """
    def __init__(self, feature_dim, gloss_vocab_size, text_vocab_size, hidden_dim=512, num_layers=2, dropout=0.1):
        super(GlossTextDecoder, self).__init__()

        # Shared input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # Gloss Decoder
        self.gloss_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout),
            num_layers=num_layers
        )
        self.gloss_out = nn.Linear(hidden_dim, gloss_vocab_size)

        # Text Decoder
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dropout=dropout),
            num_layers=num_layers
        )
        self.text_out = nn.Linear(hidden_dim, text_vocab_size)

    def forward(self, encoded_feats, gloss_targets=None, text_targets=None):
        """
        Args:
            encoded_feats: Tensor [B, T, D]
            gloss_targets: Optional Tensor [B, T'] for gloss decoder (teacher forcing)
            text_targets: Optional Tensor [B, T'] for text decoder (teacher forcing)
        """
        memory = self.input_proj(encoded_feats).permute(1, 0, 2)  # [T, B, H]

        # Gloss Decoder
        if gloss_targets is not None:
            gloss_tgt_embed = F.one_hot(gloss_targets, num_classes=self.gloss_out.out_features).float()
            gloss_tgt_embed = gloss_tgt_embed.permute(1, 0, 2)
        else:
            gloss_tgt_embed = torch.zeros((1, memory.size(1), memory.size(2)), device=memory.device)

        gloss_output = self.gloss_decoder(gloss_tgt_embed, memory)
        gloss_logits = self.gloss_out(gloss_output.permute(1, 0, 2))

        # Text Decoder
        if text_targets is not None:
            text_tgt_embed = F.one_hot(text_targets, num_classes=self.text_out.out_features).float()
            text_tgt_embed = text_tgt_embed.permute(1, 0, 2)
        else:
            text_tgt_embed = torch.zeros((1, memory.size(1), memory.size(2)), device=memory.device)

        text_output = self.text_decoder(text_tgt_embed, memory)
        text_logits = self.text_out(text_output.permute(1, 0, 2))

        return gloss_logits, text_logits