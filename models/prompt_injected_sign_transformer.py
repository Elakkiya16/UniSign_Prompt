import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptInjectedSignTransformer(nn.Module):
    """
    Prompt-Injected Sign Transformer (PI-ST+):
    Transformer encoder with prompt injection and prompt-centric residual connections.
    Expects pre-extracted visual features as input.
    """
    def __init__(self, input_dim, prompt_dim, hidden_dim, num_layers=6, num_heads=8, prompt_length=10):
        """
        Args:
            input_dim (int): Dimensionality of visual input features (e.g., frame-level embeddings).
            prompt_dim (int): Dimension of prompt embeddings.
            hidden_dim (int): Hidden dimension of Transformer layers.
            num_layers (int): Number of Transformer encoder layers.
            num_heads (int): Number of attention heads.
            prompt_length (int): Number of prompt tokens to inject.
        """
        super().__init__()
        self.prompt_length = prompt_length

        # Input projection to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Prompt projection
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable scaling parameter for prompt residual
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, visual_features, prompts):
        """
        Args:
            visual_features: Tensor [B, T, input_dim]
            prompts: Tensor [B, M, prompt_dim]
        
        Returns:
            encoded_features: Tensor [B, T, hidden_dim]
        """
        B, T, _ = visual_features.shape
        M = prompts.shape[1]

        # Project inputs and prompts
        visual_proj = self.input_proj(visual_features)  # [B, T, hidden_dim]
        prompt_proj = self.prompt_proj(prompts)  # [B, M, hidden_dim]

        # Concatenate prompts + visual features
        x = torch.cat([prompt_proj, visual_proj], dim=1)  # [B, M+T, hidden_dim]

        # Prompt-centric residual connection
        prompt_mean = prompt_proj.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        x = x + self.alpha * prompt_mean

        # Transformer encoding
        x = self.transformer(x)  # [B, M+T, hidden_dim]

        # Remove prompt tokens from output
        encoded_features = x[:, M:, :]  # [B, T, hidden_dim]

        return encoded_features
