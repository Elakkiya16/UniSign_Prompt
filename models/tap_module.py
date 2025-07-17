
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAwarePromptInjection(nn.Module):
    """
    Temporal-Aware Prompt Injection (TAP) Module
    Segment-specific prompt conditioning with temporal smoothness regularization
    """

    def __init__(self, num_segments, prompt_dim, language_dim, temp_emb_dim, hidden_dim, prompt_length):
        """
        Args:
            num_segments (int): Number of temporal segments (S).
            prompt_dim (int): Dimension of each prompt token (d).
            language_dim (int): Dimension of language embedding (e_l).
            temp_emb_dim (int): Dimension of temporal embedding (e_temp).
            hidden_dim (int): Hidden dimension in MLP layers.
            prompt_length (int): Number of prompt tokens per segment (M_s).
        """
        super(TemporalAwarePromptInjection, self).__init__()
        self.num_segments = num_segments
        self.prompt_dim = prompt_dim
        self.prompt_length = prompt_length

        # Temporal positional embeddings (one per segment)
        self.temp_embeddings = nn.Embedding(num_segments, temp_emb_dim)

        # Segment-specific MLP to generate prompt vectors
        self.segment_mlp = nn.Sequential(
            nn.Linear(language_dim + temp_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_length * prompt_dim)
        )

    def forward(self, language_embedding):
        """
        Args:
            language_embedding: Tensor of shape (B, language_dim)
        
        Returns:
            List of segment-specific prompt tensors: [P^(1), ..., P^(S)],
            where each P^(s) is (B, M_s, d)
        """
        batch_size = language_embedding.size(0)
        prompts = []

        for s in range(self.num_segments):
            temp_idx = torch.tensor(s, device=language_embedding.device).expand(batch_size)
            temp_emb = self.temp_embeddings(temp_idx)  # (B, temp_emb_dim)

            combined_emb = torch.cat([language_embedding, temp_emb], dim=-1)  # (B, language_dim + temp_emb_dim)
            prompt = self.segment_mlp(combined_emb)  # (B, M_s * d)
            prompt = prompt.view(batch_size, self.prompt_length, self.prompt_dim)  # (B, M_s, d)
            prompts.append(prompt)

        return prompts

    def temporal_smoothness_loss(self, prompts):
        """
        Temporal smoothness regularization across segments
        Args:
            prompts: List of (B, M_s, d) prompt tensors
        Returns:
            Scalar smoothness loss
        """
        loss = 0.0
        for s in range(self.num_segments - 1):
            norm_p_s = F.normalize(prompts[s], dim=-1)
            norm_p_next = F.normalize(prompts[s + 1], dim=-1)
            loss += torch.norm(norm_p_s - norm_p_next, dim=-1).pow(2).mean()

        return loss / (self.num_segments - 1)
