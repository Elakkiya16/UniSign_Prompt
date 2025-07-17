import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptRoutingMechanism(nn.Module):
    """
    Prompt Routing Mechanism (PRM):
    Dynamically routes prompts based on combined signer and language embeddings.
    Uses Gumbel-Softmax sampling for sparse prompt selection.
    """
    def __init__(self, prompt_dim, hidden_dim, num_prompts=20, tau=1.0):
        """
        Args:
            prompt_dim (int): Dimension of each prompt vector.
            hidden_dim (int): Dimension of combined signer/language hidden representation.
            num_prompts (int): Number of prompt slots in the prompt bank.
            tau (float): Temperature for Gumbel-Softmax sampling.
        """
        super().__init__()
        self.num_prompts = num_prompts
        self.tau = tau

        self.routing_fc_in = nn.Linear(hidden_dim, prompt_dim)
        self.routing_fc_out = nn.Linear(prompt_dim, num_prompts)

    def forward(self, hidden_embedding, prompt_bank):
        """
        Args:
            hidden_embedding: Tensor of shape [B, hidden_dim] combining signer/language embeddings.
            prompt_bank: Tensor of shape [num_prompts, prompt_dim]
        
        Returns:
            active_prompts: Tensor [B, prompt_dim]
            routing_sparsity_loss: Scalar loss term for routing sparsity (entropy loss).
            routing_weights: Tensor [B, num_prompts]
        """
        batch_size = hidden_embedding.size(0)

        hidden = F.relu(self.routing_fc_in(hidden_embedding))  # [B, prompt_dim]
        routing_logits = self.routing_fc_out(hidden)  # [B, num_prompts]

        # Gumbel-Softmax Sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(routing_logits) + 1e-9) + 1e-9)
        logits_with_noise = (routing_logits + gumbel_noise) / self.tau
        routing_weights = F.softmax(logits_with_noise, dim=1)  # [B, num_prompts]

        # Weighted sum of prompts
        active_prompts = torch.matmul(routing_weights, prompt_bank)  # [B, prompt_dim]

        # Entropy loss (routing sparsity)
        routing_sparsity_loss = -torch.sum(routing_weights * torch.log(routing_weights + 1e-9), dim=1).mean()

        return active_prompts, routing_sparsity_loss, routing_weights