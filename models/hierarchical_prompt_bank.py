
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCrossLingualPromptBank(nn.Module):
    """
    Hierarchical Cross-Lingual Prompt Bank (H-CLPB) Module.
    Generates hierarchical prompt embeddings conditioned on language family and signer identity.
    """
    def __init__(self, language_classes, signer_classes, d_model, family_prompt_count, signer_prompt_count, gamma=0.5):
        super().__init__()
        self.language_classes = language_classes
        self.signer_classes = signer_classes
        self.d_model = d_model
        self.family_prompt_count = family_prompt_count
        self.signer_prompt_count = signer_prompt_count
        self.gamma = gamma

        # Embedding projections
        self.language_family_embed = nn.Linear(len(language_classes), d_model)
        self.signer_embed = nn.Linear(len(signer_classes), d_model)

        # MLPs for prompt generation
        self.family_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, family_prompt_count * d_model)
        )
        self.signer_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, signer_prompt_count * d_model)
        )

    def forward(self, lang_onehot, signer_onehot):
        """
        Args:
            lang_onehot: One-hot vector of language class (B, |C|)
            signer_onehot: One-hot vector of signer identity (B, |S|)
        Returns:
            prompts: Concatenated prompt embeddings (B, M, d_model)
            l_prompt: Language family prompts (B, M_family, d_model)
            s_prompt: Signer-adaptive prompts (B, M_signer, d_model)
        """
        # Project language family and signer embeddings
        lang_embed = self.language_family_embed(lang_onehot)
        signer_embed = self.signer_embed(signer_onehot)

        # Family-level prompts
        l_prompt = self.family_mlp(lang_embed).view(-1, self.family_prompt_count, self.d_model)

        # Signer-specific prompts conditioned on both language and signer
        signer_input = torch.cat([lang_embed, signer_embed], dim=-1)
        s_prompt = self.signer_mlp(signer_input).view(-1, self.signer_prompt_count, self.d_model)

        # Concatenate family and signer prompts
        prompts = torch.cat([l_prompt, s_prompt], dim=1)

        return prompts, l_prompt, s_prompt

    def prompt_regularization_loss(self, l_prompt, s_prompt):
        """
        Computes the prompt regularization loss to suppress signer prompts magnitude.
        """
        l_loss = torch.norm(l_prompt, dim=-1).pow(2).mean()
        s_loss = torch.norm(s_prompt, dim=-1).pow(2).mean()
        reg_loss = l_loss + self.gamma * s_loss
        return reg_loss
