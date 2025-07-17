
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiObjectiveForgettingLoss(nn.Module):
    def __init__(self, lambda_forget=0.5, lambda_align=0.3, lambda_route=0.1, lambda_prompt=0.1, beta=0.1, gamma=0.1):
        super(MultiObjectiveForgettingLoss, self).__init__()
        self.lambda_forget = lambda_forget
        self.lambda_align = lambda_align
        self.lambda_route = lambda_route
        self.lambda_prompt = lambda_prompt
        self.beta = beta
        self.gamma = gamma

    def forward(self, log_probs, targets, signer_logits, signer_targets, 
                family_prompts, signer_prompts, prompt_probs, prompt_bank, language_prompt_batches):
        # Translation Loss (Negative Log-Likelihood)
        loss_trans = F.nll_loss(log_probs, targets)

        # Signer Forgetting Loss (Cross Entropy with Gradient Reversal applied externally)
        loss_forget_ce = F.cross_entropy(signer_logits, signer_targets)

        # Decorrelation Loss between family and signer prompts
        norm_family_prompts = F.normalize(family_prompts, dim=-1)
        norm_signer_prompts = F.normalize(signer_prompts, dim=-1)
        loss_decor = (norm_family_prompts.unsqueeze(1) * norm_signer_prompts.unsqueeze(0)).pow(2).mean()
        
        loss_forget = loss_forget_ce + self.beta * loss_decor

        # Cross-Lingual Alignment Loss (between language prompt batches)
        loss_align = 0.0
        if len(language_prompt_batches) > 1:
            norm_prompts = [F.normalize(p, dim=-1) for p in language_prompt_batches]
            for i in range(len(norm_prompts)-1):
                loss_align += F.mse_loss(norm_prompts[i], norm_prompts[i+1])
            loss_align /= (len(norm_prompts) - 1)

        # Routing Sparsity Loss (entropy of prompt_probs)
        loss_route = - (prompt_probs * (prompt_probs + 1e-8).log()).sum(dim=-1).mean()

        # Prompt Regularization Loss
        loss_prompt = family_prompts.pow(2).mean() + self.gamma * signer_prompts.pow(2).mean()

        # Total Loss
        total_loss = (loss_trans + 
                      self.lambda_forget * loss_forget +
                      self.lambda_align * loss_align +
                      self.lambda_route * loss_route +
                      self.lambda_prompt * loss_prompt)

        return {
            'loss_trans': loss_trans,
            'loss_forget': loss_forget,
            'loss_align': loss_align,
            'loss_route': loss_route,
            'loss_prompt': loss_prompt,
            'total_loss': total_loss
        }
