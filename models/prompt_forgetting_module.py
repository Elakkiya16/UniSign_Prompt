
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input_, lambda_):
        ctx.lambda_ = lambda_
        return input_.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class PromptForgettingModule(nn.Module):
    def __init__(self, prompt_dim, num_signers, beta=0.1, lambda_grl=1.0):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_signers = num_signers
        self.beta = beta

        self.grl = GradientReversalLayer(lambda_grl)
        self.signer_classifier = nn.Linear(prompt_dim, num_signers)

    def forward(self, signer_prompts, family_prompts, signer_labels):
        batch_size, M_signer, d = signer_prompts.size()
        assert d == self.prompt_dim, "Mismatch in prompt dimensionality"

        signer_repr = signer_prompts.mean(dim=1)

        signer_repr_adv = self.grl(signer_repr)
        signer_logits = self.signer_classifier(signer_repr_adv)
        forgetting_loss = F.cross_entropy(signer_logits, signer_labels)

        M_family = family_prompts.shape[1]
        dot_products = torch.einsum('bmd,bnd->bmn', signer_prompts, family_prompts)
        norm_signer = signer_prompts.norm(dim=-1, keepdim=True) + 1e-8
        norm_family = family_prompts.norm(dim=-1, keepdim=True) + 1e-8
        norms = torch.matmul(norm_signer, norm_family.transpose(1, 2))
        cosine_similarity = dot_products / norms
        decorrelation_loss = (cosine_similarity ** 2).mean()

        total_forgetting_loss = forgetting_loss + self.beta * decorrelation_loss

        return {
            "signer_logits": signer_logits,
            "forgetting_loss": forgetting_loss,
            "decorrelation_loss": decorrelation_loss,
            "total_forgetting_loss": total_forgetting_loss
        }
