import torch
import torch.nn as nn

from .prompt_injected_sign_transformer import PromptInjectedSignTransformer
from .hierarchical_prompt_bank import HierarchicalCrossLingualPromptBank
from .tap_module import TemporalAwarePromptInjection
from .prompt_routing_mechanism import PromptRoutingMechanism
from .prompt_forgetting_module import PromptForgettingModule
from .decoder import GlossTextDecoder


class UniSignPrompt(nn.Module):
    def __init__(self, 
                 visual_dim=1024,
                 prompt_dim=512,
                 hidden_dim=512,
                 num_layers=6,
                 num_heads=8,
                 prompt_length=10,
                 num_signers=11,
                 gloss_vocab_size=500,
                 text_vocab_size=3000,
                 max_seq_length=64):
        super(UniSignPrompt, self).__init__()

        # Prompt-Injected Sign Transformer
        self.pi_st = PromptInjectedSignTransformer(
            visual_dim=visual_dim,
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            prompt_length=prompt_length
        )

        # Hierarchical Cross-Lingual Prompt Bank
        self.clpb = HierarchicalCrossLingualPromptBank(
            prompt_dim=prompt_dim,
            prompt_length=prompt_length
        )

        # Temporal-Aware Prompt Injection
        self.tap = TemporalAwarePromptInjection(
            prompt_dim=prompt_dim,
            visual_dim=visual_dim
        )

        # Prompt Routing Mechanism
        self.prm = PromptRoutingMechanism(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim
        )

        # Prompt Forgetting Module
        self.pfm = PromptForgettingModule(
            hidden_dim=hidden_dim,
            num_signers=num_signers
        )

        # Decoder (Gloss and Text Decoding)
        self.decoder = GlossTextDecoder(
            feature_dim=hidden_dim,
            gloss_vocab_size=gloss_vocab_size,
            text_vocab_size=text_vocab_size
        )

    def forward(self, visual_features, signer_onehot=None, language_onehot=None):
        """
        visual_features: Tensor [B, T, V]
        signer_onehot: Tensor [B, S] or None
        language_onehot: Tensor [B, L] or None
        """
        # Step 1: Fetch prompts from H-CLPB
        prompts, l_prompt, s_prompt = self.clpb(language_onehot, signer_onehot)

        # Step 2: Temporal-Aware Prompt Injection
        injected_features = self.tap(visual_features, prompts)

        # Step 3: Visual Encoding via Prompt-Injected Sign Transformer
        encoded_features = self.pi_st(injected_features, prompts)

        # Step 4: Prompt Routing (assume routing by signer + language mean embeddings)
        signer_emb = signer_onehot.float() if signer_onehot is not None else torch.zeros_like(s_prompt[:, 0, :])
        language_emb = language_onehot.float() if language_onehot is not None else torch.zeros_like(l_prompt[:, 0, :])

        routed_features, routing_scores = self.prm(signer_emb, language_emb, prompts)

        # Step 5: Signer Forgetting via PFM (if signer info provided)
        if signer_onehot is not None:
            pfm_out = self.pfm(s_prompt, l_prompt, signer_onehot.argmax(dim=1))
            forgetting_loss = pfm_out["total_forgetting_loss"]
        else:
            forgetting_loss = None

        # Step 6: Decoding (Gloss and Text)
        gloss_logits, text_logits = self.decoder(routed_features)

        output = {
            'gloss_logits': gloss_logits,
            'text_logits': text_logits,
            'routing_scores': routing_scores,
            'forgetting_loss': forgetting_loss
        }

        return output
