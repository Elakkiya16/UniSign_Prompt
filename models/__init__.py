"""
models/__init__.py

This module initializes the UniSign-Prompt architecture components by exposing core classes
and modules including:
- UniSignPrompt (Main integrated architecture)
- PromptInjectedSignTransformer (PI-ST+)
- HierarchicalCrossLingualPromptBank (H-CLPB)
- TemporalAwarePromptInjection (TAP)
- PromptRoutingMechanism (PRM)
- PromptForgettingModule (PFM)
- GlossTextDecoder (Gloss → Text Decoder)
"""

from .unisign_prompt import UniSignPrompt
from .prompt_injected_sign_transformer import PromptInjectedSignTransformer
from .hierarchical_prompt_bank import HierarchicalCrossLingualPromptBank
from .tap_module import TemporalAwarePromptInjection
from .prompt_routing_mechanism import PromptRoutingMechanism
from .prompt_forgetting_module import PromptForgettingModule
from .decoder import GlossTextDecoder

__all__ = [
    "UniSignPrompt",
    "PromptInjectedSignTransformer",
    "HierarchicalCrossLingualPromptBank",
    "TemporalAwarePromptInjection",
    "PromptRoutingMechanism",
    "PromptForgettingModule",
    "GlossTextDecoder"
]