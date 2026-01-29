"""
Abstract base class for debiasing arms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseArm(ABC):
    """
    Abstract base class for debiasing intervention strategies.

    Attributes:
        name: Arm identifier string
        requires_model_access: Whether arm needs to modify model internals
        strength: Intervention strength (for tunable arms)
    """

    def __init__(self, name: str, requires_model_access: bool = False, strength: float = 1.0):
        """
        Initialize arm.

        Args:
            name: Arm name
            requires_model_access: Whether arm modifies model internals
            strength: Intervention strength
        """
        self.name = name
        self.requires_model_access = requires_model_access
        self.strength = strength

    @abstractmethod
    def apply(self, model, tokenizer, input_text: str, **kwargs) -> Dict:
        """
        Apply debiasing intervention.

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input text
            **kwargs: Additional arguments

        Returns:
            Dict with any modified state
        """
        pass

    @abstractmethod
    def generate(self, model, tokenizer, input_text: str, generation_config: Dict, **kwargs) -> str:
        """
        Generate text with intervention applied.

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input text
            generation_config: Generation parameters
            **kwargs: Additional arguments

        Returns:
            Generated text
        """
        pass
