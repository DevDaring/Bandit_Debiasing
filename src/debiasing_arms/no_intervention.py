"""
Arm 0: No intervention baseline.
Standard generation without any debiasing.
"""

from typing import Dict
from .base_arm import BaseArm


class NoInterventionArm(BaseArm):
    """
    Baseline arm - standard LLM generation without modification.

    This arm exists to:
        1. Provide baseline comparison
        2. Be selected when input has low bias risk
        3. Preserve generation quality when debiasing not needed
    """

    def __init__(self):
        """Initialize no-intervention arm."""
        super().__init__(name="no_intervention", requires_model_access=False)

    def apply(self, model, tokenizer, input_text: str, **kwargs) -> Dict:
        """No modification needed."""
        return {}

    def generate(self, model, tokenizer, input_text: str, generation_config: Dict, **kwargs) -> str:
        """
        Standard generation without modification.

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input prompt
            generation_config: Generation parameters

        Returns:
            Generated text
        """
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **generation_config)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input from output if present
        if generated_text.startswith(input_text):
            generated_text = generated_text[len(input_text):].strip()

        return generated_text
