"""
Arm 5: Output probability adjustment.
Post-hoc modification of token probabilities to reduce biased outputs.
"""

import torch
from typing import Dict
import logging
from transformers import LogitsProcessor
from .base_arm import BaseArm

logger = logging.getLogger(__name__)


class BiasAdjustmentLogitsProcessor(LogitsProcessor):
    """Custom logits processor to adjust bias-associated tokens."""

    def __init__(self, tokenizer, adjustments: Dict[str, float]):
        """
        Initialize logits processor.

        Args:
            tokenizer: Tokenizer to get token IDs
            adjustments: Dict mapping token strings to log-prob adjustments
        """
        self.tokenizer = tokenizer
        self.adjustments = adjustments

        # Convert token strings to IDs
        self.token_adjustments = {}
        for token_str, adjustment in adjustments.items():
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if token_ids:
                # Use first token ID if multiple
                self.token_adjustments[token_ids[0]] = adjustment

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply adjustments to logits.

        Args:
            input_ids: Input token IDs
            scores: Logit scores for next token

        Returns:
            Modified logit scores
        """
        for token_id, adjustment in self.token_adjustments.items():
            if token_id < scores.shape[-1]:
                scores[:, token_id] += adjustment

        return scores


class OutputAdjustmentArm(BaseArm):
    """
    Adjust output token probabilities during generation to reduce bias.

    Method: Token probability dampening
        - Identify bias-associated tokens (e.g., gendered pronouns)
        - Reduce their probability during sampling
        - Boost neutral alternatives
    """

    def __init__(self, method: str = 'dampening', strength: float = 1.0):
        """
        Initialize output adjustment arm.

        Args:
            method: Adjustment method ('dampening' or 'contrastive')
            strength: Strength of adjustments
        """
        super().__init__(
            name="output_adjustment",
            requires_model_access=False,
            strength=strength
        )

        self.method = method

        # Define token adjustments (will be scaled by strength)
        self.base_adjustments = {
            # Dampen gendered pronouns
            'he': -2.0,
            'He': -2.0,
            'she': -2.0,
            'She': -2.0,
            'him': -2.0,
            'Him': -2.0,
            'her': -2.0,
            'Her': -2.0,
            'his': -2.0,
            'His': -2.0,
            'hers': -2.0,
            'Hers': -2.0,
            # Boost neutral alternatives
            'they': +1.0,
            'They': +1.0,
            'them': +1.0,
            'Them': +1.0,
            'their': +1.0,
            'Their': +1.0,
            'theirs': +1.0,
            'Theirs': +1.0,
        }

    def apply(self, model, tokenizer, input_text: str, **kwargs) -> Dict:
        """
        Prepare logits processor.

        Args:
            model: Language model (not used)
            tokenizer: Tokenizer
            input_text: Input text (not used)

        Returns:
            Dict with 'logits_processor'
        """
        # Scale adjustments by strength
        scaled_adjustments = {
            token: adj * self.strength
            for token, adj in self.base_adjustments.items()
        }

        processor = BiasAdjustmentLogitsProcessor(tokenizer, scaled_adjustments)

        return {'logits_processor': processor}

    def generate(self, model, tokenizer, input_text: str, generation_config: Dict, **kwargs) -> str:
        """
        Generate with output adjustment.

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input text
            generation_config: Generation parameters

        Returns:
            Generated text
        """
        # Create logits processor
        processor_data = self.apply(model, tokenizer, input_text)
        processor = processor_data['logits_processor']

        # Add processor to generation config
        generation_config = generation_config.copy()
        if 'logits_processor' in generation_config:
            generation_config['logits_processor'].append(processor)
        else:
            generation_config['logits_processor'] = [processor]

        # Generate
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, **generation_config)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input from output
        if generated_text.startswith(input_text):
            generated_text = generated_text[len(input_text):].strip()

        return generated_text
