"""
Text generation with intervention support.
"""

import torch
import time
import logging
from typing import Optional, Dict, List, Any

from config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class Generator:
    """Generate text with optional debiasing interventions."""

    def __init__(self, model, tokenizer):
        """
        Initialize generator.

        Args:
            model: Loaded language model
            tokenizer: Loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = ModelConfig()

    def generate(
        self,
        input_text: str,
        intervention: Optional[Any] = None,
        generation_config: Optional[Dict] = None,
        **intervention_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text with optional intervention.

        Args:
            input_text: Input prompt
            intervention: Optional debiasing arm to apply
            generation_config: Optional custom generation config
            **intervention_kwargs: Additional args for intervention

        Returns:
            Dict with 'text', 'tokens', 'time_seconds', 'metadata'
        """
        start_time = time.time()

        # Use default generation config if not provided
        if generation_config is None:
            generation_config = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "do_sample": self.config.do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

        try:
            # Apply intervention if provided
            if intervention is not None:
                output_text = intervention.generate(
                    self.model, self.tokenizer, input_text, generation_config, **intervention_kwargs
                )
            else:
                # Standard generation
                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(**inputs, **generation_config)
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove input from output if present
                if output_text.startswith(input_text):
                    output_text = output_text[len(input_text) :].strip()

            generation_time = time.time() - start_time

            return {
                "text": output_text,
                "tokens": len(self.tokenizer.encode(output_text)),
                "time_seconds": generation_time,
                "metadata": {
                    "intervention": intervention.name if intervention else "none",
                    "input_length": len(input_text),
                    "output_length": len(output_text),
                },
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "text": "",
                "tokens": 0,
                "time_seconds": time.time() - start_time,
                "metadata": {"error": str(e)},
            }

    def generate_batch(
        self, inputs: List[str], intervention: Optional[Any] = None, generation_config: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate text for batch of inputs (sequential processing for memory safety).

        Args:
            inputs: List of input prompts
            intervention: Optional debiasing arm
            generation_config: Optional generation config

        Returns:
            List of generation results
        """
        results = []
        for input_text in inputs:
            result = self.generate(input_text, intervention, generation_config)
            results.append(result)
        return results
