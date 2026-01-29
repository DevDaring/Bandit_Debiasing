"""
Arms 1-3: Steering vector-based debiasing.
Apply pre-computed steering vectors to model activations during generation.
"""

import torch
from typing import Dict, List
import logging
from .base_arm import BaseArm

logger = logging.getLogger(__name__)


class SteeringVectorArm(BaseArm):
    """
    Apply steering vectors to hidden states during generation.

    Steering vector methodology:
        1. Pre-compute steering vector S = E[h_biased] - E[h_neutral]
        2. During generation, modify hidden states: h' = h - alpha * S
        3. Apply to multiple layers for stronger effect
    """

    def __init__(self, bias_type: str, vector_path: str, strength: float = 1.0,
                 start_layer: int = 12, end_layer: int = 24):
        """
        Initialize steering vector arm.

        Args:
            bias_type: Type of bias ('gender', 'race', 'religion')
            vector_path: Path to pre-computed steering vector
            strength: Scaling factor (alpha)
            start_layer: First layer to apply steering
            end_layer: Last layer to apply steering
        """
        super().__init__(
            name=f"{bias_type}_steering",
            requires_model_access=True,
            strength=strength
        )

        self.bias_type = bias_type
        self.vector_path = vector_path
        self.start_layer = start_layer
        self.end_layer = end_layer

        # Load steering vector
        self.steering_vector = self._load_vector()

        # Track hooks for cleanup
        self.active_hooks = []

    def _load_vector(self) -> torch.Tensor:
        """Load pre-computed steering vector."""
        try:
            vector_data = torch.load(self.vector_path, map_location='cpu')

            if isinstance(vector_data, dict):
                steering = vector_data['steering_vector']
            else:
                steering = vector_data

            logger.info(f"Loaded steering vector for {self.bias_type}: shape {steering.shape}")
            return steering

        except FileNotFoundError:
            logger.warning(f"Steering vector not found at {self.vector_path}. Using zeros.")
            # Return dummy vector if file doesn't exist
            return torch.zeros(32, 4096)  # Typical shape for 7B model

    def apply(self, model, tokenizer, input_text: str, **kwargs) -> Dict:
        """
        Register forward hooks for steering.

        Args:
            model: Language model
            tokenizer: Tokenizer (not used)
            input_text: Input text (not used)

        Returns:
            Dict with 'hooks' list
        """
        hooks = []

        def make_steering_hook(layer_idx):
            """Create steering hook for specific layer."""
            def hook(module, input, output):
                # Output is typically a tuple: (hidden_states, ...)
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Apply steering: h' = h - strength * steering
                if layer_idx < len(self.steering_vector):
                    steering = self.steering_vector[layer_idx].to(hidden_states.device)

                    # Steering shape: (hidden_dim,)
                    # Hidden states shape: (batch, seq_len, hidden_dim)
                    # Broadcast steering across batch and sequence
                    hidden_states = hidden_states - self.strength * steering

                # Return modified output
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states

            return hook

        # Register hooks on transformer layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
            for idx in range(self.start_layer, min(self.end_layer + 1, len(layers))):
                hook_handle = layers[idx].register_forward_hook(make_steering_hook(idx))
                hooks.append(hook_handle)
                self.active_hooks.append(hook_handle)

        logger.info(f"Registered {len(hooks)} steering hooks for {self.bias_type}")

        return {'hooks': hooks}

    def generate(self, model, tokenizer, input_text: str, generation_config: Dict, **kwargs) -> str:
        """
        Generate text with steering applied.

        Args:
            model: Language model
            tokenizer: Tokenizer
            input_text: Input text
            generation_config: Generation parameters

        Returns:
            Generated text
        """
        # Register hooks
        hook_data = self.apply(model, tokenizer, input_text)

        try:
            # Generate with steering active
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, **generation_config)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove input from output
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()

        finally:
            # CRITICAL: Remove hooks after generation
            self._remove_hooks()

        return generated_text

    def _remove_hooks(self):
        """Remove all active hooks to prevent memory leaks."""
        for hook in self.active_hooks:
            hook.remove()
        self.active_hooks = []
        logger.debug(f"Removed steering hooks for {self.bias_type}")


# Convenience classes for each bias type
class GenderSteeringArm(SteeringVectorArm):
    """Steering arm for gender bias."""

    def __init__(self, vector_path: str = "data/steering_vectors/gender_steering.pt",
                 strength: float = 1.5, **kwargs):
        super().__init__(bias_type='gender', vector_path=vector_path, strength=strength, **kwargs)


class RaceSteeringArm(SteeringVectorArm):
    """Steering arm for race/ethnicity bias."""

    def __init__(self, vector_path: str = "data/steering_vectors/race_steering.pt",
                 strength: float = 1.2, **kwargs):
        super().__init__(bias_type='race', vector_path=vector_path, strength=strength, **kwargs)


class ReligionSteeringArm(SteeringVectorArm):
    """Steering arm for religion bias."""

    def __init__(self, vector_path: str = "data/steering_vectors/religion_steering.pt",
                 strength: float = 1.0, **kwargs):
        super().__init__(bias_type='religion', vector_path=vector_path, strength=strength, **kwargs)
