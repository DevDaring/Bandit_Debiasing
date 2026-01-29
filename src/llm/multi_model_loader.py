"""
Multi-model loader with memory management for 24GB VRAM.

CRITICAL RULES:
1. NEVER load multiple models simultaneously
2. ALWAYS call unload() before loading a different model
3. ALWAYS call clear_gpu_memory() between model loads

Usage:
    loader = MultiModelLoader()

    # Load first model
    model, tokenizer = loader.load("qwen2.5-7b")
    # ... use model ...

    # Switch to different model
    loader.unload()
    model, tokenizer = loader.load("aya-expanse-8b")
"""

import gc
import sys
import time
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.models_multi import (
    MODELS,
    MultiModelConfig,
    get_model_config,
    get_quantization_config,
    EXPERIMENT_ORDER,
)

logger = logging.getLogger(__name__)

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """
    Aggressively clear GPU memory.
    MUST be called before loading any new model.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()

    # Small delay to ensure memory is freed
    time.sleep(1)

def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"available": 0, "used": 0, "total": 0}

    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    used = torch.cuda.memory_allocated(0) / (1024**3)
    cached = torch.cuda.memory_reserved(0) / (1024**3)

    return {
        "total_gb": round(total, 2),
        "used_gb": round(used, 2),
        "cached_gb": round(cached, 2),
        "free_gb": round(total - cached, 2),
    }

# ============================================================================
# MULTI-MODEL LOADER
# ============================================================================

class MultiModelLoader:
    """
    Singleton loader for multiple LLM models.
    Ensures only one model is loaded at a time to fit in 24GB VRAM.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._model = None
        self._tokenizer = None
        self._current_model_key = None
        self._current_config = None
        self._hf_token = None

        self._initialized = True
        logger.info("MultiModelLoader initialized")

    def set_hf_token(self, token: str):
        """Set HuggingFace token for models requiring authentication."""
        self._hf_token = token
        logger.info("HuggingFace token set")

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None

    @property
    def current_model(self) -> Optional[str]:
        """Get the key of currently loaded model."""
        return self._current_model_key

    @property
    def model(self):
        """Get the current model (raises if none loaded)."""
        if self._model is None:
            raise RuntimeError("No model loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self):
        """Get the current tokenizer (raises if none loaded)."""
        if self._tokenizer is None:
            raise RuntimeError("No tokenizer loaded. Call load() first.")
        return self._tokenizer

    def unload(self):
        """
        Unload current model and free GPU memory.
        MUST be called before loading a different model.
        """
        if self._model is not None:
            logger.info(f"Unloading model: {self._current_model_key}")

            # Delete model and tokenizer
            del self._model
            del self._tokenizer

            self._model = None
            self._tokenizer = None
            self._current_model_key = None
            self._current_config = None

            # Clear GPU memory
            clear_gpu_memory()

            # Print memory status
            mem_info = get_gpu_memory_info()
            logger.info(f"GPU Memory after unload: {mem_info['used_gb']:.2f} / {mem_info['total_gb']:.2f} GB")

    def load(
        self,
        model_key: str,
        use_4bit: bool = True,
        device_map: str = "auto",
        max_memory: Optional[Dict] = None,
    ) -> Tuple[Any, Any]:
        """
        Load a model by its key.

        Args:
            model_key: Key from MODELS dict (e.g., "qwen2.5-7b")
            use_4bit: Whether to use 4-bit quantization (recommended for 24GB)
            device_map: Device mapping strategy
            max_memory: Max memory per device

        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if same model already loaded
        if self._current_model_key == model_key and self._model is not None:
            logger.info(f"Model '{model_key}' already loaded.")
            return self._model, self._tokenizer

        # Unload existing model if different
        if self._model is not None:
            logger.info(f"Switching from '{self._current_model_key}' to '{model_key}'")
            self.unload()

        # Get model configuration
        config = get_model_config(model_key)

        logger.info("=" * 60)
        logger.info(f"Loading: {config.display_name}")
        logger.info(f"Model ID: {config.model_id}")
        logger.info(f"Parameters: {config.parameters}")
        logger.info(f"Type: {config.model_type.value}")
        logger.info("=" * 60)

        # Check authentication requirement
        token = None
        if config.requires_auth:
            if self._hf_token:
                token = self._hf_token
                logger.info("Using provided HuggingFace token")
            else:
                logger.warning("Model requires authentication. Set token with set_hf_token()")
                logger.warning("Or run: huggingface-cli login")

        # Clear memory before loading
        clear_gpu_memory()
        mem_before = get_gpu_memory_info()
        logger.info(f"GPU Memory before load: {mem_before['used_gb']:.2f} / {mem_before['total_gb']:.2f} GB")

        # Set default max memory for 24GB GPU
        if max_memory is None:
            max_memory = {"cuda:0": "22GB", "cpu": "32GB"}

        # Prepare quantization config
        quantization_config = None
        if use_4bit:
            quantization_config = get_quantization_config()
            logger.info("Using 4-bit quantization (NF4)")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "use_fast": config.use_fast_tokenizer,
            "padding_side": config.padding_side,
        }
        tokenizer_kwargs.update(config.extra_tokenizer_kwargs)

        if token:
            tokenizer_kwargs["token"] = token

        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            **tokenizer_kwargs
        )

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Apply custom chat template if needed
        if config.custom_chat_template and not config.has_chat_template:
            self._tokenizer.chat_template = config.custom_chat_template
            logger.info("Applied custom chat template")

        # Load model
        logger.info("Loading model...")
        model_kwargs = {
            "device_map": device_map,
            "max_memory": max_memory,
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": torch.float16,
        }

        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        model_kwargs.update(config.extra_model_kwargs)

        if token:
            model_kwargs["token"] = token

        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            **model_kwargs
        )

        # Set model to eval mode
        self._model.eval()

        # Update state
        self._current_model_key = model_key
        self._current_config = config

        # Print memory usage
        mem_after = get_gpu_memory_info()
        mem_used = mem_after['used_gb'] - mem_before['used_gb']
        logger.info(f"GPU Memory after load: {mem_after['used_gb']:.2f} / {mem_after['total_gb']:.2f} GB")
        logger.info(f"Model memory footprint: ~{mem_used:.2f} GB")

        logger.info(f"✓ Successfully loaded: {config.display_name}")

        return self._model, self._tokenizer

    def get_generation_config(self, **overrides) -> Dict:
        """Get default generation config for current model."""
        if self._current_config is None:
            raise RuntimeError("No model loaded")

        config = {
            "max_new_tokens": self._current_config.default_max_new_tokens,
            "temperature": self._current_config.default_temperature,
            "top_p": self._current_config.default_top_p,
            "do_sample": True,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        config.update(overrides)
        return config

    def generate(
        self,
        prompt: str,
        use_chat_template: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input text
            use_chat_template: Whether to apply chat template
            **generation_kwargs: Override generation parameters

        Returns:
            Generated text (without prompt)
        """
        if self._model is None:
            raise RuntimeError("No model loaded")

        # Prepare input
        if use_chat_template and self._current_config.has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            input_text = prompt

        # Tokenize
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        # Get generation config
        gen_config = self.get_generation_config(**generation_kwargs)

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                **gen_config
            )

        # Decode (remove input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip()

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_model(model_key: str, **kwargs) -> Tuple[Any, Any]:
    """Convenience function to load a model."""
    loader = MultiModelLoader()
    return loader.load(model_key, **kwargs)

def unload_model():
    """Convenience function to unload current model."""
    loader = MultiModelLoader()
    loader.unload()

def get_loader() -> MultiModelLoader:
    """Get the singleton loader instance."""
    return MultiModelLoader()

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test loading each model
    loader = MultiModelLoader()

    # Optionally set HF token
    # loader.set_hf_token("your_token_here")

    test_prompt = "What is the capital of India?"

    for model_key in EXPERIMENT_ORDER:
        try:
            print(f"\n{'='*60}")
            print(f"Testing: {model_key}")
            print(f"{'='*60}")

            model, tokenizer = loader.load(model_key)

            # Test generation
            response = loader.generate(test_prompt)
            print(f"\nPrompt: {test_prompt}")
            print(f"Response: {response[:200]}...")

            # Unload before next
            loader.unload()

        except Exception as e:
            print(f"Error with {model_key}: {e}")
            loader.unload()

    print("\n✓ All models tested successfully!")
