"""
Load and manage LLM with 24GB VRAM constraint.
CRITICAL: Models must be loaded sequentially with explicit cleanup.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import logging
from typing import Tuple, Optional

from config.model_config import ModelConfig, clear_gpu_memory, get_vram_usage_mb, log_memory_stats

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton model loader with memory management.

    CRITICAL MEMORY RULES:
        1. NEVER load multiple models simultaneously
        2. ALWAYS call unload() before loading a different model
        3. ALWAYS call clear_memory() between loads
    """

    _instance = None
    _model = None
    _tokenizer = None
    _model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.config = ModelConfig()

    def load(self, model_name: Optional[str] = None) -> Tuple:
        """
        Load model with 4-bit quantization.

        Args:
            model_name: Model identifier (defaults to config)

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or self.config.model_name

        # Check if model already loaded
        if self._model is not None:
            if self._model_name == model_name:
                logger.info(f"Model {model_name} already loaded")
                return self._model, self._tokenizer
            else:
                logger.warning(f"Different model requested. Unloading {self._model_name}")
                self.unload()

        logger.info(f"Loading model: {model_name}")
        logger.info(f"VRAM before load: {get_vram_usage_mb():.2f} MB")

        # Clear memory before loading
        self.clear_memory()

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

        try:
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.config.device_map,
                max_memory=self.config.max_memory,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            self._model_name = model_name

            # Set pad token if not set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            vram_after = get_vram_usage_mb()
            logger.info(f"VRAM after load: {vram_after:.2f} MB")
            logger.info(f"Model loaded successfully: {model_name}")

            return self._model, self._tokenizer

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.clear_memory()
            raise

    def unload(self):
        """Unload current model and free memory."""
        if self._model is not None:
            logger.info(f"Unloading model: {self._model_name}")
            vram_before = get_vram_usage_mb()

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._model_name = None

            self.clear_memory()

            vram_after = get_vram_usage_mb()
            logger.info(f"VRAM before unload: {vram_before:.2f} MB")
            logger.info(f"VRAM after unload: {vram_after:.2f} MB")
            logger.info(f"Memory freed: {vram_before - vram_after:.2f} MB")

    def clear_memory(self):
        """Aggressive GPU memory cleanup."""
        clear_gpu_memory()

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def get_current_model_name(self) -> Optional[str]:
        """Get currently loaded model name."""
        return self._model_name

    @property
    def current_vram_mb(self) -> float:
        """Get current VRAM usage in MB."""
        return get_vram_usage_mb()

    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics."""
        stats = log_memory_stats()
        stats["is_loaded"] = self.is_loaded()
        stats["current_model"] = self.get_current_model_name()
        return stats
