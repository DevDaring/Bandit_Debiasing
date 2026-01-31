"""
Model configuration for 24GB VRAM constraint.
All models use 4-bit quantization via BitsAndBytes.
Models are NEVER loaded simultaneously - always sequential loading with explicit cleanup.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import torch
import gc


@dataclass
class ModelConfig:
    """Configuration for LLM loading with memory constraints."""

    # Primary model choice (Qwen for best Hindi/Bengali support)
    # Updated to 1.5B per My_Improvement_Prompts.md (24GB VRAM constraint)
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Quantization settings for 24GB VRAM
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Memory management
    device_map: str = "auto"
    max_memory: Dict = field(default_factory=lambda: {"cuda:0": "22GB", "cpu": "32GB"})
    offload_folder: str = "./offload"

    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    # Batch size (keep at 1 for memory safety)
    batch_size: int = 1

    # Timeout for generation (seconds)
    generation_timeout: int = 60


@dataclass
class EmbeddingModelConfig:
    """Smaller model for context encoding - loaded separately from main LLM."""

    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length: int = 256
    embedding_dim: int = 384


def clear_gpu_memory():
    """
    Aggressively clear GPU memory before loading new model.
    MUST be called before loading any new model.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def get_vram_usage_mb():
    """Get current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def log_memory_stats():
    """Log detailed memory statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024

        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
        }
    return {}


# ============================================================================
# MODEL REGISTRY FOR FAIR-CB EXPERIMENTS
# ============================================================================

MODEL_REGISTRY = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_family": "Qwen",
        "display_name": "Qwen 2.5 (1.5B)",
        "parameters": "1.5B",
        "hidden_size": 1536,
        "num_layers": 28,
        "recommended_sae_layer": 14,
        "quantization": "4bit",
        "vram_4bit_gb": 2.5,
        "supports_flash_attention": True,
        "supported_languages": ["en", "hi", "bn", "zh"],
        "trust_remote_code": True,
        "requires_auth": False,
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "model_family": "Llama",
        "display_name": "Llama 3.2 (1B)",
        "parameters": "1B",
        "hidden_size": 2048,
        "num_layers": 16,
        "recommended_sae_layer": 8,
        "quantization": "4bit",
        "vram_4bit_gb": 1.5,
        "supports_flash_attention": True,
        "supported_languages": ["en", "hi", "de", "fr", "es"],
        "trust_remote_code": False,
        "requires_auth": True,
    },
    "google/gemma-2-2b-it": {
        "model_name": "google/gemma-2-2b-it",
        "model_family": "Gemma",
        "display_name": "Gemma 2 (2B)",
        "parameters": "2B",
        "hidden_size": 2304,
        "num_layers": 26,
        "recommended_sae_layer": 13,
        "quantization": "4bit",
        "vram_4bit_gb": 3.0,
        "supports_flash_attention": True,
        "supported_languages": ["en", "hi", "bn", "de", "fr", "ja"],
        "trust_remote_code": False,
        "requires_auth": True,
        "extra_model_kwargs": {"attn_implementation": "eager"},
    },
}

# Model aliases for convenience
MODEL_ALIASES = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma": "google/gemma-2-2b-it",
    "gemma2b": "google/gemma-2-2b-it",
}


def get_model_config(model_name: str) -> Dict:
    """
    Get configuration for a specific model.

    Args:
        model_name: HuggingFace model identifier or alias

    Returns:
        Dict with model configuration

    Raises:
        ValueError: If model not found in registry
    """
    # Resolve alias if provided
    resolved_name = MODEL_ALIASES.get(model_name.lower(), model_name)

    if resolved_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{model_name}'. Available: {available}")

    return MODEL_REGISTRY[resolved_name].copy()


def get_all_models() -> list:
    """
    Get list of all registered model names.

    Returns:
        List of HuggingFace model identifiers
    """
    return list(MODEL_REGISTRY.keys())


def get_model_hidden_size(model_name: str) -> int:
    """
    Get hidden dimension for a model (needed for SAE training).

    Args:
        model_name: HuggingFace model identifier or alias

    Returns:
        Hidden size dimension
    """
    config = get_model_config(model_name)
    return config["hidden_size"]


def get_models_for_language(language: str) -> list:
    """
    Get models that support a specific language.

    Args:
        language: Language code (e.g., 'en', 'hi', 'bn')

    Returns:
        List of model names supporting the language
    """
    return [
        name for name, config in MODEL_REGISTRY.items()
        if language in config["supported_languages"]
    ]


def print_model_summary():
    """Print summary of all registered models."""
    print("\n" + "=" * 70)
    print("REGISTERED MODELS FOR FAIR-CB EXPERIMENTS")
    print("=" * 70)

    print(f"\n{'Model':<35} {'Params':<8} {'VRAM':<10} {'Languages'}")
    print("-" * 70)

    for name, config in MODEL_REGISTRY.items():
        langs = ", ".join(config["supported_languages"][:3])
        if len(config["supported_languages"]) > 3:
            langs += f" +{len(config['supported_languages']) - 3}"

        print(f"{config['display_name']:<35} {config['parameters']:<8} "
              f"{config['vram_4bit_gb']:<10.1f} {langs}")

    print("=" * 70 + "\n")
