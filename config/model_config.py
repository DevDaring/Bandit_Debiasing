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
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

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
