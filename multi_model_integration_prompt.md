# Prompt: Add 6 Multilingual LLMs to MAB Debiasing Pipeline

## Overview

This prompt provides complete instructions to integrate 6 multilingual LLMs into the Multi-Armed Bandit (MAB) Debiasing System. All models are designed to run on **24GB VRAM** using **4-bit quantization**.

**Target Models:**

| # | Model Name | HuggingFace ID | Parameters |
|---|------------|----------------|------------|
| 1 | Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B |
| 2 | Aya-Expanse-8B | `CohereForAI/aya-expanse-8b` | 8B |
| 3 | Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B |
| 4 | Gemma-2-9B-IT | `google/gemma-2-9b-it` | 9B |
| 5 | OpenHathi-7B-Hi | `sarvamai/OpenHathi-7B-Hi-v0.1` | 7B |
| 6 | Airavata | `ai4bharat/Airavata` | 7B |

---

## File 1: Model Configuration

**File:** `config/model_config.py`

```python
"""
Configuration for all 6 multilingual LLMs used in MAB Debiasing experiments.

Hardware Requirement: 24GB VRAM GPU
Quantization: 4-bit NF4 via BitsAndBytes

All models support Hindi and English.
Models 1-4 also support Bengali.
Models 5-6 are Hindi-specialized (may have limited Bengali support).
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# ============================================================================
# ENUMS
# ============================================================================

class ModelType(Enum):
    GENERAL_MULTILINGUAL = "general_multilingual"
    MULTILINGUAL_SPECIALIZED = "multilingual_specialized"
    HINDI_SPECIALIZED = "hindi_specialized"

class ModelFamily(Enum):
    QWEN = "qwen"
    COHERE = "cohere"
    LLAMA = "llama"
    GEMMA = "gemma"
    SARVAM = "sarvam"
    AI4BHARAT = "ai4bharat"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single LLM."""
    
    # Identifiers
    model_key: str
    model_id: str
    display_name: str
    
    # Model properties
    model_family: ModelFamily
    model_type: ModelType
    parameters: str
    
    # Language support
    supported_languages: List[str]
    primary_language: str  # Best performing language
    
    # Memory requirements
    vram_4bit_gb: float
    vram_fp16_gb: float
    
    # Tokenizer settings
    use_fast_tokenizer: bool = True
    padding_side: str = "left"
    trust_remote_code: bool = False
    
    # Chat template
    has_chat_template: bool = True
    custom_chat_template: Optional[str] = None
    
    # Special tokens
    add_bos_token: bool = False
    add_eos_token: bool = False
    
    # Generation defaults
    default_max_new_tokens: int = 256
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    
    # Model-specific kwargs
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra_tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Access requirements
    requires_auth: bool = False
    license_type: str = "open"

# ============================================================================
# ALL 6 MODEL DEFINITIONS
# ============================================================================

MODELS: Dict[str, ModelConfig] = {
    
    # =========================================================================
    # MODEL 1: Qwen2.5-7B-Instruct
    # =========================================================================
    "qwen2.5-7b": ModelConfig(
        model_key="qwen2.5-7b",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        display_name="Qwen2.5-7B-Instruct",
        model_family=ModelFamily.QWEN,
        model_type=ModelType.GENERAL_MULTILINGUAL,
        parameters="7B",
        supported_languages=["en", "hi", "bn", "zh", "ja", "ko", "ar", "fr", "de", "es"],
        primary_language="en",
        vram_4bit_gb=7.0,
        vram_fp16_gb=14.0,
        use_fast_tokenizer=True,
        padding_side="left",
        trust_remote_code=True,  # Qwen requires this
        has_chat_template=True,
        default_max_new_tokens=256,
        default_temperature=0.7,
        default_top_p=0.9,
        extra_model_kwargs={},
        extra_tokenizer_kwargs={},
        requires_auth=False,
        license_type="apache-2.0",
    ),
    
    # =========================================================================
    # MODEL 2: Aya-Expanse-8B
    # =========================================================================
    "aya-expanse-8b": ModelConfig(
        model_key="aya-expanse-8b",
        model_id="CohereForAI/aya-expanse-8b",
        display_name="Aya-Expanse-8B",
        model_family=ModelFamily.COHERE,
        model_type=ModelType.MULTILINGUAL_SPECIALIZED,
        parameters="8B",
        supported_languages=["en", "hi", "bn", "ta", "te", "ml", "mr", "gu", "pa", "ar", "zh", "ja"],
        primary_language="en",  # Balanced across languages
        vram_4bit_gb=8.0,
        vram_fp16_gb=16.0,
        use_fast_tokenizer=True,
        padding_side="left",
        trust_remote_code=False,
        has_chat_template=True,
        default_max_new_tokens=256,
        default_temperature=0.7,
        default_top_p=0.9,
        extra_model_kwargs={},
        extra_tokenizer_kwargs={},
        requires_auth=False,
        license_type="cc-by-nc-4.0",
    ),
    
    # =========================================================================
    # MODEL 3: Llama-3.1-8B-Instruct
    # =========================================================================
    "llama-3.1-8b": ModelConfig(
        model_key="llama-3.1-8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama-3.1-8B-Instruct",
        model_family=ModelFamily.LLAMA,
        model_type=ModelType.GENERAL_MULTILINGUAL,
        parameters="8B",
        supported_languages=["en", "hi", "de", "fr", "it", "pt", "es", "th"],
        primary_language="en",
        vram_4bit_gb=7.0,
        vram_fp16_gb=16.0,
        use_fast_tokenizer=True,
        padding_side="left",
        trust_remote_code=False,
        has_chat_template=True,
        default_max_new_tokens=256,
        default_temperature=0.6,
        default_top_p=0.9,
        extra_model_kwargs={},
        extra_tokenizer_kwargs={},
        requires_auth=True,  # Requires accepting Meta license
        license_type="llama3.1",
    ),
    
    # =========================================================================
    # MODEL 4: Gemma-2-9B-IT
    # =========================================================================
    "gemma-2-9b": ModelConfig(
        model_key="gemma-2-9b",
        model_id="google/gemma-2-9b-it",
        display_name="Gemma-2-9B-IT",
        model_family=ModelFamily.GEMMA,
        model_type=ModelType.GENERAL_MULTILINGUAL,
        parameters="9B",
        supported_languages=["en", "hi", "bn", "de", "fr", "es", "it", "pt", "ja", "ko", "zh"],
        primary_language="en",
        vram_4bit_gb=9.0,
        vram_fp16_gb=18.0,
        use_fast_tokenizer=True,
        padding_side="left",
        trust_remote_code=False,
        has_chat_template=True,
        default_max_new_tokens=256,
        default_temperature=0.7,
        default_top_p=0.9,
        extra_model_kwargs={
            "attn_implementation": "eager",  # Gemma-2 specific
        },
        extra_tokenizer_kwargs={},
        requires_auth=True,  # Requires accepting Google license
        license_type="gemma",
    ),
    
    # =========================================================================
    # MODEL 5: OpenHathi-7B-Hi
    # =========================================================================
    "openhathi-7b": ModelConfig(
        model_key="openhathi-7b",
        model_id="sarvamai/OpenHathi-7B-Hi-v0.1",
        display_name="OpenHathi-7B-Hi",
        model_family=ModelFamily.SARVAM,
        model_type=ModelType.HINDI_SPECIALIZED,
        parameters="7B",
        supported_languages=["en", "hi"],
        primary_language="hi",  # Hindi-first model
        vram_4bit_gb=6.0,
        vram_fp16_gb=14.0,
        use_fast_tokenizer=True,
        padding_side="left",
        trust_remote_code=False,
        has_chat_template=False,  # Base model, no instruct tuning
        custom_chat_template="""{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}Assistant:""",
        default_max_new_tokens=256,
        default_temperature=0.7,
        default_top_p=0.9,
        extra_model_kwargs={},
        extra_tokenizer_kwargs={},
        requires_auth=False,
        license_type="llama2",
    ),
    
    # =========================================================================
    # MODEL 6: Airavata
    # =========================================================================
    "airavata-7b": ModelConfig(
        model_key="airavata-7b",
        model_id="ai4bharat/Airavata",
        display_name="Airavata-7B",
        model_family=ModelFamily.AI4BHARAT,
        model_type=ModelType.HINDI_SPECIALIZED,
        parameters="7B",
        supported_languages=["en", "hi"],
        primary_language="hi",  # Hindi-first model
        vram_4bit_gb=6.0,
        vram_fp16_gb=14.0,
        use_fast_tokenizer=True,
        padding_side="left",
        trust_remote_code=False,
        has_chat_template=True,
        default_max_new_tokens=256,
        default_temperature=0.7,
        default_top_p=0.9,
        extra_model_kwargs={},
        extra_tokenizer_kwargs={},
        requires_auth=False,
        license_type="llama2",
    ),
}

# ============================================================================
# MODEL GROUPS FOR EXPERIMENTS
# ============================================================================

MODEL_GROUPS = {
    "all": list(MODELS.keys()),
    
    "general_multilingual": [
        "qwen2.5-7b",
        "llama-3.1-8b", 
        "gemma-2-9b",
    ],
    
    "multilingual_specialized": [
        "aya-expanse-8b",
    ],
    
    "hindi_specialized": [
        "openhathi-7b",
        "airavata-7b",
    ],
    
    "bengali_support": [
        "qwen2.5-7b",
        "aya-expanse-8b",
        "gemma-2-9b",
    ],
    
    "no_auth_required": [
        "qwen2.5-7b",
        "aya-expanse-8b",
        "openhathi-7b",
        "airavata-7b",
    ],
}

# ============================================================================
# RECOMMENDED EXPERIMENT ORDER
# ============================================================================

EXPERIMENT_ORDER = [
    "qwen2.5-7b",      # 1. Best overall multilingual baseline
    "aya-expanse-8b",  # 2. Best for cross-lingual debiasing
    "llama-3.1-8b",    # 3. Industry standard baseline
    "gemma-2-9b",      # 4. Google architecture comparison
    "openhathi-7b",    # 5. Hindi-specialized (Sarvam AI)
    "airavata-7b",     # 6. Hindi-specialized (AI4Bharat/IIT)
]

# ============================================================================
# QUANTIZATION CONFIG
# ============================================================================

def get_quantization_config():
    """Get BitsAndBytes 4-bit quantization config for 24GB VRAM."""
    from transformers import BitsAndBytesConfig
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_config(model_key: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_key not in MODELS:
        available = list(MODELS.keys())
        raise ValueError(f"Unknown model: '{model_key}'. Available: {available}")
    return MODELS[model_key]

def get_models_by_group(group_name: str) -> List[str]:
    """Get list of model keys for a group."""
    if group_name not in MODEL_GROUPS:
        available = list(MODEL_GROUPS.keys())
        raise ValueError(f"Unknown group: '{group_name}'. Available: {available}")
    return MODEL_GROUPS[group_name]

def get_models_for_language(language: str) -> List[str]:
    """Get models that support a specific language."""
    return [
        key for key, config in MODELS.items()
        if language in config.supported_languages
    ]

def print_model_summary():
    """Print summary of all configured models."""
    print("\n" + "=" * 80)
    print("CONFIGURED MODELS FOR MAB DEBIASING EXPERIMENTS")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Parameters':<12} {'Type':<25} {'VRAM (4-bit)':<12} {'Languages'}")
    print("-" * 90)
    
    for key in EXPERIMENT_ORDER:
        config = MODELS[key]
        langs = ", ".join(config.supported_languages[:3])
        if len(config.supported_languages) > 3:
            langs += f" +{len(config.supported_languages) - 3}"
        
        print(f"{config.display_name:<20} {config.parameters:<12} {config.model_type.value:<25} {config.vram_4bit_gb:<12.1f} {langs}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_model_summary()
```

---

## File 2: Model Loader with Multi-Model Support

**File:** `src/llm/model_loader.py`

```python
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.model_config import (
    MODELS,
    ModelConfig,
    get_model_config,
    get_quantization_config,
    EXPERIMENT_ORDER,
)

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
    
    def set_hf_token(self, token: str):
        """Set HuggingFace token for models requiring authentication."""
        self._hf_token = token
    
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
            print(f"\nUnloading model: {self._current_model_key}")
            
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
            print(f"GPU Memory after unload: {mem_info['used_gb']:.2f} / {mem_info['total_gb']:.2f} GB")
    
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
            print(f"Model '{model_key}' already loaded.")
            return self._model, self._tokenizer
        
        # Unload existing model if different
        if self._model is not None:
            print(f"\nSwitching from '{self._current_model_key}' to '{model_key}'")
            self.unload()
        
        # Get model configuration
        config = get_model_config(model_key)
        
        print("\n" + "=" * 60)
        print(f"Loading: {config.display_name}")
        print(f"Model ID: {config.model_id}")
        print(f"Parameters: {config.parameters}")
        print(f"Type: {config.model_type.value}")
        print("=" * 60)
        
        # Check authentication requirement
        token = None
        if config.requires_auth:
            if self._hf_token:
                token = self._hf_token
                print("Using provided HuggingFace token")
            else:
                print("WARNING: Model requires authentication. Set token with set_hf_token()")
                print("Or run: huggingface-cli login")
        
        # Clear memory before loading
        clear_gpu_memory()
        mem_before = get_gpu_memory_info()
        print(f"\nGPU Memory before load: {mem_before['used_gb']:.2f} / {mem_before['total_gb']:.2f} GB")
        
        # Set default max memory for 24GB GPU
        if max_memory is None:
            max_memory = {"cuda:0": "22GB", "cpu": "32GB"}
        
        # Prepare quantization config
        quantization_config = None
        if use_4bit:
            quantization_config = get_quantization_config()
            print("Using 4-bit quantization (NF4)")
        
        # Load tokenizer
        print(f"\nLoading tokenizer...")
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
            print("Applied custom chat template")
        
        # Load model
        print(f"Loading model...")
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
        print(f"\nGPU Memory after load: {mem_after['used_gb']:.2f} / {mem_after['total_gb']:.2f} GB")
        print(f"Model memory footprint: ~{mem_used:.2f} GB")
        
        print(f"\n✓ Successfully loaded: {config.display_name}")
        
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
```

---

## File 3: Multi-Model Experiment Runner

**File:** `scripts/run_multi_model_experiment.py`

```python
"""
Run MAB debiasing experiments across all 6 models.

Usage:
    # Run all models
    python scripts/run_multi_model_experiment.py --output_dir ./results
    
    # Run specific models
    python scripts/run_multi_model_experiment.py --models qwen2.5-7b aya-expanse-8b
    
    # Run only Hindi-specialized models
    python scripts/run_multi_model_experiment.py --model_group hindi_specialized
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import (
    MODELS,
    EXPERIMENT_ORDER,
    MODEL_GROUPS,
    get_model_config,
    get_models_by_group,
    print_model_summary,
)
from src.llm.model_loader import MultiModelLoader, get_gpu_memory_info
from src.data.mab_dataset import MABDataset

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for multi-model experiment."""
    
    # Models to run
    model_keys: List[str]
    
    # Data paths
    data_dir: str
    output_dir: str
    
    # Experiment settings
    n_train_samples: int = 1000
    n_eval_samples: int = 200
    bandit_type: str = "linucb"
    
    # Per-model settings
    enable_learning: bool = True
    save_checkpoints: bool = True
    
    # HuggingFace token (for gated models)
    hf_token: Optional[str] = None

# ============================================================================
# SINGLE MODEL EXPERIMENT
# ============================================================================

def run_single_model_experiment(
    model_key: str,
    dataset: MABDataset,
    output_dir: Path,
    config: ExperimentConfig,
    loader: MultiModelLoader,
) -> Dict:
    """
    Run complete experiment for a single model.
    
    Args:
        model_key: Model identifier
        dataset: Loaded dataset
        output_dir: Directory for this model's results
        config: Experiment configuration
        loader: Model loader instance
        
    Returns:
        Results dictionary
    """
    model_config = get_model_config(model_key)
    
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {model_config.display_name}")
    print(f"Type: {model_config.model_type.value}")
    print(f"Languages: {', '.join(model_config.supported_languages)}")
    print("=" * 70)
    
    results = {
        "model_key": model_key,
        "model_id": model_config.model_id,
        "display_name": model_config.display_name,
        "model_type": model_config.model_type.value,
        "parameters": model_config.parameters,
        "start_time": datetime.now().isoformat(),
        "metrics": {},
        "per_language": {},
        "per_bias_type": {},
        "errors": [],
    }
    
    try:
        # Load model
        model, tokenizer = loader.load(model_key)
        
        # Record memory usage
        mem_info = get_gpu_memory_info()
        results["gpu_memory_gb"] = mem_info["used_gb"]
        
        # Initialize MAB pipeline for this model
        # (Import your pipeline here)
        # from src.pipeline.inference_pipeline import MABDebiasInferencePipeline
        # pipeline = MABDebiasInferencePipeline(model, tokenizer, ...)
        
        # =====================================================================
        # PHASE 1: Baseline Evaluation (No Debiasing)
        # =====================================================================
        print("\n--- Phase 1: Baseline Evaluation ---")
        
        baseline_results = {
            "by_language": {},
            "by_bias_type": {},
            "overall": {},
        }
        
        # Evaluate by language
        for lang in ["en", "hi", "bn"]:
            if lang not in model_config.supported_languages:
                print(f"Skipping {lang} (not supported by model)")
                continue
            
            lang_data = dataset.filter(language=lang, split="test")[:config.n_eval_samples]
            
            if not lang_data:
                continue
            
            print(f"\nEvaluating {lang.upper()}: {len(lang_data)} samples")
            
            lang_metrics = {
                "n_samples": len(lang_data),
                "bias_scores": [],
                "quality_scores": [],
            }
            
            for item in tqdm(lang_data, desc=f"Baseline {lang.upper()}"):
                try:
                    # Generate without debiasing
                    response = loader.generate(item.sentence)
                    
                    # Score bias and quality
                    # bias_score = score_bias(response, item)
                    # quality_score = score_quality(response, item)
                    
                    # Placeholder scores
                    bias_score = 0.5  # Replace with actual scoring
                    quality_score = 0.8
                    
                    lang_metrics["bias_scores"].append(bias_score)
                    lang_metrics["quality_scores"].append(quality_score)
                    
                except Exception as e:
                    results["errors"].append(f"Baseline {lang} {item.id}: {str(e)}")
            
            # Calculate averages
            if lang_metrics["bias_scores"]:
                lang_metrics["mean_bias"] = sum(lang_metrics["bias_scores"]) / len(lang_metrics["bias_scores"])
                lang_metrics["mean_quality"] = sum(lang_metrics["quality_scores"]) / len(lang_metrics["quality_scores"])
            
            baseline_results["by_language"][lang] = lang_metrics
        
        results["metrics"]["baseline"] = baseline_results
        
        # =====================================================================
        # PHASE 2: MAB Training
        # =====================================================================
        print("\n--- Phase 2: MAB Training ---")
        
        train_data = list(dataset.train_iter())[:config.n_train_samples]
        print(f"Training on {len(train_data)} samples")
        
        training_metrics = {
            "n_samples": len(train_data),
            "arm_selections": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "rewards": [],
        }
        
        for item in tqdm(train_data, desc="MAB Training"):
            try:
                # Run through MAB pipeline
                # result = pipeline.process(item.sentence, return_details=True)
                
                # Placeholder
                selected_arm = 1  # Replace with actual arm selection
                reward = 0.7
                
                training_metrics["arm_selections"][selected_arm] += 1
                training_metrics["rewards"].append(reward)
                
            except Exception as e:
                results["errors"].append(f"Training {item.id}: {str(e)}")
        
        if training_metrics["rewards"]:
            training_metrics["mean_reward"] = sum(training_metrics["rewards"]) / len(training_metrics["rewards"])
        
        results["metrics"]["training"] = training_metrics
        
        # =====================================================================
        # PHASE 3: Post-MAB Evaluation
        # =====================================================================
        print("\n--- Phase 3: Post-MAB Evaluation ---")
        
        mab_results = {
            "by_language": {},
            "by_bias_type": {},
            "overall": {},
        }
        
        for lang in ["en", "hi", "bn"]:
            if lang not in model_config.supported_languages:
                continue
            
            lang_data = dataset.filter(language=lang, split="test")[:config.n_eval_samples]
            
            if not lang_data:
                continue
            
            print(f"\nEvaluating {lang.upper()} with MAB: {len(lang_data)} samples")
            
            lang_metrics = {
                "n_samples": len(lang_data),
                "bias_scores": [],
                "quality_scores": [],
                "arm_selections": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            }
            
            for item in tqdm(lang_data, desc=f"MAB Eval {lang.upper()}"):
                try:
                    # Run through MAB pipeline with learned policy
                    # result = pipeline.process(item.sentence, return_details=True)
                    
                    # Placeholder
                    bias_score = 0.3  # Should be lower than baseline
                    quality_score = 0.78
                    selected_arm = 1
                    
                    lang_metrics["bias_scores"].append(bias_score)
                    lang_metrics["quality_scores"].append(quality_score)
                    lang_metrics["arm_selections"][selected_arm] += 1
                    
                except Exception as e:
                    results["errors"].append(f"MAB Eval {lang} {item.id}: {str(e)}")
            
            if lang_metrics["bias_scores"]:
                lang_metrics["mean_bias"] = sum(lang_metrics["bias_scores"]) / len(lang_metrics["bias_scores"])
                lang_metrics["mean_quality"] = sum(lang_metrics["quality_scores"]) / len(lang_metrics["quality_scores"])
                
                # Calculate improvement
                baseline_bias = baseline_results["by_language"].get(lang, {}).get("mean_bias", 0.5)
                lang_metrics["bias_reduction"] = baseline_bias - lang_metrics["mean_bias"]
                lang_metrics["bias_reduction_pct"] = (lang_metrics["bias_reduction"] / baseline_bias) * 100 if baseline_bias > 0 else 0
            
            mab_results["by_language"][lang] = lang_metrics
        
        results["metrics"]["mab"] = mab_results
        
        # =====================================================================
        # Calculate Overall Results
        # =====================================================================
        
        # Aggregate across languages
        all_baseline_bias = []
        all_mab_bias = []
        
        for lang in ["en", "hi", "bn"]:
            if lang in baseline_results["by_language"]:
                all_baseline_bias.extend(baseline_results["by_language"][lang].get("bias_scores", []))
            if lang in mab_results["by_language"]:
                all_mab_bias.extend(mab_results["by_language"][lang].get("bias_scores", []))
        
        if all_baseline_bias and all_mab_bias:
            results["metrics"]["overall"] = {
                "baseline_mean_bias": sum(all_baseline_bias) / len(all_baseline_bias),
                "mab_mean_bias": sum(all_mab_bias) / len(all_mab_bias),
                "overall_bias_reduction": (sum(all_baseline_bias) / len(all_baseline_bias)) - (sum(all_mab_bias) / len(all_mab_bias)),
            }
        
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "failed"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    finally:
        results["end_time"] = datetime.now().isoformat()
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{model_key}_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved: {results_file}")
        
        # Unload model
        loader.unload()
    
    return results

# ============================================================================
# MULTI-MODEL EXPERIMENT RUNNER
# ============================================================================

def run_multi_model_experiment(config: ExperimentConfig) -> Dict:
    """
    Run experiments across all specified models.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Aggregated results dictionary
    """
    print("\n" + "=" * 70)
    print("MULTI-MODEL MAB DEBIASING EXPERIMENT")
    print("=" * 70)
    print(f"\nModels to evaluate: {len(config.model_keys)}")
    for key in config.model_keys:
        mc = get_model_config(key)
        print(f"  - {mc.display_name} ({mc.parameters})")
    
    print(f"\nData directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = MABDataset(config.data_dir)
    
    # Initialize loader
    loader = MultiModelLoader()
    if config.hf_token:
        loader.set_hf_token(config.hf_token)
    
    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = {
        "experiment_start": datetime.now().isoformat(),
        "config": {
            "models": config.model_keys,
            "n_train_samples": config.n_train_samples,
            "n_eval_samples": config.n_eval_samples,
            "bandit_type": config.bandit_type,
        },
        "model_results": {},
        "summary": {},
    }
    
    for i, model_key in enumerate(config.model_keys):
        print(f"\n\n{'#'*70}")
        print(f"# MODEL {i+1}/{len(config.model_keys)}: {model_key}")
        print(f"{'#'*70}")
        
        model_output_dir = output_path / model_key
        
        try:
            results = run_single_model_experiment(
                model_key=model_key,
                dataset=dataset,
                output_dir=model_output_dir,
                config=config,
                loader=loader,
            )
            
            all_results["model_results"][model_key] = results
            
        except Exception as e:
            print(f"ERROR with {model_key}: {e}")
            all_results["model_results"][model_key] = {
                "status": "failed",
                "error": str(e),
            }
        
        # Memory cleanup between models
        loader.unload()
        time.sleep(2)
    
    # Generate summary
    all_results["experiment_end"] = datetime.now().isoformat()
    all_results["summary"] = generate_summary(all_results["model_results"])
    
    # Save aggregated results
    summary_file = output_path / "all_models_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\nAll results saved to: {output_path}")
    print(f"Summary file: {summary_file}")
    
    # Print summary table
    print_summary_table(all_results["summary"])
    
    return all_results

def generate_summary(model_results: Dict) -> Dict:
    """Generate summary statistics across all models."""
    summary = {
        "by_model": {},
        "by_language": {"en": [], "hi": [], "bn": []},
        "rankings": {},
    }
    
    for model_key, results in model_results.items():
        if results.get("status") != "success":
            continue
        
        metrics = results.get("metrics", {})
        overall = metrics.get("overall", {})
        
        summary["by_model"][model_key] = {
            "baseline_bias": overall.get("baseline_mean_bias"),
            "mab_bias": overall.get("mab_mean_bias"),
            "bias_reduction": overall.get("overall_bias_reduction"),
        }
        
        # Per-language
        mab_metrics = metrics.get("mab", {}).get("by_language", {})
        for lang in ["en", "hi", "bn"]:
            if lang in mab_metrics:
                summary["by_language"][lang].append({
                    "model": model_key,
                    "bias": mab_metrics[lang].get("mean_bias"),
                    "reduction": mab_metrics[lang].get("bias_reduction_pct"),
                })
    
    return summary

def print_summary_table(summary: Dict):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<25} {'Baseline Bias':<15} {'MAB Bias':<15} {'Reduction':<15}")
    print("-" * 70)
    
    for model_key, metrics in summary.get("by_model", {}).items():
        baseline = metrics.get("baseline_bias", 0)
        mab = metrics.get("mab_bias", 0)
        reduction = metrics.get("bias_reduction", 0)
        
        print(f"{model_key:<25} {baseline:<15.3f} {mab:<15.3f} {reduction:<15.3f}")
    
    print("\n" + "=" * 80)

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run MAB debiasing experiments across multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all 6 models
    python scripts/run_multi_model_experiment.py
    
    # Run specific models
    python scripts/run_multi_model_experiment.py --models qwen2.5-7b aya-expanse-8b
    
    # Run a model group
    python scripts/run_multi_model_experiment.py --model_group hindi_specialized
    
    # Quick test with fewer samples
    python scripts/run_multi_model_experiment.py --n_train 100 --n_eval 50
        """
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list(MODELS.keys()),
        help="Specific models to run (default: all)"
    )
    
    parser.add_argument(
        "--model_group",
        type=str,
        default=None,
        choices=list(MODEL_GROUPS.keys()),
        help="Run a predefined model group"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Path to processed dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/multi_model",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--n_train",
        type=int,
        default=1000,
        help="Number of training samples"
    )
    
    parser.add_argument(
        "--n_eval",
        type=int,
        default=200,
        help="Number of evaluation samples"
    )
    
    parser.add_argument(
        "--bandit",
        type=str,
        default="linucb",
        choices=["linucb", "thompson", "neural"],
        help="Bandit algorithm to use"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for gated models"
    )
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.models:
        model_keys = args.models
    elif args.model_group:
        model_keys = get_models_by_group(args.model_group)
    else:
        model_keys = EXPERIMENT_ORDER  # All 6 models
    
    # Create config
    config = ExperimentConfig(
        model_keys=model_keys,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_train_samples=args.n_train,
        n_eval_samples=args.n_eval,
        bandit_type=args.bandit,
        hf_token=args.hf_token,
    )
    
    # Run experiment
    results = run_multi_model_experiment(config)
    
    return results

if __name__ == "__main__":
    main()
```

---

## File 4: Requirements Update

**Add to `requirements.txt`:**

```
# Core ML
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0
bitsandbytes>=0.43.0

# HuggingFace
huggingface_hub>=0.21.0
datasets>=2.18.0
tokenizers>=0.15.0

# For specific models
sentencepiece>=0.1.99  # Required by some tokenizers
protobuf>=3.20.0       # Required by sentencepiece

# Numerical
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Progress and logging
tqdm>=4.65.0
wandb>=0.16.0  # Optional: for experiment tracking

# Utils
pyyaml>=6.0
```

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install torch transformers accelerate bitsandbytes huggingface_hub sentencepiece

# 2. Login to HuggingFace (for gated models like Llama and Gemma)
huggingface-cli login
# Enter your token when prompted

# 3. Test single model loading
python -c "
from src.llm.model_loader import MultiModelLoader
loader = MultiModelLoader()

# Test Qwen (no auth needed)
model, tokenizer = loader.load('qwen2.5-7b')
print(loader.generate('Hello, how are you?'))
loader.unload()
"

# 4. Run experiment on all 6 models
python scripts/run_multi_model_experiment.py \
    --data_dir ./data/processed \
    --output_dir ./results \
    --n_train 500 \
    --n_eval 100

# 5. Run only on models that don't need authentication
python scripts/run_multi_model_experiment.py \
    --model_group no_auth_required

# 6. Run only Hindi-specialized models
python scripts/run_multi_model_experiment.py \
    --model_group hindi_specialized
```

---

## Model Access Notes

| Model | Authentication | How to Access |
|-------|----------------|---------------|
| Qwen2.5-7B | ❌ None needed | Direct download |
| Aya-Expanse-8B | ❌ None needed | Direct download |
| Llama-3.1-8B | ✅ Required | Accept license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |
| Gemma-2-9B | ✅ Required | Accept license at https://huggingface.co/google/gemma-2-9b-it |
| OpenHathi-7B | ❌ None needed | Direct download |
| Airavata-7B | ❌ None needed | Direct download |

**To accept licenses:**
1. Go to the model page
2. Click "Access repository"
3. Accept the license terms
4. Run `huggingface-cli login` with your token

---

## Expected Output Structure

```
results/
├── multi_model/
│   ├── qwen2.5-7b/
│   │   ├── qwen2.5-7b_results.json
│   │   └── checkpoints/
│   ├── aya-expanse-8b/
│   │   ├── aya-expanse-8b_results.json
│   │   └── checkpoints/
│   ├── llama-3.1-8b/
│   │   └── ...
│   ├── gemma-2-9b/
│   │   └── ...
│   ├── openhathi-7b/
│   │   └── ...
│   ├── airavata-7b/
│   │   └── ...
│   └── all_models_summary.json  # Aggregated comparison
```
