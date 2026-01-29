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
class MultiModelConfig:
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

MODELS: Dict[str, MultiModelConfig] = {

    # =========================================================================
    # MODEL 1: Qwen2.5-7B-Instruct
    # =========================================================================
    "qwen2.5-7b": MultiModelConfig(
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
    "aya-expanse-8b": MultiModelConfig(
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
    "llama-3.1-8b": MultiModelConfig(
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
    "gemma-2-9b": MultiModelConfig(
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
    "openhathi-7b": MultiModelConfig(
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
    "airavata-7b": MultiModelConfig(
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

def get_model_config(model_key: str) -> MultiModelConfig:
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

    print(f"\n{'Model':<25} {'Parameters':<12} {'Type':<25} {'VRAM (4-bit)':<12} {'Languages'}")
    print("-" * 100)

    for key in EXPERIMENT_ORDER:
        config = MODELS[key]
        langs = ", ".join(config.supported_languages[:3])
        if len(config.supported_languages) > 3:
            langs += f" +{len(config.supported_languages) - 3}"

        print(f"{config.display_name:<25} {config.parameters:<12} {config.model_type.value:<25} {config.vram_4bit_gb:<12.1f} {langs}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_model_summary()
