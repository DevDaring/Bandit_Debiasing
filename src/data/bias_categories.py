"""
Bias category and language mappings for CSV output standardization.

CRITICAL: All output columns must use full forms, not abbreviations.
Examples:
- "English" not "en"
- "Gender Identity" not "gender"
- "Intersectional Bias Reduction (IBR)" not "ibr"
"""

from enum import Enum
from typing import Dict, List, Optional


# ============================================================================
# BIAS CATEGORY MAPPINGS
# ============================================================================

BIAS_CATEGORY_NAMES: Dict[str, str] = {
    # Multi-CrowS-Pairs categories
    "race-color": "Race and Color",
    "race": "Race and Ethnicity",
    "gender": "Gender Identity",
    "socioeconomic": "Socioeconomic Status",
    "nationality": "National Origin",
    "religion": "Religious Belief",
    "age": "Age Group",
    "sexual-orientation": "Sexual Orientation",
    "sexual_orientation": "Sexual Orientation",
    "physical-appearance": "Physical Appearance",
    "physical_appearance": "Physical Appearance",
    "disability": "Disability Status",
    # IndiBias categories
    "caste": "Caste System",
    "religious": "Religious Belief",
}

# India-specific categories for special analysis
INDIA_SPECIFIC_CATEGORIES = ["Caste System", "Religious Belief"]


# ============================================================================
# LANGUAGE MAPPINGS
# ============================================================================

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "english": "English",
    "hindi": "Hindi",
    "bengali": "Bengali",
}

SUPPORTED_LANGUAGES = ["English", "Hindi", "Bengali"]


# ============================================================================
# MODEL DISPLAY NAMES
# ============================================================================

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 2.5 (1.5B Parameters)",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 (1B Parameters)",
    "google/gemma-2-2b-it": "Gemma 2 (2B Parameters)",
    # Larger models from models_multi.py
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 (7B Parameters)",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 (8B Parameters)",
    "google/gemma-2-9b-it": "Gemma 2 (9B Parameters)",
    "CohereForAI/aya-expanse-8b": "Aya Expanse (8B Parameters)",
    "sarvamai/OpenHathi-7B-Hi-v0.1": "OpenHathi (7B Parameters)",
    "ai4bharat/Airavata": "Airavata (7B Parameters)",
}


# ============================================================================
# METRIC NAMES
# ============================================================================

METRIC_NAMES: Dict[str, str] = {
    "ibr": "Intersectional Bias Reduction (IBR)",
    "far": "Fairness-Aware Regret (FAR)",
    "bias_score": "Bias Score",
    "quality_score": "Output Quality Score",
    "regret": "Cumulative Regret",
    "violation": "Fairness Violation",
}


# ============================================================================
# DEBIASING ARM NAMES
# ============================================================================

ARM_NAMES: Dict[str, str] = {
    "arm_0": "No Intervention (Baseline)",
    "arm_1": "Gender Steering Vector",
    "arm_2": "Race Steering Vector",
    "arm_3": "Religion Steering Vector",
    "arm_4": "Prompt Prefix Debiasing",
    "arm_5": "Output Adjustment",
    "no_intervention": "No Intervention (Baseline)",
    "gender_steering": "Gender Steering Vector",
    "race_steering": "Race Steering Vector",
    "religion_steering": "Religion Steering Vector",
    "prompt_prefix": "Prompt Prefix Debiasing",
    "output_adjustment": "Output Adjustment",
}

ARM_INDEX_TO_NAME: Dict[int, str] = {
    0: "No Intervention (Baseline)",
    1: "Gender Steering Vector",
    2: "Race Steering Vector",
    3: "Religion Steering Vector",
    4: "Prompt Prefix Debiasing",
    5: "Output Adjustment",
}


# ============================================================================
# BANDIT ALGORITHM NAMES
# ============================================================================

BANDIT_NAMES: Dict[str, str] = {
    "linucb": "Linear Upper Confidence Bound (LinUCB)",
    "thompson": "Thompson Sampling",
    "neural": "Neural Contextual Bandit",
}


# ============================================================================
# STATISTICAL COLUMN NAMES
# ============================================================================

STATISTICAL_COLUMN_NAMES: Dict[str, str] = {
    "mean": "Mean Value",
    "std": "Standard Deviation",
    "ci_lower": "95% Confidence Interval (Lower)",
    "ci_upper": "95% Confidence Interval (Upper)",
    "p_value": "Statistical Significance (p-value)",
    "cohens_d": "Effect Size (Cohen's d)",
    "is_significant": "Is Statistically Significant (p < 0.05)",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_full_bias_category_name(short_name: str) -> str:
    """
    Convert short bias category name to full form.

    Args:
        short_name: Short form like 'gender', 'race-color'

    Returns:
        Full form like 'Gender Identity', 'Race and Color'
    """
    return BIAS_CATEGORY_NAMES.get(short_name.lower(), short_name)


def get_full_language_name(short_name: str) -> str:
    """
    Convert short language code to full name.

    Args:
        short_name: Short form like 'en', 'hi', 'bn'

    Returns:
        Full form like 'English', 'Hindi', 'Bengali'
    """
    return LANGUAGE_NAMES.get(short_name.lower(), short_name)


def get_full_model_name(model_id: str) -> str:
    """
    Convert HuggingFace model ID to display name.

    Args:
        model_id: HuggingFace identifier

    Returns:
        Display name for publications
    """
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def get_arm_name(arm_id: int | str) -> str:
    """
    Get full name for debiasing arm.

    Args:
        arm_id: Arm index (0-5) or string identifier

    Returns:
        Full arm name
    """
    if isinstance(arm_id, int):
        return ARM_INDEX_TO_NAME.get(arm_id, f"Unknown Arm {arm_id}")
    return ARM_NAMES.get(str(arm_id).lower(), str(arm_id))


def get_all_bias_categories() -> List[str]:
    """Get list of all bias category full names (deduplicated)."""
    return list(set(BIAS_CATEGORY_NAMES.values()))


def is_india_specific_category(category: str) -> bool:
    """Check if category is India-specific (Caste, Religious)."""
    full_name = get_full_bias_category_name(category)
    return full_name in INDIA_SPECIFIC_CATEGORIES


def normalize_column_name(name: str) -> str:
    """
    Normalize a column name to full form.

    Handles compound names like 'bias_score_en' -> 'Bias Score - English'
    """
    parts = name.replace("-", "_").split("_")
    normalized_parts = []

    for part in parts:
        # Check all mapping dictionaries
        if part.lower() in LANGUAGE_NAMES:
            normalized_parts.append(LANGUAGE_NAMES[part.lower()])
        elif part.lower() in BIAS_CATEGORY_NAMES:
            normalized_parts.append(BIAS_CATEGORY_NAMES[part.lower()])
        elif part.lower() in METRIC_NAMES:
            normalized_parts.append(METRIC_NAMES[part.lower()])
        elif part.lower() in BANDIT_NAMES:
            normalized_parts.append(BANDIT_NAMES[part.lower()])
        else:
            # Capitalize first letter of unrecognized parts
            normalized_parts.append(part.capitalize())

    return " - ".join(normalized_parts) if len(normalized_parts) > 1 else normalized_parts[0]
