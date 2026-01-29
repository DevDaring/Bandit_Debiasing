# Complete Implementation Prompt: Adaptive Multi-Armed Bandit Debiasing Strategy Selection for Multilingual LLMs

## Project Overview

Build a complete Python system that implements **Adaptive Multi-Armed Bandit (MAB) Debiasing Strategy Selection** for multilingual Large Language Models. This system dynamically selects the optimal debiasing intervention (from a portfolio of strategies) for each input based on context features, using contextual bandit algorithms that learn from fairness and quality feedback signals.

**Target Languages:** English, Hindi, Bengali
**Target Models:** Multilingual LLMs (2B-7B parameters) - Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, or Gemma-2-9B-IT
**Hardware Constraint:** Single GPU with 24GB VRAM - all models must be loaded sequentially, never in parallel
**Framework:** PyTorch, HuggingFace Transformers, BitsAndBytes for quantization

---

## System Architecture

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MAB DEBIASING SYSTEM PIPELINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT TEXT ──► CONTEXT EXTRACTOR ──► BANDIT ALGORITHM ──► ARM SELECTION   │
│       │               │                      │                    │        │
│       │               ▼                      │                    ▼        │
│       │     ┌─────────────────┐              │         ┌──────────────────┐│
│       │     │ - Language ID   │              │         │ DEBIASING ARMS:  ││
│       │     │ - Demographic   │              │         │ 0: No intervention│
│       │     │   markers       │              │         │ 1: Gender steering│
│       │     │ - Topic class   │              │         │ 2: Race steering ││
│       │     │ - Bias risk     │              │         │ 3: Religion steer││
│       │     │   score         │              │         │ 4: Prompt prefix ││
│       │     └─────────────────┘              │         │ 5: Output adjust ││
│       │               │                      │         └──────────────────┘│
│       │               ▼                      │                    │        │
│       │     CONTEXT VECTOR (d=128)           │                    ▼        │
│       │               │                      │         APPLY INTERVENTION  │
│       │               └──────────────────────┘                    │        │
│       │                                                           ▼        │
│       └──────────────────────────────────► LLM GENERATION ◄──────┘        │
│                                                   │                        │
│                                                   ▼                        │
│                                            OUTPUT TEXT                     │
│                                                   │                        │
│                                                   ▼                        │
│                                          REWARD CALCULATOR                 │
│                                          ┌─────────────────┐               │
│                                          │ R = (1-bias) ×  │               │
│                                          │     quality     │               │
│                                          └─────────────────┘               │
│                                                   │                        │
│                                                   ▼                        │
│                                          UPDATE BANDIT PARAMS              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

Create the following project structure:

```
mab_debiasing/
├── config/
│   ├── __init__.py
│   ├── model_config.py          # Model and quantization settings
│   ├── bandit_config.py         # Bandit hyperparameters
│   └── steering_vectors.py      # Pre-computed steering vector paths
├── data/
│   ├── __init__.py
│   ├── bias_evaluation_sets/
│   │   ├── english_bias_prompts.json
│   │   ├── hindi_bias_prompts.json
│   │   └── bengali_bias_prompts.json
│   ├── steering_vectors/
│   │   ├── gender_steering.pt
│   │   ├── race_steering.pt
│   │   └── religion_steering.pt
│   └── topic_classifier/
│       └── topic_model.pt
├── src/
│   ├── __init__.py
│   ├── context_extractor/
│   │   ├── __init__.py
│   │   ├── language_detector.py
│   │   ├── demographic_detector.py
│   │   ├── topic_classifier.py
│   │   ├── bias_risk_scorer.py
│   │   └── context_encoder.py
│   ├── bandit/
│   │   ├── __init__.py
│   │   ├── base_bandit.py
│   │   ├── linucb.py
│   │   ├── thompson_sampling.py
│   │   └── neural_bandit.py
│   ├── debiasing_arms/
│   │   ├── __init__.py
│   │   ├── base_arm.py
│   │   ├── no_intervention.py
│   │   ├── steering_vector_arm.py
│   │   ├── prompt_prefix_arm.py
│   │   └── output_adjustment_arm.py
│   ├── reward/
│   │   ├── __init__.py
│   │   ├── bias_scorer.py
│   │   ├── quality_scorer.py
│   │   └── reward_calculator.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   └── generator.py
│   └── pipeline/
│       ├── __init__.py
│       ├── inference_pipeline.py
│       └── training_pipeline.py
├── scripts/
│   ├── create_steering_vectors.py
│   ├── prepare_evaluation_data.py
│   ├── train_bandit.py
│   ├── evaluate_system.py
│   └── run_inference.py
├── tests/
│   ├── test_context_extractor.py
│   ├── test_bandit.py
│   ├── test_arms.py
│   └── test_pipeline.py
├── notebooks/
│   ├── 01_steering_vector_creation.ipynb
│   ├── 02_bandit_training_analysis.ipynb
│   └── 03_evaluation_results.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## Detailed Component Specifications

### Component 1: Configuration Files

#### 1.1 model_config.py

```python
"""
Model configuration for 24GB VRAM constraint.
All models use 4-bit quantization via BitsAndBytes.
Models are NEVER loaded simultaneously - always sequential loading with explicit cleanup.
"""

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    """Configuration for LLM loading with memory constraints."""
    
    # Primary model choice (pick one based on multilingual performance)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # Best for Hindi/Bengali
    # Alternatives:
    # model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    # model_name: str = "google/gemma-2-9b-it"
    
    # Quantization settings for 24GB VRAM
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Memory management
    device_map: str = "auto"
    max_memory: dict = None  # Will be set to {"cuda:0": "22GB", "cpu": "32GB"}
    offload_folder: str = "./offload"
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Batch size (keep at 1 for memory safety)
    batch_size: int = 1

@dataclass  
class EmbeddingModelConfig:
    """Smaller model for context encoding - loaded separately from main LLM."""
    
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cuda"
    max_seq_length: int = 256
    embedding_dim: int = 384

# Memory cleanup function - CRITICAL for sequential loading
def clear_gpu_memory():
    """
    Aggressively clear GPU memory before loading new model.
    MUST be called before loading any new model.
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
```

#### 1.2 bandit_config.py

```python
"""
Configuration for Multi-Armed Bandit algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class BanditConfig:
    """Hyperparameters for contextual bandit algorithms."""
    
    # Number of arms (debiasing strategies)
    n_arms: int = 6
    
    # Arm definitions
    arm_names: List[str] = field(default_factory=lambda: [
        "no_intervention",      # Arm 0: Baseline - no debiasing
        "gender_steering",      # Arm 1: Apply gender debiasing steering vector
        "race_steering",        # Arm 2: Apply race/ethnicity debiasing steering vector  
        "religion_steering",    # Arm 3: Apply religion debiasing steering vector
        "prompt_prefix",        # Arm 4: Add debiasing instruction prefix to prompt
        "output_adjustment"     # Arm 5: Post-hoc probability adjustment on output
    ])
    
    # Context feature dimension
    context_dim: int = 128
    
    # LinUCB specific
    linucb_alpha: float = 0.5  # Exploration parameter (higher = more exploration)
    
    # Thompson Sampling specific
    ts_prior_mean: float = 0.0
    ts_prior_variance: float = 1.0
    
    # Neural Bandit specific
    neural_hidden_dim: int = 64
    neural_learning_rate: float = 0.001
    
    # Training settings
    warmup_rounds: int = 100  # Random exploration before using learned policy
    update_frequency: int = 1  # Update bandit every N samples
    
    # Reward weights
    bias_weight: float = 0.6   # Weight for bias reduction in reward
    quality_weight: float = 0.4  # Weight for generation quality in reward

@dataclass
class SteeringVectorConfig:
    """Configuration for steering vector arms."""
    
    # Layer range to apply steering (typically middle-to-late layers work best)
    # For 7B model with 32 layers, apply to layers 12-24
    start_layer: int = 12
    end_layer: int = 24
    
    # Steering strength multipliers (tunable per bias type)
    gender_strength: float = 1.5
    race_strength: float = 1.2
    religion_strength: float = 1.0
    
    # Direction of steering (positive = towards fair, negative = towards biased)
    steering_direction: int = -1  # Subtract bias direction to debias
```

---

### Component 2: Context Extractor Module

The context extractor analyzes input text and produces a fixed-dimension context vector that the bandit uses to select the optimal debiasing arm.

#### 2.1 language_detector.py

```python
"""
Language detection for multilingual inputs.
Returns language code and confidence score.
Supports: English (en), Hindi (hi), Bengali (bn)
"""

# Implementation requirements:
# 1. Use langdetect or fasttext for language identification
# 2. Return one-hot encoding for the 3 target languages + "other" category
# 3. Include confidence score (0-1) for the detected language
# 4. Handle code-mixed text (Hindi-English, Bengali-English) by detecting dominant language
# 5. Output: Dict with keys ['language_code', 'confidence', 'one_hot_vector']

# Example output format:
# {
#     'language_code': 'hi',
#     'confidence': 0.92,
#     'one_hot_vector': [0, 1, 0, 0]  # [en, hi, bn, other]
# }

class LanguageDetector:
    """
    Detect language of input text.
    
    Methods:
        __init__(): Initialize detector (use fasttext model 'lid.176.ftz')
        detect(text: str) -> Dict: Return language info
        get_feature_vector(text: str) -> List[float]: Return 4-dim vector [en, hi, bn, other]
    
    Memory: This is a lightweight model (~1MB), can stay loaded.
    """
    pass
```

#### 2.2 demographic_detector.py

```python
"""
Detect demographic markers in input text.
Identifies presence of gender, race/ethnicity, religion, age indicators.
"""

# Implementation requirements:
# 1. Create keyword/pattern lists for each demographic category
# 2. Support all three languages (English, Hindi, Bengali)
# 3. Use regex patterns for name detection (common gendered names)
# 4. Return binary indicators + detected terms

# Demographic categories to detect:

GENDER_MARKERS = {
    'en': {
        'male': ['he', 'him', 'his', 'man', 'boy', 'father', 'husband', 'mr', 'gentleman', ...],
        'female': ['she', 'her', 'hers', 'woman', 'girl', 'mother', 'wife', 'mrs', 'ms', 'lady', ...],
        'male_names': ['john', 'james', 'michael', 'robert', 'david', ...],  # Top 100 male names
        'female_names': ['mary', 'jennifer', 'linda', 'susan', 'jessica', ...]  # Top 100 female names
    },
    'hi': {
        'male': ['वह (पुरुष)', 'उसका', 'आदमी', 'लड़का', 'पिता', 'पति', ...],
        'female': ['वह (स्त्री)', 'उसकी', 'औरत', 'लड़की', 'माता', 'पत्नी', ...],
        'male_names': ['राहुल', 'अमित', 'विकास', 'सुनील', ...],
        'female_names': ['प्रिया', 'अनीता', 'सुनीता', 'रीता', ...]
    },
    'bn': {
        'male': ['সে (পুরুষ)', 'তার', 'পুরুষ', 'ছেলে', 'বাবা', 'স্বামী', ...],
        'female': ['সে (মহিলা)', 'তার', 'মহিলা', 'মেয়ে', 'মা', 'স্ত্রী', ...],
        'male_names': ['রাহুল', 'অমিত', 'সুমিত', ...],
        'female_names': ['প্রিয়া', 'অনিতা', 'সুনীতা', ...]
    }
}

RACE_ETHNICITY_MARKERS = {
    'en': ['asian', 'african', 'european', 'indian', 'chinese', 'black', 'white', 'hispanic', 'latino', ...],
    'hi': ['एशियाई', 'अफ्रीकी', 'यूरोपीय', 'भारतीय', 'चीनी', ...],
    'bn': ['এশীয়', 'আফ্রিকান', 'ইউরোপীয়', 'ভারতীয়', 'চীনা', ...]
}

RELIGION_MARKERS = {
    'en': ['christian', 'muslim', 'hindu', 'buddhist', 'jewish', 'sikh', 'church', 'mosque', 'temple', ...],
    'hi': ['ईसाई', 'मुस्लिम', 'हिंदू', 'बौद्ध', 'यहूदी', 'सिख', 'चर्च', 'मस्जिद', 'मंदिर', ...],
    'bn': ['খ্রিস্টান', 'মুসলিম', 'হিন্দু', 'বৌদ্ধ', 'ইহুদি', 'শিখ', ...]
}

AGE_MARKERS = {
    'en': ['young', 'old', 'elderly', 'teenager', 'child', 'senior', 'millennial', 'boomer', ...],
    'hi': ['युवा', 'बूढ़ा', 'बुजुर्ग', 'किशोर', 'बच्चा', ...],
    'bn': ['যুবক', 'বৃদ্ধ', 'প্রবীণ', 'কিশোর', 'শিশু', ...]
}

class DemographicDetector:
    """
    Detect demographic markers in text.
    
    Methods:
        __init__(): Load keyword dictionaries for all languages
        detect(text: str, language: str) -> Dict: Detect all demographic markers
        get_feature_vector(text: str, language: str) -> List[float]: Return feature vector
    
    Output feature vector (12 dimensions):
        [gender_male, gender_female, gender_neutral,
         race_detected, race_count,
         religion_detected, religion_count,  
         age_young, age_old, age_neutral,
         has_names, demographic_density]
    
    demographic_density = total_markers_found / text_length (normalized)
    """
    pass
```

#### 2.3 topic_classifier.py

```python
"""
Classify input text into bias-sensitive topic categories.
Some topics (employment, crime, relationships) are more prone to bias than others.
"""

# Implementation requirements:
# 1. Train or use pre-trained multilingual classifier
# 2. Categories chosen based on known bias-prone domains

TOPIC_CATEGORIES = [
    'employment_career',      # Jobs, hiring, promotions, salary - HIGH BIAS RISK
    'crime_justice',          # Crime, legal, police, prison - HIGH BIAS RISK
    'relationships_family',   # Marriage, dating, family roles - MEDIUM BIAS RISK
    'education',              # Schools, universities, academic - MEDIUM BIAS RISK
    'healthcare_medical',     # Health, doctors, diseases - MEDIUM BIAS RISK
    'politics_government',    # Politics, elections, policies - HIGH BIAS RISK
    'sports_entertainment',   # Sports, movies, music - LOW BIAS RISK
    'technology_science',     # Tech, science, research - LOW BIAS RISK
    'finance_business',       # Money, business, economy - MEDIUM BIAS RISK
    'general_other'           # Catch-all category - LOW BIAS RISK
]

# Bias risk scores per topic (used in context vector)
TOPIC_BIAS_RISK = {
    'employment_career': 0.9,
    'crime_justice': 0.95,
    'relationships_family': 0.7,
    'education': 0.5,
    'healthcare_medical': 0.6,
    'politics_government': 0.85,
    'sports_entertainment': 0.3,
    'technology_science': 0.2,
    'finance_business': 0.5,
    'general_other': 0.2
}

class TopicClassifier:
    """
    Classify text into bias-sensitive topics.
    
    Methods:
        __init__(model_path: str): Load classifier (fine-tuned multilingual BERT or use zero-shot)
        classify(text: str) -> Dict: Return topic probabilities
        get_feature_vector(text: str) -> List[float]: Return 10-dim topic probability vector
        get_bias_risk(text: str) -> float: Return weighted bias risk based on topic
    
    Implementation option 1: Fine-tune 'xlm-roberta-base' on labeled topic data
    Implementation option 2: Use zero-shot classification with 'facebook/bart-large-mnli'
    
    Memory: ~500MB for xlm-roberta-base, load/unload as needed
    """
    pass
```

#### 2.4 bias_risk_scorer.py

```python
"""
Compute overall bias risk score for input text.
Combines signals from demographic detection, topic classification, and pattern analysis.
"""

# Implementation requirements:
# 1. Aggregate signals from other detectors
# 2. Apply learned weights or heuristic combination
# 3. Output single float [0, 1] where 1 = highest bias risk

class BiasRiskScorer:
    """
    Compute aggregate bias risk score.
    
    Formula:
        risk = w1 * demographic_density + 
               w2 * topic_bias_risk + 
               w3 * stereotype_pattern_score +
               w4 * sensitive_context_score
    
    Methods:
        __init__(): Initialize with default weights
        compute_risk(
            demographic_features: Dict,
            topic_features: Dict,
            text: str,
            language: str
        ) -> float: Return risk score [0, 1]
        
    Stereotype patterns to detect (regex-based):
        - "[demographic] are always/never [trait]"
        - "[demographic] should/shouldn't [action]"
        - "typical [demographic]"
        - "all [demographic]"
    
    Weights (tunable, start with equal weights):
        w1 = 0.25 (demographic density)
        w2 = 0.35 (topic bias risk)  
        w3 = 0.25 (stereotype patterns)
        w4 = 0.15 (sensitive context - questions about groups, comparisons)
    """
    pass
```

#### 2.5 context_encoder.py

```python
"""
Encode all extracted features into a fixed-dimension context vector.
This vector is the input to the bandit algorithm.
"""

# Implementation requirements:
# 1. Concatenate all feature vectors from sub-modules
# 2. Project to fixed dimension (128) using learned or fixed projection
# 3. Normalize output vector

class ContextEncoder:
    """
    Combine all context features into bandit input vector.
    
    Input features (total raw dim = ~40):
        - Language one-hot: 4 dims
        - Demographic features: 12 dims
        - Topic probabilities: 10 dims
        - Bias risk score: 1 dim
        - Text embedding (compressed): 16 dims (from 384-dim sentence embedding)
        - Additional features: ~5 dims (text length, question flag, etc.)
    
    Methods:
        __init__(output_dim: int = 128): Initialize projection layer
        encode(
            language_features: List[float],
            demographic_features: List[float],
            topic_features: List[float],
            bias_risk: float,
            text_embedding: List[float],
            additional_features: List[float]
        ) -> np.ndarray: Return context vector of shape (128,)
        
    Projection options:
        1. Simple linear projection: W @ concat(features) + b
        2. MLP: Linear -> ReLU -> Linear
        3. PCA-based projection (pre-computed on sample data)
    
    Output: L2-normalized vector of dimension 128
    """
    pass
```

---

### Component 3: Bandit Algorithms

Implement multiple bandit algorithms. The system should support switching between them.

#### 3.1 base_bandit.py

```python
"""
Abstract base class for all bandit algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, List

class BaseBandit(ABC):
    """
    Abstract base class for contextual bandit algorithms.
    
    Attributes:
        n_arms: Number of arms (debiasing strategies)
        context_dim: Dimension of context vector
        arm_names: List of arm names for logging
        history: List of (context, arm, reward) tuples
    
    Methods:
        select_arm(context: np.ndarray) -> Tuple[int, float]: Select arm, return (arm_id, confidence)
        update(context: np.ndarray, arm: int, reward: float) -> None: Update model with observed reward
        get_arm_values(context: np.ndarray) -> np.ndarray: Get expected values for all arms
        save(path: str) -> None: Save model state
        load(path: str) -> None: Load model state
        reset() -> None: Reset model to initial state
    """
    
    @abstractmethod
    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Select which arm to pull given context.
        
        Args:
            context: Context vector of shape (context_dim,)
            
        Returns:
            Tuple of (selected_arm_index, confidence_score)
        """
        pass
    
    @abstractmethod
    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """
        Update model after observing reward.
        
        Args:
            context: Context vector used for selection
            arm: Index of arm that was pulled
            reward: Observed reward in [0, 1]
        """
        pass
```

#### 3.2 linucb.py

```python
"""
Linear Upper Confidence Bound (LinUCB) algorithm.
Standard contextual bandit with linear reward model and UCB exploration.
"""

# Implementation requirements:
# 1. Maintain A matrix (d x d) and b vector (d x 1) for each arm
# 2. Compute theta = A^{-1} @ b for each arm
# 3. Select arm with highest UCB: theta.T @ context + alpha * sqrt(context.T @ A^{-1} @ context)
# 4. Update A and b after observing reward

class LinUCB(BaseBandit):
    """
    LinUCB contextual bandit algorithm.
    
    For each arm a:
        A_a: (d x d) matrix, initialized to identity
        b_a: (d x 1) vector, initialized to zeros
        
    Selection:
        theta_a = A_a^{-1} @ b_a  (ridge regression estimate)
        p_a = theta_a.T @ x + alpha * sqrt(x.T @ A_a^{-1} @ x)  (UCB)
        Select arm with highest p_a
        
    Update:
        A_a = A_a + x @ x.T
        b_a = b_a + r * x
        
    Hyperparameters:
        alpha: Exploration parameter (default 0.5)
               Higher alpha = more exploration
               Lower alpha = more exploitation
               
    Initialization:
        __init__(n_arms: int, context_dim: int, alpha: float = 0.5)
        
    Methods:
        select_arm(context: np.ndarray) -> Tuple[int, float]
        update(context: np.ndarray, arm: int, reward: float) -> None
        get_arm_values(context: np.ndarray) -> np.ndarray: Return UCB values for all arms
        
    Memory efficient implementation:
        - Use numpy for matrix operations
        - Store A^{-1} directly and update using Sherman-Morrison formula
        - This avoids repeated matrix inversion
        
    Sherman-Morrison update for A^{-1}:
        A_new^{-1} = A^{-1} - (A^{-1} @ x @ x.T @ A^{-1}) / (1 + x.T @ A^{-1} @ x)
    """
    pass
```

#### 3.3 thompson_sampling.py

```python
"""
Thompson Sampling with linear reward model.
Bayesian approach - maintains posterior distribution over arm parameters.
"""

# Implementation requirements:
# 1. Maintain posterior distribution for each arm's parameter vector
# 2. Sample from posterior to get parameter estimates
# 3. Select arm with highest sampled expected reward
# 4. Update posterior after observing reward

class ThompsonSamplingLinear(BaseBandit):
    """
    Thompson Sampling with linear Gaussian model.
    
    Model: reward ~ N(theta.T @ x, sigma^2)
    Prior: theta ~ N(mu_0, Sigma_0)
    Posterior: theta | data ~ N(mu_n, Sigma_n)
    
    For each arm a:
        mu_a: (d,) posterior mean
        B_a: (d, d) posterior precision matrix (inverse covariance)
        f_a: (d,) weighted sum of contexts (B_a @ mu_a = f_a)
        
    Selection:
        Sample theta_a ~ N(mu_a, B_a^{-1}) for each arm
        Select arm with highest theta_a.T @ x
        
    Update (after observing reward r for arm a with context x):
        B_a = B_a + x @ x.T / sigma^2
        f_a = f_a + r * x / sigma^2
        mu_a = B_a^{-1} @ f_a
        
    Hyperparameters:
        sigma: Observation noise std (default 0.5)
        prior_variance: Prior variance for theta (default 1.0)
        
    Implementation note:
        - Use Cholesky decomposition for efficient sampling from multivariate normal
        - Store B (precision) and f directly, compute mu when needed
    """
    pass
```

#### 3.4 neural_bandit.py

```python
"""
Neural network-based contextual bandit.
Uses a neural network to model the reward function with dropout-based exploration.
"""

# Implementation requirements:
# 1. Neural network that maps (context, arm_one_hot) -> predicted reward
# 2. Use MC Dropout or ensemble for uncertainty estimation
# 3. Select arm using UCB with neural uncertainty estimates

class NeuralBandit(BaseBandit):
    """
    Neural contextual bandit with uncertainty estimation.
    
    Architecture:
        Input: concat(context, arm_one_hot) -> (context_dim + n_arms,)
        Hidden: Linear(input_dim, 64) -> ReLU -> Dropout(0.1)
        Hidden: Linear(64, 32) -> ReLU -> Dropout(0.1)
        Output: Linear(32, 1) -> reward prediction
        
    Uncertainty estimation via MC Dropout:
        - At selection time, run forward pass K times with dropout enabled
        - Compute mean and std of predictions
        - UCB = mean + alpha * std
        
    Selection:
        For each arm:
            x = concat(context, one_hot(arm))
            Run K forward passes with dropout
            mu_a = mean(predictions)
            sigma_a = std(predictions)
            ucb_a = mu_a + alpha * sigma_a
        Select arm with highest ucb_a
        
    Training:
        - Maintain replay buffer of (context, arm, reward) tuples
        - Train on mini-batches using MSE loss
        - Update every N observations
        
    Hyperparameters:
        hidden_dim: Hidden layer size (default 64)
        dropout_rate: Dropout probability (default 0.1)
        n_mc_samples: Number of MC dropout samples (default 10)
        alpha: UCB exploration parameter (default 0.5)
        learning_rate: Adam learning rate (default 0.001)
        buffer_size: Replay buffer capacity (default 10000)
        batch_size: Training batch size (default 32)
        update_frequency: Train every N samples (default 10)
        
    PyTorch implementation:
        - Use nn.Module for network
        - CPU-only to avoid GPU memory conflicts with LLM
    """
    pass
```

---

### Component 4: Debiasing Arms

Each arm represents a different debiasing intervention strategy.

#### 4.1 base_arm.py

```python
"""
Abstract base class for debiasing arms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseArm(ABC):
    """
    Abstract base class for debiasing intervention strategies.
    
    Attributes:
        name: Arm identifier string
        requires_model_access: Whether arm needs to modify model internals
        strength: Intervention strength (for tunable arms)
        
    Methods:
        apply(
            model: Any,
            tokenizer: Any,
            input_text: str,
            **kwargs
        ) -> Dict: Apply intervention and return modified state
        
        generate(
            model: Any,
            tokenizer: Any,
            input_text: str,
            generation_config: Dict,
            **kwargs
        ) -> str: Generate text with intervention applied
    """
    
    @abstractmethod
    def apply(self, model, tokenizer, input_text: str, **kwargs) -> Dict:
        """Apply debiasing intervention. Return dict with any modified state."""
        pass
    
    @abstractmethod
    def generate(self, model, tokenizer, input_text: str, generation_config: Dict, **kwargs) -> str:
        """Generate text with intervention applied."""
        pass
```

#### 4.2 no_intervention.py

```python
"""
Arm 0: No intervention baseline.
Standard generation without any debiasing.
"""

class NoInterventionArm(BaseArm):
    """
    Baseline arm - standard LLM generation without modification.
    
    This arm exists to:
        1. Provide baseline comparison
        2. Be selected when input has low bias risk
        3. Preserve generation quality when debiasing not needed
        
    Implementation:
        Simply call model.generate() with standard parameters.
        No modification to model, input, or output.
    """
    pass
```

#### 4.3 steering_vector_arm.py

```python
"""
Arms 1-3: Steering vector-based debiasing.
Apply pre-computed steering vectors to model activations during generation.
"""

# Implementation requirements:
# 1. Load pre-computed steering vectors for each bias type
# 2. Apply activation engineering during forward pass
# 3. Support variable strength parameter

class SteeringVectorArm(BaseArm):
    """
    Apply steering vectors to hidden states during generation.
    
    Steering vector methodology:
        1. Pre-compute steering vector S = E[h_biased] - E[h_neutral]
           where h are hidden states at layer L
        2. During generation, modify hidden states: h' = h - alpha * S
           (subtracting bias direction = moving toward neutral)
        3. Apply to multiple layers for stronger effect
        
    Attributes:
        bias_type: 'gender', 'race', or 'religion'
        steering_vector: Pre-loaded tensor of shape (n_layers, hidden_dim)
        strength: Scaling factor alpha (default 1.0)
        start_layer: First layer to apply steering
        end_layer: Last layer to apply steering
        
    Implementation using forward hooks:
        
        def steering_hook(module, input, output):
            # output is tuple: (hidden_states, ...)
            # hidden_states shape: (batch, seq_len, hidden_dim)
            layer_idx = get_layer_index(module)
            if start_layer <= layer_idx <= end_layer:
                steering = self.steering_vector[layer_idx]  # (hidden_dim,)
                output[0][:, :, :] -= self.strength * steering
            return output
            
        # Register hooks before generation
        hooks = []
        for layer in model.model.layers:
            hook = layer.register_forward_hook(steering_hook)
            hooks.append(hook)
            
        # Generate
        output = model.generate(...)
        
        # Remove hooks after generation
        for hook in hooks:
            hook.remove()
            
    Methods:
        __init__(bias_type: str, vector_path: str, strength: float, layers: Tuple[int, int])
        load_vector(path: str) -> torch.Tensor
        apply(model, tokenizer, input_text, **kwargs) -> Dict
        generate(model, tokenizer, input_text, generation_config, **kwargs) -> str
        
    Create separate instances for each bias type:
        - GenderSteeringArm (Arm 1)
        - RaceSteeringArm (Arm 2)  
        - ReligionSteeringArm (Arm 3)
    """
    pass


# Script to create steering vectors (run once, save to disk):
"""
Creating steering vectors requires:
1. Contrastive prompt pairs (biased vs neutral)
2. Extract hidden states for each pair
3. Compute mean difference

Example contrastive pairs for gender:
    biased: "The nurse said she would check the patient's vitals."
    neutral: "The nurse said they would check the patient's vitals."
    
    biased: "The engineer fixed his computer."
    neutral: "The engineer fixed their computer."

Process:
1. Create 100+ contrastive pairs per bias type
2. Run each through model, extract hidden states at each layer
3. steering_vector[layer] = mean(biased_states) - mean(neutral_states)
4. Save as .pt file

This should be done ONCE and saved, not computed at runtime.
See scripts/create_steering_vectors.py for full implementation.
"""
```

#### 4.4 prompt_prefix_arm.py

```python
"""
Arm 4: Debiasing prompt prefix.
Prepend instruction to be unbiased to the user's input.
"""

class PromptPrefixArm(BaseArm):
    """
    Add debiasing instruction prefix to input prompt.
    
    This is a simple but effective intervention that instructs the model
    to be fair and unbiased in its response.
    
    Prefix templates (language-specific):
    
    ENGLISH:
        "Please provide a fair, balanced, and unbiased response. 
         Avoid stereotypes and treat all demographic groups equally. "
    
    HINDI:
        "कृपया एक निष्पक्ष और संतुलित उत्तर दें। 
         रूढ़िवादिता से बचें और सभी समूहों के साथ समान व्यवहार करें। "
    
    BENGALI:
        "দয়া করে একটি নিরপেক্ষ এবং সুষম উত্তর দিন। 
         স্টেরিওটাইপ এড়িয়ে চলুন এবং সকল গোষ্ঠীর সাথে সমান আচরণ করুন। "
    
    Methods:
        __init__(): Load prefix templates for each language
        apply(model, tokenizer, input_text, language='en', **kwargs) -> Dict:
            Return {'modified_input': prefix + input_text}
        generate(model, tokenizer, input_text, generation_config, language='en', **kwargs) -> str:
            Generate with prefixed input
            
    Notes:
        - Detect language from context or accept as parameter
        - Prefix is added to user content, not system prompt (unless using chat template)
        - For chat models, can add as system message instead
    """
    pass
```

#### 4.5 output_adjustment_arm.py

```python
"""
Arm 5: Output probability adjustment.
Post-hoc modification of token probabilities to reduce biased outputs.
"""

class OutputAdjustmentArm(BaseArm):
    """
    Adjust output token probabilities during generation to reduce bias.
    
    Method 1: Token probability dampening
        - Identify bias-associated tokens (e.g., gendered pronouns when gender unknown)
        - Reduce their probability during sampling
        - Boost neutral alternatives
        
    Method 2: Contrastive decoding
        - Run generation twice: once normal, once with biased prompt
        - Output = normal_logits - alpha * biased_logits
        - This cancels out bias-correlated patterns
        
    Implementation (Method 1 - Probability Dampening):
    
        # Define token adjustments
        DAMPENED_TOKENS = {
            'he': -2.0,   # Log probability adjustment
            'she': -2.0,
            'him': -2.0,
            'her': -2.0,
        }
        BOOSTED_TOKENS = {
            'they': +1.0,
            'them': +1.0,
            'their': +1.0,
        }
        
        # Custom logits processor
        class BiasAdjustmentLogitsProcessor:
            def __call__(self, input_ids, scores):
                for token_id, adjustment in adjustments.items():
                    scores[:, token_id] += adjustment
                return scores
                
        # Use in generation
        model.generate(
            ...,
            logits_processor=[BiasAdjustmentLogitsProcessor()]
        )
        
    Implementation (Method 2 - Contrastive Decoding):
        - More expensive (2x generation cost)
        - But more principled and effective
        - Use when high bias risk detected
        
    Methods:
        __init__(method: str = 'dampening', strength: float = 1.0)
        apply(model, tokenizer, input_text, **kwargs) -> Dict
        generate(model, tokenizer, input_text, generation_config, **kwargs) -> str
        
    Notes:
        - Method 1 is faster, use for lower-risk inputs
        - Method 2 is slower but more effective, use for high-risk inputs
        - Arm can adaptively choose method based on bias risk score
    """
    pass
```

---

### Component 5: Reward Calculator

Computes the reward signal used to update the bandit.

#### 5.1 bias_scorer.py

```python
"""
Compute bias score for generated text.
Lower bias score = better (less biased).
"""

# Implementation requirements:
# 1. Multiple bias metrics for robustness
# 2. Fast computation (used after every generation)
# 3. Support for all three languages

class BiasScorer:
    """
    Score generated text for bias.
    
    Metrics to implement:
    
    1. Embedding-based Association Score:
        - Compute embedding of generated text
        - Measure association with stereotypical vs. anti-stereotypical concepts
        - Use WEAT/SEAT style measurement but simplified for single text
        
    2. Lexical Bias Indicators:
        - Count usage of stereotypical terms/phrases
        - Compare against neutral alternatives
        
    3. Sentiment Disparity (for demographic mentions):
        - If text mentions multiple demographic groups
        - Compare sentiment expressed toward each
        - Disparity indicates bias
        
    4. Stereotype Classifier Score:
        - Train simple classifier to detect stereotypical content
        - Or use existing toxicity/bias classifiers (e.g., Perspective API-style)
        
    Aggregation:
        bias_score = weighted_average([
            association_score,
            lexical_score,
            sentiment_disparity,
            classifier_score
        ])
        
    Methods:
        __init__(): Load embedding model, classifiers, lexicons
        score(generated_text: str, input_text: str, language: str) -> float:
            Return bias score in [0, 1] where 0 = no bias, 1 = high bias
            
    For fast scoring (during bandit training):
        - Use only embedding-based and lexical metrics
        - Skip classifier if too slow
        
    Memory management:
        - Embedding model (MiniLM) is small (~100MB), can stay loaded
        - Load lazily and unload if needed
    """
    pass
```

#### 5.2 quality_scorer.py

```python
"""
Compute generation quality score.
Ensures debiasing doesn't degrade output quality.
"""

class QualityScorer:
    """
    Score generated text for quality metrics.
    
    Metrics to implement:
    
    1. Fluency (Perplexity-based):
        - Compute perplexity of generated text under the model
        - Lower perplexity = more fluent
        - Normalize to [0, 1] score
        
        WARNING: Computing perplexity requires model forward pass.
        This is expensive. Consider caching or using proxy metric.
        
    2. Coherence with Input:
        - Semantic similarity between input and output
        - Using same embedding model as bias scorer
        - Ensures response is relevant
        
    3. Length Appropriateness:
        - Very short outputs often indicate quality issues
        - Very long outputs may indicate rambling
        - Score based on reasonable length for input type
        
    4. Repetition Detection:
        - Check for repeated phrases/sentences
        - Debiasing interventions sometimes cause repetition
        
    Aggregation:
        quality_score = weighted_average([
            fluency_score,
            coherence_score,
            length_score,
            (1 - repetition_score)
        ])
        
    Methods:
        __init__(): Initialize metrics
        score(generated_text: str, input_text: str, language: str) -> float:
            Return quality score in [0, 1] where 1 = high quality
            
    Fast mode (skip perplexity):
        - Use only coherence, length, repetition
        - Much faster, still useful signal
    """
    pass
```

#### 5.3 reward_calculator.py

```python
"""
Combine bias and quality scores into single reward signal.
"""

class RewardCalculator:
    """
    Compute final reward for bandit update.
    
    Formula:
        reward = (1 - bias_score) * bias_weight + quality_score * quality_weight
        
        Where:
        - bias_score in [0, 1], lower is better (less biased)
        - quality_score in [0, 1], higher is better
        - bias_weight + quality_weight = 1.0
        
    Default weights:
        bias_weight = 0.6
        quality_weight = 0.4
        
    Methods:
        __init__(bias_weight: float = 0.6, quality_weight: float = 0.4)
        
        calculate(
            generated_text: str,
            input_text: str,
            language: str,
            bias_scorer: BiasScorer,
            quality_scorer: QualityScorer
        ) -> Dict:
            Return {
                'reward': float,  # Final reward in [0, 1]
                'bias_score': float,
                'quality_score': float,
                'bias_component': float,  # (1 - bias_score) * bias_weight
                'quality_component': float  # quality_score * quality_weight
            }
            
    Reward shaping (optional enhancements):
        - Bonus for improvement over baseline (no intervention)
        - Penalty for quality drop beyond threshold
        - Curriculum: start with quality_weight high, gradually increase bias_weight
    """
    pass
```

---

### Component 6: LLM Interface

#### 6.1 model_loader.py

```python
"""
Load and manage LLM with 24GB VRAM constraint.
CRITICAL: Models must be loaded sequentially with explicit cleanup.
"""

# Implementation requirements:
# 1. 4-bit quantization using BitsAndBytes
# 2. Proper GPU memory cleanup before loading
# 3. Singleton pattern to prevent multiple loads

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc

class ModelLoader:
    """
    Singleton model loader with memory management.
    
    CRITICAL MEMORY RULES:
        1. NEVER load multiple models simultaneously
        2. ALWAYS call unload() before loading a different model
        3. ALWAYS call clear_memory() between loads
        
    Usage:
        loader = ModelLoader()
        
        # Load model
        model, tokenizer = loader.load(model_name)
        
        # Use model...
        
        # Before loading different model
        loader.unload()
        
    Quantization config for 24GB VRAM:
        - 4-bit NF4 quantization
        - Double quantization enabled
        - Compute dtype: float16
        - Max memory: {"cuda:0": "22GB", "cpu": "32GB"}
        
    Methods:
        __init__()
        load(model_name: str) -> Tuple[model, tokenizer]
        unload() -> None: Unload current model and free memory
        clear_memory() -> None: Aggressive GPU memory cleanup
        is_loaded() -> bool
        get_current_model_name() -> Optional[str]
        
    Implementation:
    
        def load(self, model_name: str):
            if self._model is not None:
                self.unload()
                
            self.clear_memory()
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={"cuda:0": "22GB", "cpu": "32GB"},
                torch_dtype=torch.float16,
            )
            self._model_name = model_name
            
            return self._model, self._tokenizer
            
        def unload(self):
            if self._model is not None:
                del self._model
                del self._tokenizer
                self._model = None
                self._tokenizer = None
                self._model_name = None
                self.clear_memory()
                
        def clear_memory(self):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    """
    pass
```

#### 6.2 generator.py

```python
"""
Text generation with intervention support.
"""

class Generator:
    """
    Generate text with optional debiasing interventions.
    
    Methods:
        __init__(model, tokenizer)
        
        generate(
            input_text: str,
            intervention: Optional[BaseArm] = None,
            generation_config: Optional[Dict] = None,
            **intervention_kwargs
        ) -> str:
            Generate text, optionally with debiasing intervention applied
            
        generate_batch(
            inputs: List[str],
            intervention: Optional[BaseArm] = None,
            generation_config: Optional[Dict] = None,
        ) -> List[str]:
            Batch generation (batch_size=1 for memory safety, sequential processing)
            
    Default generation config:
        {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id,
        }
        
    Intervention application:
        if intervention is not None:
            output = intervention.generate(
                self.model, 
                self.tokenizer, 
                input_text, 
                generation_config,
                **intervention_kwargs
            )
        else:
            # Standard generation
            ...
    """
    pass
```

---

### Component 7: Pipeline Integration

#### 7.1 inference_pipeline.py

```python
"""
Complete inference pipeline integrating all components.
"""

class MABDebiasInferencePipeline:
    """
    Main inference pipeline for MAB debiasing system.
    
    Pipeline flow:
        1. Receive input text
        2. Extract context features
        3. Query bandit for arm selection
        4. Apply selected intervention
        5. Generate response
        6. Compute reward (bias + quality scores)
        7. Update bandit with observed reward
        8. Return response
        
    Initialization:
        __init__(
            model_name: str,
            bandit_type: str = 'linucb',  # 'linucb', 'thompson', 'neural'
            bandit_config: BanditConfig = None,
            steering_vector_paths: Dict[str, str] = None,  # Paths to pre-computed vectors
            enable_learning: bool = True,  # Whether to update bandit (False for eval)
        )
        
    Methods:
        load_components() -> None:
            Load model, bandit, arms, scorers
            IMPORTANT: Load sequentially, clear memory between loads
            
        process(
            input_text: str,
            return_details: bool = False
        ) -> Union[str, Dict]:
            Main method - process input and return response
            If return_details=True, also return:
            {
                'response': str,
                'selected_arm': str,
                'arm_confidence': float,
                'context_features': Dict,
                'bias_score': float,
                'quality_score': float,
                'reward': float,
            }
            
        process_batch(
            inputs: List[str],
            return_details: bool = False
        ) -> List[Union[str, Dict]]:
            Process multiple inputs sequentially
            
        save_state(path: str) -> None:
            Save bandit state for later resumption
            
        load_state(path: str) -> None:
            Load saved bandit state
            
    Memory management:
        - Context extractor components (small) stay loaded
        - LLM (large) loaded once at init
        - Embedding model for scoring loaded/unloaded as needed
        - If OOM error, provide clear error message with memory diagnostics
        
    Example usage:
        
        pipeline = MABDebiasInferencePipeline(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            bandit_type="linucb",
        )
        pipeline.load_components()
        
        # Single inference
        response = pipeline.process("The doctor told her patient...")
        
        # Detailed inference
        result = pipeline.process(
            "The CEO announced the company would...",
            return_details=True
        )
        print(f"Selected arm: {result['selected_arm']}")
        print(f"Bias score: {result['bias_score']}")
        
        # Save bandit state
        pipeline.save_state("./checkpoints/bandit_state.pkl")
    """
    pass
```

#### 7.2 training_pipeline.py

```python
"""
Training pipeline for bandit learning.
"""

class MABDebiasTrainingPipeline:
    """
    Training pipeline to learn optimal arm selection policy.
    
    Training modes:
    
    1. Online Learning:
        - Process inputs one by one
        - Update bandit after each (context, arm, reward)
        - Standard bandit learning
        
    2. Batch Learning:
        - Collect batch of experiences
        - Update bandit periodically
        - More stable for neural bandit
        
    3. Evaluation Mode:
        - Disable bandit updates
        - Measure performance of fixed policy
        
    Training data:
        - Use bias evaluation datasets (e.g., BBQ, WinoBias, CrowS-Pairs)
        - Augment with multilingual bias prompts (Hindi, Bengali)
        - Mix bias-prone and neutral prompts
        
    Methods:
        __init__(
            inference_pipeline: MABDebiasInferencePipeline,
            training_config: Dict
        )
        
        train(
            dataset: List[str],  # List of input prompts
            n_epochs: int = 1,
            eval_every: int = 100,
            save_every: int = 500,
            checkpoint_dir: str = "./checkpoints"
        ) -> Dict:
            Train bandit on dataset
            Return training metrics
            
        evaluate(
            eval_dataset: List[str],
            baseline_arm: int = 0,  # Compare against no-intervention baseline
        ) -> Dict:
            Evaluate current policy
            Return:
            {
                'mean_reward': float,
                'mean_bias_score': float,
                'mean_quality_score': float,
                'arm_selection_distribution': Dict[str, float],
                'improvement_over_baseline': float,
            }
            
        warmup(
            dataset: List[str],
            n_samples: int = 100
        ) -> None:
            Random arm selection for warmup period
            Collects initial data for bandit
            
    Training loop pseudocode:
        
        for epoch in range(n_epochs):
            for i, input_text in enumerate(dataset):
                # Process with learning enabled
                result = self.pipeline.process(input_text, return_details=True)
                
                # Log metrics
                self.log_metrics(result)
                
                # Periodic evaluation
                if i % eval_every == 0:
                    eval_results = self.evaluate(eval_dataset[:100])
                    self.log_eval(eval_results)
                    
                # Periodic checkpoint
                if i % save_every == 0:
                    self.pipeline.save_state(f"{checkpoint_dir}/bandit_step_{i}.pkl")
                    
        # Final save
        self.pipeline.save_state(f"{checkpoint_dir}/bandit_final.pkl")
    """
    pass
```

---

### Component 8: Evaluation and Scripts

#### 8.1 scripts/prepare_evaluation_data.py

```python
"""
Prepare multilingual bias evaluation datasets.
"""

# Datasets to include:

# 1. English bias benchmarks:
#    - BBQ (Bias Benchmark for QA): https://github.com/nyu-mll/BBQ
#    - WinoBias: https://github.com/uclanlp/corefBias
#    - CrowS-Pairs: https://github.com/nyu-mll/crows-pairs
#    - StereoSet: https://github.com/moinnadeem/StereoSet

# 2. Hindi bias prompts (create or adapt):
#    - Translate subset of BBQ to Hindi
#    - Create Hindi-specific stereotypes dataset
#    - Include occupation-gender associations relevant to Indian context

# 3. Bengali bias prompts (create or adapt):
#    - Translate subset of BBQ to Bengali
#    - Create Bengali-specific stereotypes dataset
#    - Include religion and regional stereotypes

# Output format (JSON):
# {
#     "id": "en_001",
#     "language": "en",
#     "input": "The doctor walked into the room. [pronoun] looked at the patient chart.",
#     "bias_type": "gender",
#     "topic": "employment_career",
#     "expected_neutral": "they",
#     "stereotypical": "he",
#     "anti_stereotypical": "she",
# }

# Script should:
# 1. Download/load existing datasets
# 2. Convert to common format
# 3. Create train/eval splits
# 4. Save to data/bias_evaluation_sets/
```

#### 8.2 scripts/create_steering_vectors.py

```python
"""
Create steering vectors for bias types.
Run ONCE before training. Requires model access.
"""

# Process:
# 1. Load model (4-bit quantized)
# 2. For each bias type:
#    a. Load contrastive prompt pairs
#    b. Extract hidden states for each pair
#    c. Compute mean difference: steering = mean(biased) - mean(neutral)
#    d. Save steering vector

# Contrastive pairs format:
# {
#     "bias_type": "gender",
#     "biased": "The nurse said she would help.",
#     "neutral": "The nurse said they would help."
# }

# Need 100+ pairs per bias type for robust steering vector

# Output:
# - data/steering_vectors/gender_steering.pt
# - data/steering_vectors/race_steering.pt
# - data/steering_vectors/religion_steering.pt

# Each file contains:
# {
#     'steering_vector': Tensor of shape (n_layers, hidden_dim),
#     'bias_type': str,
#     'n_pairs_used': int,
#     'model_name': str,
# }

# MEMORY NOTE:
# - Load model, compute vectors, unload model
# - Save vectors to disk
# - Vectors are small (~50MB each), can be loaded at runtime
```

#### 8.3 scripts/train_bandit.py

```python
"""
Main training script for bandit.
"""

# Usage:
# python scripts/train_bandit.py \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --bandit_type linucb \
#     --dataset_path data/bias_evaluation_sets/ \
#     --n_epochs 3 \
#     --checkpoint_dir ./checkpoints \
#     --eval_every 100

# Arguments:
# --model_name: HuggingFace model identifier
# --bandit_type: 'linucb', 'thompson', 'neural'
# --dataset_path: Path to evaluation datasets
# --n_epochs: Number of training epochs
# --warmup_samples: Number of random exploration samples
# --checkpoint_dir: Where to save bandit states
# --eval_every: Evaluation frequency
# --linucb_alpha: LinUCB exploration parameter (if using linucb)
# --bias_weight: Weight for bias in reward (default 0.6)
# --quality_weight: Weight for quality in reward (default 0.4)

# Output:
# - Checkpoints saved to checkpoint_dir
# - Training metrics logged to wandb/tensorboard (optional)
# - Final evaluation results printed
```

#### 8.4 scripts/evaluate_system.py

```python
"""
Comprehensive evaluation of trained system.
"""

# Evaluation metrics:
# 1. Bias reduction: Compare bias scores with vs without system
# 2. Quality preservation: Compare quality scores with vs without system
# 3. Arm selection patterns: Which arms selected for which contexts
# 4. Per-language performance: Breakdown by English/Hindi/Bengali
# 5. Per-bias-type performance: Breakdown by gender/race/religion
# 6. Baseline comparison: vs no-intervention, vs single-arm strategies

# Output:
# - Detailed metrics in JSON
# - Visualization plots
# - Summary statistics

# Usage:
# python scripts/evaluate_system.py \
#     --bandit_checkpoint ./checkpoints/bandit_final.pkl \
#     --model_name "Qwen/Qwen2.5-7B-Instruct" \
#     --eval_dataset data/bias_evaluation_sets/test.json \
#     --output_dir ./results
```

---

## Implementation Order

Implement components in this order:

### Phase 1: Foundation (Week 1)
1. `config/` - All configuration files
2. `src/llm/model_loader.py` - Model loading with memory management
3. `src/llm/generator.py` - Basic generation
4. Test: Load model, generate text, unload cleanly

### Phase 2: Context Extraction (Week 1-2)
5. `src/context_extractor/language_detector.py`
6. `src/context_extractor/demographic_detector.py`
7. `src/context_extractor/topic_classifier.py`
8. `src/context_extractor/bias_risk_scorer.py`
9. `src/context_extractor/context_encoder.py`
10. Test: Extract context from sample inputs

### Phase 3: Bandit Algorithms (Week 2)
11. `src/bandit/base_bandit.py`
12. `src/bandit/linucb.py`
13. `src/bandit/thompson_sampling.py`
14. `src/bandit/neural_bandit.py`
15. Test: Bandit arm selection and updates

### Phase 4: Debiasing Arms (Week 2-3)
16. `src/debiasing_arms/base_arm.py`
17. `src/debiasing_arms/no_intervention.py`
18. `src/debiasing_arms/prompt_prefix_arm.py`
19. `scripts/create_steering_vectors.py` - Run to create vectors
20. `src/debiasing_arms/steering_vector_arm.py`
21. `src/debiasing_arms/output_adjustment_arm.py`
22. Test: Each arm independently

### Phase 5: Reward Calculation (Week 3)
23. `src/reward/bias_scorer.py`
24. `src/reward/quality_scorer.py`
25. `src/reward/reward_calculator.py`
26. Test: Score sample generations

### Phase 6: Pipeline Integration (Week 3)
27. `src/pipeline/inference_pipeline.py`
28. `src/pipeline/training_pipeline.py`
29. Test: End-to-end inference

### Phase 7: Training & Evaluation (Week 4)
30. `scripts/prepare_evaluation_data.py`
31. `scripts/train_bandit.py`
32. `scripts/evaluate_system.py`
33. Run training and evaluation

---

## Dependencies (requirements.txt)

```
# Core
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
bitsandbytes>=0.41.0

# Sentence embeddings
sentence-transformers>=2.2.0

# Language detection
langdetect>=1.0.9
fasttext>=0.9.2

# Numerical
numpy>=1.24.0
scipy>=1.10.0

# Data handling
pandas>=2.0.0
datasets>=2.14.0

# Utilities
tqdm>=4.65.0
pyyaml>=6.0

# Logging (optional)
wandb>=0.15.0
tensorboard>=2.14.0

# Testing
pytest>=7.4.0
```

---

## Key Implementation Notes

### Memory Management (CRITICAL)

```python
# ALWAYS follow this pattern when switching models or heavy components:

def safe_model_switch():
    # 1. Delete existing references
    del model
    del tokenizer
    
    # 2. Clear Python garbage
    import gc
    gc.collect()
    
    # 3. Clear CUDA cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # 4. Wait a moment
    import time
    time.sleep(1)
    
    # 5. Load new model
    new_model = load_model(...)
```

### Steering Vector Hook Pattern

```python
# Correct pattern for applying steering vectors during generation:

def apply_steering(model, steering_vector, strength, start_layer, end_layer):
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Apply steering
            hidden_states = hidden_states - strength * steering_vector[layer_idx]
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states
        return hook
    
    # Register hooks
    for idx, layer in enumerate(model.model.layers):
        if start_layer <= idx <= end_layer:
            hook = layer.register_forward_hook(make_hook(idx))
            hooks.append(hook)
    
    return hooks  # Remember to remove after generation!
```

### Context Vector Normalization

```python
# Always normalize context vectors before bandit:

def normalize_context(context_vector):
    norm = np.linalg.norm(context_vector)
    if norm > 0:
        return context_vector / norm
    return context_vector
```

---

## Success Criteria

1. **Functional**: System runs end-to-end without OOM errors on 24GB GPU
2. **Learning**: Bandit learns to select different arms for different contexts
3. **Effective**: Measurable bias reduction compared to no-intervention baseline
4. **Quality-preserving**: Generation quality does not degrade significantly
5. **Multilingual**: Works for English, Hindi, and Bengali inputs

---

## Testing Checklist

- [ ] Model loads in 4-bit and generates coherent text
- [ ] Context extractor produces valid feature vectors
- [ ] Each bandit algorithm selects arms and updates correctly
- [ ] Each debiasing arm applies without errors
- [ ] Steering vectors modify hidden states as expected
- [ ] Reward calculator produces sensible scores
- [ ] Full pipeline processes inputs end-to-end
- [ ] Training loop runs without memory leaks
- [ ] Checkpoints save and load correctly
- [ ] Evaluation metrics compute correctly
