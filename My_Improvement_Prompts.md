# Fair-CB Implementation Prompts for Antigravity/GitHub Copilot

## Overview

This document contains detailed prompts to enhance the existing MAB Debiasing codebase (`https://github.com/DevDaring/Bandit_Debiasing`) for TACL publication. Each prompt is self-contained and can be given to Antigravity or GitHub Copilot to implement specific enhancements.

**Important Constraints:**
- Models: Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, Gemma-2-2B-IT (run one at a time)
- Languages: English, Hindi, Bengali (process one at a time to fit 24GB VRAM)
- Datasets: Multi-CrowS-Pairs, Indian-Multilingual-Bias-Dataset (from private HuggingFace repo)
- HuggingFace token available via `HF_TOKEN` environment variable
- CSV outputs must use full form column names (e.g., "English" not "en", "Gender Bias Score" not "gender_bias")

---

# PROMPT 1: Update Configuration for Multiple Models

```
## Task: Update Model Configuration for Multi-Model Support

Update the existing `config/model_config.py` to support three smaller models that can run on 24GB VRAM. The system should process one model at a time.

### Requirements:

1. Create a model registry in `config/model_config.py` with these models:
   - Qwen/Qwen2.5-1.5B-Instruct (1.5B parameters)
   - meta-llama/Llama-3.2-1B-Instruct (1B parameters)  
   - google/gemma-2-2b-it (2B parameters)

2. Each model entry should include:
   - model_name: HuggingFace model identifier
   - model_family: "Qwen", "Llama", or "Gemma"
   - parameters: parameter count as string (e.g., "1.5B")
   - hidden_size: model's hidden dimension for SAE training
   - num_layers: total transformer layers
   - recommended_sae_layer: middle layer index for SAE (approximately num_layers // 2)
   - quantization: "4bit" using bitsandbytes NF4
   - max_memory_mb: estimated VRAM usage
   - supports_flash_attention: boolean

3. Add a function `get_model_config(model_name: str)` that returns the configuration dict.

4. Add a function `get_all_models()` that returns list of all model names.

5. Add a function `get_model_hidden_size(model_name: str)` for SAE dimension configuration.

6. Update `config/bandit_config.py` to use model-specific context dimensions.

7. Ensure all configurations support sequential processing (one model at a time) to fit within 24GB VRAM.

### Example Usage:
```python
from config.model_config import get_model_config, get_all_models

for model_name in get_all_models():
    config = get_model_config(model_name)
    print(f"Processing {config['model_family']} with {config['parameters']} parameters")
```

### File Locations:
- Update: `config/model_config.py`
- Update: `config/bandit_config.py`
```

---

# PROMPT 2: Dataset Integration - Multi-CrowS-Pairs and IndiBias

```
## Task: Implement Dataset Loaders for Evaluation Datasets

Create a new module `src/data/dataset_loader.py` to load evaluation datasets from private HuggingFace repositories. The datasets are:
1. Multi-CrowS-Pairs (Debk/Multi-CrowS-Pairs) - 1422 sentence pairs across 9 bias categories
2. Indian-Multilingual-Bias-Dataset (Debk/Indian-Multilingual-Bias-Dataset) - 774 sentences per language across 4 India-specific bias categories

### Requirements:

1. Create `src/data/__init__.py` if not exists.

2. Create `src/data/dataset_loader.py` with the following classes:

#### Class: MultiCrowsPairsLoader
```python
class MultiCrowsPairsLoader:
    """
    Loader for Multi-CrowS-Pairs dataset.
    
    Dataset structure:
    - Languages: English, Hindi, Bengali
    - Bias categories: race-color, gender, socioeconomic, nationality, religion, 
                       age, sexual-orientation, physical-appearance, disability
    - Fields: Index, Target_Stereotypical, Target_Anti-Stereotypical, Sentence, 
              stereo_antistereo, bias_type, annotations, anon_writer, anon_annotators
    """
    
    def __init__(self, hf_token: str = None):
        """Initialize with HuggingFace token from env if not provided."""
        
    def load_language(self, language: str) -> pd.DataFrame:
        """
        Load dataset for specific language.
        Args:
            language: One of "English", "Hindi", "Bengali"
        Returns:
            DataFrame with all columns
        """
        
    def load_all_languages(self) -> Dict[str, pd.DataFrame]:
        """Load datasets for all three languages."""
        
    def get_bias_category_samples(self, language: str, bias_type: str) -> pd.DataFrame:
        """Filter samples by bias category."""
        
    def get_statistics(self, language: str) -> Dict:
        """Return dataset statistics including counts per bias category."""
```

#### Class: IndiBiasLoader
```python
class IndiBiasLoader:
    """
    Loader for Indian-Multilingual-Bias-Dataset.
    
    Dataset structure:
    - Languages: English, Hindi, Bengali
    - Bias categories: Caste, Gender, Religious, Race
    - Fields: Target_Stereotypical, Target_Anti-Stereotypical, Sentence
    """
    
    def __init__(self, hf_token: str = None):
        """Initialize with HuggingFace token from env if not provided."""
        
    def load_category(self, language: str, category: str) -> pd.DataFrame:
        """
        Load specific bias category for a language.
        Args:
            language: One of "English", "Hindi", "Bengali"
            category: One of "Caste", "Gender", "Religious", "Race"
        """
        
    def load_language(self, language: str) -> pd.DataFrame:
        """Load all categories for a language, combined."""
        
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all languages and categories."""
        
    def get_india_specific_samples(self, language: str) -> pd.DataFrame:
        """Return only India-specific categories: Caste and Religious."""
```

#### Class: UnifiedBiasDataset
```python
class UnifiedBiasDataset:
    """
    Unified interface for both datasets with consistent format.
    """
    
    def __init__(self, hf_token: str = None):
        self.crows_loader = MultiCrowsPairsLoader(hf_token)
        self.indibias_loader = IndiBiasLoader(hf_token)
        
    def get_evaluation_samples(self, language: str, dataset: str = "both") -> pd.DataFrame:
        """
        Get evaluation samples in unified format.
        
        Returns DataFrame with columns:
        - sentence: The test sentence with MASK token
        - target_stereotypical: List of stereotypical targets
        - target_anti_stereotypical: List of anti-stereotypical targets  
        - bias_category: Category name (full form, e.g., "Race and Color" not "race-color")
        - source_dataset: "Multi-CrowS-Pairs" or "Indian-Multilingual-Bias-Dataset"
        - language: "English", "Hindi", or "Bengali" (full form)
        """
        
    def create_train_val_test_split(
        self, 
        language: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = "bias_category"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified splits maintaining category balance."""
        
    def export_to_json(self, output_dir: str):
        """Export all splits to JSON format for training."""
```

3. Create `src/data/bias_categories.py` with category mappings:
```python
# Map short forms to full forms for CSV outputs
BIAS_CATEGORY_NAMES = {
    # Multi-CrowS-Pairs categories
    "race-color": "Race and Color",
    "gender": "Gender Identity",
    "socioeconomic": "Socioeconomic Status",
    "nationality": "National Origin",
    "religion": "Religious Belief",
    "age": "Age Group",
    "sexual-orientation": "Sexual Orientation",
    "physical-appearance": "Physical Appearance",
    "disability": "Disability Status",
    # IndiBias categories
    "Caste": "Caste System",
    "Gender": "Gender Identity", 
    "Religious": "Religious Belief",
    "Race": "Race and Ethnicity"
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi", 
    "bn": "Bengali"
}

MODEL_NAMES = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 2.5 (1.5B)",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama 3.2 (1B)",
    "google/gemma-2-2b-it": "Gemma 2 (2B)"
}
```

4. Use HuggingFace `datasets` library with authentication:
```python
from datasets import load_dataset
import os

hf_token = os.getenv("HF_TOKEN")
dataset = load_dataset("Debk/Multi-CrowS-Pairs", token=hf_token)
```

5. Handle the dataset file structure:
   - Multi-CrowS-Pairs: `English/crows_pair_english.csv`, `Hindi/crows_pair_hindi.csv`, `Bengali/crows_pair_bengali.csv`
   - IndiBias: `english/Caste.csv`, `bengali/Caste_Bengali.csv`, `hindi/Caste_Hindi.csv`, etc.

### File Locations:
- Create: `src/data/__init__.py`
- Create: `src/data/dataset_loader.py`
- Create: `src/data/bias_categories.py`
```

---

# PROMPT 3: Theoretical Framework - Regret Analysis Module

```
## Task: Implement Theoretical Regret Analysis Module

Create a new module `src/theory/` to implement fairness-constrained contextual bandit theory with regret bounds. This provides the theoretical foundation for Fair-CB.

### Requirements:

1. Create `src/theory/__init__.py`

2. Create `src/theory/regret_tracker.py`:

```python
class RegretTracker:
    """
    Track cumulative regret during bandit training.
    
    Regret Definition:
    R(T) = Σ_{t=1}^{T} [r(x_t, a*_t) - r(x_t, a_t)]
    
    where:
    - x_t is context at time t
    - a*_t is optimal arm for context x_t
    - a_t is selected arm
    - r(x,a) is reward function
    """
    
    def __init__(self, num_arms: int = 6):
        self.num_arms = num_arms
        self.cumulative_regret = 0.0
        self.regret_history = []
        self.optimal_rewards_history = []
        self.actual_rewards_history = []
        
    def compute_instantaneous_regret(
        self,
        context: np.ndarray,
        selected_arm: int,
        all_arm_rewards: np.ndarray
    ) -> float:
        """
        Compute regret for single decision.
        
        Args:
            context: Feature vector (not used directly, for logging)
            selected_arm: Arm that was selected
            all_arm_rewards: Rewards for all arms (if available)
            
        Returns:
            Instantaneous regret = max(all_rewards) - reward(selected_arm)
        """
        
    def update(
        self,
        selected_arm: int,
        reward: float,
        oracle_reward: float = None
    ):
        """Update cumulative regret with new observation."""
        
    def get_cumulative_regret(self) -> float:
        """Return current cumulative regret."""
        
    def get_average_regret(self) -> float:
        """Return average regret per step."""
        
    def get_regret_over_time(self) -> List[float]:
        """Return list of cumulative regret at each step."""
        
    def to_dict(self) -> Dict:
        """Export tracker state to dictionary."""
```

3. Create `src/theory/fairness_tracker.py`:

```python
class FairnessViolationTracker:
    """
    Track fairness constraint violations.
    
    Fairness Constraint: bias_score(output) ≤ τ (threshold)
    Violation: V(t) = max(0, bias_score_t - τ)
    Cumulative: V(T) = Σ_{t=1}^{T} V(t)
    """
    
    def __init__(self, threshold: float = 0.5, lambda_weight: float = 0.5):
        """
        Args:
            threshold: τ - maximum acceptable bias score
            lambda_weight: λ - Lagrangian multiplier for fairness penalty
        """
        self.threshold = threshold
        self.lambda_weight = lambda_weight
        self.violations = []
        self.bias_scores = []
        
    def compute_violation(self, bias_score: float) -> float:
        """Compute violation for single output."""
        return max(0.0, bias_score - self.threshold)
        
    def update(self, bias_score: float):
        """Record new bias score and compute violation."""
        
    def get_cumulative_violations(self) -> float:
        """Return V(T) = sum of all violations."""
        
    def get_weighted_violations(self) -> float:
        """Return λ * V(T) for fairness-aware regret."""
        
    def get_violation_rate(self) -> float:
        """Return proportion of steps with violations."""
        
    def to_dict(self) -> Dict:
        """Export tracker state."""
```

4. Create `src/theory/bounds.py`:

```python
class TheoreticalBoundComputer:
    """
    Compute theoretical regret bounds for Fair-CB.
    
    Theorem 1 (LinUCB Regret Bound):
    R(T) ≤ O(d√(KT log(T/δ)))
    
    Theorem 2 (Fairness-Aware Bound):
    R_fair(T) ≤ O(d√(KT log(T/δ)) + λ·E[V(T)])
    """
    
    def __init__(
        self,
        context_dim: int = 128,
        num_arms: int = 6,
        confidence_delta: float = 0.05
    ):
        self.d = context_dim
        self.K = num_arms
        self.delta = confidence_delta
        
    def compute_linucb_bound(self, T: int) -> float:
        """
        Compute theoretical LinUCB regret bound.
        
        R(T) ≤ d * √(K * T * log(T/δ))
        
        Args:
            T: Number of rounds
        Returns:
            Theoretical upper bound on cumulative regret
        """
        
    def compute_fairness_aware_bound(
        self,
        T: int,
        lambda_weight: float,
        expected_violations: float
    ) -> float:
        """
        Compute fairness-aware regret bound.
        
        R_fair(T) = R(T) + λ * E[V(T)]
        """
        
    def verify_bound_satisfaction(
        self,
        empirical_regret: float,
        T: int,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Verify if empirical regret satisfies theoretical bound.
        
        Returns:
            {
                "theoretical_bound": float,
                "empirical_regret": float,
                "bound_satisfied": bool,
                "margin": float,
                "confidence_level": float
            }
        """
```

5. Create `src/theory/adaptive_vs_static.py`:

```python
class AdaptiveStaticComparator:
    """
    Compare adaptive MAB selection against best static arm.
    
    Theorem 3: R_adaptive(T) / R_static(T) → 0 as T → ∞
    """
    
    def __init__(self, num_arms: int = 6):
        self.num_arms = num_arms
        self.arm_rewards = {i: [] for i in range(num_arms)}
        self.adaptive_rewards = []
        
    def record_arm_performance(
        self,
        arm: int,
        reward: float,
        was_selected: bool
    ):
        """Record reward for arm (whether selected or counterfactual)."""
        
    def compute_best_static_regret(self, T: int) -> float:
        """
        Compute regret of best fixed arm strategy.
        
        best_static_reward = max over arms (average reward for that arm)
        R_static = T * best_static_reward - sum(best_arm_rewards)
        """
        
    def compute_adaptive_static_ratio(self) -> float:
        """
        Compute R_adaptive / R_static.
        
        Should decrease over time, approaching 0.
        """
        
    def get_ratio_over_time(self, window: int = 100) -> List[float]:
        """Return ratio computed at regular intervals."""
        
    def identify_best_static_arm(self) -> int:
        """Return arm index with highest average reward."""
        
    def to_dict(self) -> Dict:
        """Export comparison results."""
```

6. Create `src/theory/theorem_verification.py`:

```python
class TheoremVerifier:
    """
    Numerical verification of theoretical claims.
    """
    
    def __init__(self, regret_tracker: RegretTracker, bound_computer: TheoreticalBoundComputer):
        self.regret_tracker = regret_tracker
        self.bound_computer = bound_computer
        
    def run_monte_carlo_verification(
        self,
        bandit,
        test_contexts: np.ndarray,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Run Monte Carlo simulations to verify bounds hold with high probability.
        
        Returns:
            {
                "bound_violation_rate": float,
                "empirical_regret_distribution": List[float],
                "confidence_interval_95": Tuple[float, float],
                "theorem_1_satisfied": bool,
                "theorem_2_satisfied": bool,
                "theorem_3_convergence_rate": str
            }
        """
        
    def generate_verification_report(self, output_path: str):
        """Generate comprehensive verification report as JSON."""
        
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper inclusion."""
```

### File Locations:
- Create: `src/theory/__init__.py`
- Create: `src/theory/regret_tracker.py`
- Create: `src/theory/fairness_tracker.py`
- Create: `src/theory/bounds.py`
- Create: `src/theory/adaptive_vs_static.py`
- Create: `src/theory/theorem_verification.py`
```

---

# PROMPT 4: Novel Metrics - IBR and FAR Implementation

```
## Task: Implement Novel Evaluation Metrics (IBR and FAR)

Create new metrics module `src/metrics/` with Intersectional Bias Reduction (IBR) and Fairness-Aware Regret (FAR) metrics.

### Requirements:

1. Create `src/metrics/__init__.py`

2. Create `src/metrics/ibr.py`:

```python
class IntersectionalBiasReduction:
    """
    Intersectional Bias Reduction (IBR) metric.
    
    IBR = HarmonicMean(BR_cat1, BR_cat2, ..., BR_catN)
    
    where BR_category = (baseline_bias - method_bias) / baseline_bias
    
    Why Harmonic Mean?
    - Penalizes methods that excel on one category but fail on others
    - A method scoring 0.8 on gender but 0.2 on caste gets IBR = 0.33, not 0.5
    """
    
    def __init__(self, categories: List[str] = None):
        """
        Args:
            categories: List of bias categories to include.
                       If None, use all available categories.
        """
        self.categories = categories or [
            "Gender Identity",
            "Race and Color", 
            "Religious Belief",
            "Caste System",
            "Socioeconomic Status",
            "Age Group",
            "Disability Status",
            "Sexual Orientation",
            "Physical Appearance",
            "National Origin"
        ]
        self.baseline_scores = {}
        self.method_scores = {}
        
    def set_baseline_scores(self, scores: Dict[str, float]):
        """
        Set baseline (no intervention) bias scores per category.
        
        Args:
            scores: {category_name: bias_score}
        """
        
    def compute_bias_reduction(
        self,
        method_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute bias reduction for each category.
        
        BR_category = (baseline - method) / baseline
        
        Returns:
            {category: bias_reduction_ratio}
        """
        
    def compute_ibr(self, method_scores: Dict[str, float]) -> float:
        """
        Compute IBR as harmonic mean of all bias reductions.
        
        IBR = N / Σ(1/BR_i) where N = number of categories
        
        Returns:
            IBR score in [0, 1] where higher is better
        """
        
    def compute_intersectional_ibr(
        self,
        method_scores: Dict[str, float],
        intersectional_categories: List[str] = None
    ) -> float:
        """
        Compute IBR focusing on intersectional categories.
        
        Default intersectional categories:
        - "Gender and Caste" (Gender × Caste)
        - "Religion and Gender" (Religion × Gender)
        """
        
    def get_category_rankings(
        self,
        method_scores: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """Return categories ranked by bias reduction (worst to best)."""
        
    def to_dict(self) -> Dict:
        """Export all metrics to dictionary for CSV/JSON."""
```

3. Create `src/metrics/far.py`:

```python
class FairnessAwareRegret:
    """
    Fairness-Aware Regret (FAR) metric.
    
    FAR(T) = Cumulative_Regret(T) + λ × Cumulative_Fairness_Violation(T)
    
    This combines standard bandit regret with fairness constraint violations.
    """
    
    def __init__(
        self,
        lambda_weight: float = 0.5,
        fairness_threshold: float = 0.5
    ):
        """
        Args:
            lambda_weight: Weight for fairness violations (λ)
            fairness_threshold: τ - maximum acceptable bias score
        """
        self.lambda_weight = lambda_weight
        self.fairness_threshold = fairness_threshold
        self.regret_history = []
        self.violation_history = []
        
    def update(
        self,
        instant_regret: float,
        bias_score: float,
        quality_score: float
    ):
        """
        Record metrics for one step.
        
        Args:
            instant_regret: r(optimal) - r(selected) for this step
            bias_score: Bias score of output
            quality_score: Quality score of output
        """
        
    def compute_far(self) -> float:
        """
        Compute FAR(T) = R(T) + λ*V(T)
        
        Returns:
            Fairness-aware cumulative regret
        """
        
    def compute_components(self) -> Dict[str, float]:
        """
        Return FAR broken down by components.
        
        Returns:
            {
                "cumulative_regret": float,
                "cumulative_violations": float,
                "lambda_weight": float,
                "fairness_penalty": float,  # λ * V(T)
                "far_total": float
            }
        """
        
    def get_far_over_time(self) -> List[float]:
        """Return FAR computed at each step."""
        
    def compare_methods(
        self,
        method_results: Dict[str, 'FairnessAwareRegret']
    ) -> pd.DataFrame:
        """
        Compare FAR across multiple methods.
        
        Returns DataFrame with columns:
        - Method Name
        - Cumulative Regret
        - Cumulative Violations  
        - Fairness Penalty
        - Fairness-Aware Regret (FAR)
        """
```

4. Create `src/metrics/comprehensive_evaluator.py`:

```python
class ComprehensiveEvaluator:
    """
    Unified evaluator combining all metrics for paper results.
    """
    
    def __init__(
        self,
        languages: List[str] = ["English", "Hindi", "Bengali"],
        models: List[str] = None
    ):
        self.languages = languages
        self.models = models or [
            "Qwen 2.5 (1.5B)",
            "Llama 3.2 (1B)",
            "Gemma 2 (2B)"
        ]
        self.results = {}
        
    def evaluate_method(
        self,
        method_name: str,
        predictions: Dict,
        baseline_predictions: Dict
    ) -> Dict:
        """
        Compute all metrics for a method.
        
        Returns:
            {
                "bias_score": float,
                "quality_score": float,
                "ibr": float,
                "far": float,
                "per_category_bias": Dict[str, float],
                "per_language_bias": Dict[str, float]
            }
        """
        
    def generate_main_results_table(self) -> pd.DataFrame:
        """
        Generate Table 1 for paper: Main Results.
        
        Columns (full names):
        - Method Name
        - CrowS-Pairs English Bias Score
        - CrowS-Pairs Hindi Bias Score  
        - CrowS-Pairs Bengali Bias Score
        - IndiBias English Bias Score
        - IndiBias Hindi Bias Score
        - IndiBias Bengali Bias Score
        - Intersectional Bias Reduction (IBR)
        - Fairness-Aware Regret (FAR)
        - Output Quality Score
        """
        
    def generate_per_category_table(self) -> pd.DataFrame:
        """
        Generate detailed per-category results.
        
        Columns:
        - Bias Category (full name)
        - Language
        - Baseline Bias Score
        - Fair-CB Bias Score
        - Bias Reduction Percentage
        - Statistical Significance (p-value)
        """
        
    def export_to_csv(self, output_dir: str):
        """
        Export all results to CSV files with full-form column names.
        
        Files:
        - main_results.csv
        - per_category_results.csv
        - per_language_results.csv
        - per_model_results.csv
        - ibr_breakdown.csv
        - far_analysis.csv
        """
```

### File Locations:
- Create: `src/metrics/__init__.py`
- Create: `src/metrics/ibr.py`
- Create: `src/metrics/far.py`
- Create: `src/metrics/comprehensive_evaluator.py`
```

---

# PROMPT 5: Cross-Lingual Transfer Analysis

```
## Task: Implement Cross-Lingual Bias Transfer Analysis Module

Create `src/crosslingual/` module to analyze how debiasing strategies transfer across English, Hindi, and Bengali.

### Requirements:

1. Create `src/crosslingual/__init__.py`

2. Create `src/crosslingual/transfer_analyzer.py`:

```python
class CrossLingualTransferAnalyzer:
    """
    Analyze cross-lingual transfer of debiasing strategies.
    
    Research Questions:
    1. Do steering vectors trained on English transfer to Hindi/Bengali?
    2. Does the bandit learn language-specific arm preferences?
    3. How does bias manifestation differ across languages?
    """
    
    def __init__(self, languages: List[str] = ["English", "Hindi", "Bengali"]):
        self.languages = languages
        self.transfer_results = {}
        
    def compute_steering_vector_transfer_efficacy(
        self,
        source_lang: str,
        target_lang: str,
        steering_type: str,  # "gender", "race", "religion"
        source_bias_reduction: float,
        target_bias_reduction: float
    ) -> float:
        """
        Compute transfer ratio for steering vectors.
        
        Transfer Ratio = target_reduction / source_reduction
        
        1.0 = perfect transfer
        <1.0 = reduced efficacy
        >1.0 = improved efficacy (rare)
        """
        
    def compute_all_transfer_efficacies(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Compute transfer matrix for all language pairs and strategies.
        
        Args:
            results: {
                "English": {"gender_sv": 0.45, "race_sv": 0.42, ...},
                "Hindi": {"gender_sv": 0.32, ...},
                ...
            }
            
        Returns DataFrame with columns:
        - Source Language
        - Target Language
        - Debiasing Strategy (full name)
        - Source Efficacy
        - Target Efficacy
        - Transfer Ratio
        """
        
    def compute_arm_selection_agreement(
        self,
        arm_selections: Dict[str, List[int]]
    ) -> Dict:
        """
        Compute agreement in arm selection across languages for parallel inputs.
        
        Args:
            arm_selections: {"English": [1,2,1,3,...], "Hindi": [1,3,1,2,...], ...}
            
        Returns:
            {
                "English-Hindi Agreement": 0.67,
                "English-Bengali Agreement": 0.62,
                "Hindi-Bengali Agreement": 0.74,
                "Per-Arm Agreement": {
                    "No Intervention": 0.82,
                    "Gender Steering Vector": 0.45,
                    ...
                }
            }
        """
        
    def analyze_language_specific_preferences(
        self,
        arm_selections_by_language: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """
        Compute P(arm | language) for each language.
        
        Returns DataFrame with columns:
        - Language (full name)
        - Arm Name (full name, e.g., "Gender Steering Vector")
        - Selection Probability
        - Standard Error
        - Is Statistically Significant (vs uniform)
        """
        
    def generate_transfer_report(self, output_path: str):
        """Generate comprehensive cross-lingual analysis report."""
```

3. Create `src/crosslingual/code_mixing_handler.py`:

```python
class CodeMixingDetector:
    """
    Detect and handle code-mixed inputs (Hindi-English, Bengali-English).
    """
    
    def __init__(self):
        # Use langdetect or similar library
        pass
        
    def detect_code_mixing(self, text: str) -> Dict:
        """
        Analyze text for code-mixing.
        
        Returns:
            {
                "is_code_mixed": bool,
                "primary_language": "Hindi" | "Bengali" | "English",
                "secondary_language": str | None,
                "mixing_ratio": float,  # 0 = monolingual, 1 = fully mixed
                "detected_languages": List[str]
            }
        """
        
    def analyze_arm_preference_by_mixing(
        self,
        samples: List[Dict],
        arm_selections: List[int]
    ) -> pd.DataFrame:
        """
        Analyze if arm preferences differ for code-mixed vs monolingual inputs.
        
        Hypothesis: Prompt-based interventions work better for code-mixed 
        inputs because steering vectors are trained on monolingual data.
        
        Returns DataFrame with columns:
        - Input Type (Monolingual/Code-Mixed)
        - Mixing Ratio Range
        - Preferred Arm
        - Arm Selection Distribution
        - Correlation with Mixing Ratio
        """
```

4. Create `src/crosslingual/parallel_evaluator.py`:

```python
class ParallelCorpusEvaluator:
    """
    Evaluate on parallel samples across languages.
    """
    
    def __init__(self, dataset_loader):
        self.loader = dataset_loader
        
    def create_parallel_evaluation_set(
        self,
        n_samples: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Create aligned samples across all three languages.
        
        Uses Multi-CrowS-Pairs which has parallel translations.
        """
        
    def evaluate_consistency(
        self,
        model,
        bandit,
        parallel_samples: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Run evaluation on parallel samples.
        
        Returns:
            {
                "per_language_bias": {"English": 0.32, "Hindi": 0.38, ...},
                "per_language_quality": {...},
                "arm_selection_consistency": float,
                "bias_reduction_consistency": float
            }
        """
        
    def generate_cross_lingual_report(
        self,
        results: Dict,
        output_path: str
    ):
        """Generate report with visualizations."""
```

### File Locations:
- Create: `src/crosslingual/__init__.py`
- Create: `src/crosslingual/transfer_analyzer.py`
- Create: `src/crosslingual/code_mixing_handler.py`
- Create: `src/crosslingual/parallel_evaluator.py`
```

---

# PROMPT 6: Ablation Study Framework

```
## Task: Implement Comprehensive Ablation Study Framework

Create `src/ablation/` module for rigorous component necessity analysis.

### Requirements:

1. Create `src/ablation/__init__.py`

2. Create `src/ablation/ablation_configs.py`:

```python
ABLATION_CONFIGURATIONS = {
    "Full System": {
        "description": "Complete Fair-CB system with all components",
        "disable_arms": [],
        "disable_features": [],
        "bandit_type": "linucb",
        "use_fairness_constraint": True,
        "use_cross_lingual": True
    },
    "Without Context Features": {
        "description": "Random context (no semantic features)",
        "disable_arms": [],
        "disable_features": ["all"],
        "use_random_context": True
    },
    "Without Steering Vectors": {
        "description": "Disable all steering vector arms",
        "disable_arms": [1, 2, 3],  # gender, race, religion SVs
        "disable_features": []
    },
    "Without Prompt Prefix": {
        "description": "Disable prompt-based debiasing",
        "disable_arms": [4],
        "disable_features": []
    },
    "Without Output Adjustment": {
        "description": "Disable output post-processing",
        "disable_arms": [5],
        "disable_features": []
    },
    "Only Steering Vectors": {
        "description": "Only steering vector arms enabled",
        "disable_arms": [0, 4, 5],
        "disable_features": []
    },
    "Only Prompt Prefix": {
        "description": "Only prompt-based intervention",
        "disable_arms": [0, 1, 2, 3, 5],
        "disable_features": []
    },
    "LinUCB Only": {
        "description": "LinUCB algorithm only",
        "bandit_type": "linucb",
        "disable_arms": []
    },
    "Thompson Sampling Only": {
        "description": "Thompson Sampling only",
        "bandit_type": "thompson",
        "disable_arms": []
    },
    "Neural Bandit Only": {
        "description": "Neural Bandit only",
        "bandit_type": "neural",
        "disable_arms": []
    },
    "Without Language Features": {
        "description": "Remove language detection features",
        "disable_features": ["language"],
        "disable_arms": []
    },
    "Without Demographic Features": {
        "description": "Remove demographic marker features",
        "disable_features": ["demographic"],
        "disable_arms": []
    },
    "Without Fairness Constraint": {
        "description": "Standard bandit without fairness penalty",
        "use_fairness_constraint": False,
        "disable_arms": []
    },
    "English Only Training": {
        "description": "Train only on English, test on all",
        "train_languages": ["English"],
        "test_languages": ["English", "Hindi", "Bengali"]
    },
    "Best Static Arm": {
        "description": "Always select best-performing fixed arm",
        "use_static_best": True,
        "disable_arms": []
    }
}

def get_ablation_config(name: str) -> Dict:
    """Get configuration for specific ablation."""
    return ABLATION_CONFIGURATIONS.get(name, {})

def get_all_ablation_names() -> List[str]:
    """Get list of all ablation configuration names."""
    return list(ABLATION_CONFIGURATIONS.keys())
```

3. Create `src/ablation/ablation_runner.py`:

```python
class AblationRunner:
    """
    Run comprehensive ablation studies.
    """
    
    def __init__(
        self,
        base_config: Dict,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_name: str,
        language: str
    ):
        self.base_config = base_config
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model_name = model_name
        self.language = language
        
    def run_single_ablation(
        self,
        ablation_name: str,
        seed: int = 42
    ) -> Dict:
        """
        Run training and evaluation with specific ablation configuration.
        
        Returns:
            {
                "ablation_name": str,
                "crows_pairs_bias": float,
                "indibias_bias": float,
                "ibr": float,
                "far": float,
                "quality": float,
                "training_time_seconds": float
            }
        """
        
    def run_all_ablations(
        self,
        num_seeds: int = 5,
        save_checkpoints: bool = False
    ) -> pd.DataFrame:
        """
        Run all ablations with multiple random seeds.
        
        Returns DataFrame with columns:
        - Ablation Configuration
        - Configuration Description
        - CrowS-Pairs Bias Score (Mean)
        - CrowS-Pairs Bias Score (Standard Deviation)
        - CrowS-Pairs Bias Score (95% Confidence Interval Lower)
        - CrowS-Pairs Bias Score (95% Confidence Interval Upper)
        - IndiBias Score (Mean)
        - IndiBias Score (Standard Deviation)
        - Intersectional Bias Reduction (Mean)
        - Fairness-Aware Regret (Mean)
        - Quality Score (Mean)
        - p-value vs Full System
        - Effect Size (Cohen's d)
        - Is Statistically Significant
        """
        
    def compute_statistical_significance(
        self,
        full_system_results: List[float],
        ablation_results: List[float]
    ) -> Dict:
        """
        Compute significance tests.
        
        Returns:
            {
                "t_statistic": float,
                "p_value": float,
                "cohens_d": float,
                "is_significant": bool  # p < 0.05
            }
        """
```

4. Create `src/ablation/component_analyzer.py`:

```python
class ComponentNecessityAnalyzer:
    """
    Analyze which components are necessary vs redundant.
    """
    
    def __init__(self, ablation_results: pd.DataFrame):
        self.results = ablation_results
        
    def rank_components_by_importance(self) -> pd.DataFrame:
        """
        Rank components by performance drop when removed.
        
        Returns DataFrame with columns:
        - Component Name
        - Performance Drop (Bias Score Increase)
        - Performance Drop Percentage
        - Importance Rank
        - Is Critical (p < 0.05 and drop > 10%)
        """
        
    def identify_redundant_components(
        self,
        significance_threshold: float = 0.05,
        effect_threshold: float = 0.1
    ) -> List[str]:
        """
        Find components that can be removed without significant impact.
        
        Returns list of component names that are not critical.
        """
        
    def analyze_component_interactions(self) -> pd.DataFrame:
        """
        Test interaction effects between components.
        
        Returns DataFrame showing if removing A hurts more when B is also removed.
        """
        
    def generate_ablation_report(self, output_dir: str):
        """
        Generate comprehensive ablation report.
        
        Creates:
        - ablation_results.csv
        - component_importance.csv
        - interaction_effects.csv
        - ablation_summary.md
        """
```

5. Create `src/ablation/sensitivity_analyzer.py`:

```python
class SensitivityAnalyzer:
    """
    Analyze sensitivity to hyperparameters.
    """
    
    def analyze_reward_weight_sensitivity(
        self,
        alpha_values: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ) -> pd.DataFrame:
        """
        Vary α (bias weight in reward) and measure performance.
        
        Reward = α * (1 - bias_score) + (1-α) * quality_score
        
        Returns DataFrame with columns:
        - Bias Weight (Alpha)
        - Quality Weight (1 - Alpha)
        - Bias Score
        - Quality Score
        - Combined Reward
        - IBR
        - FAR
        """
        
    def analyze_fairness_threshold_sensitivity(
        self,
        tau_values: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
    ) -> pd.DataFrame:
        """
        Vary fairness threshold τ and measure violations.
        
        Returns DataFrame showing tradeoff between strictness and performance.
        """
        
    def analyze_exploration_sensitivity(
        self,
        alpha_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0]
    ) -> pd.DataFrame:
        """
        Vary LinUCB exploration parameter.
        """
        
    def find_optimal_hyperparameters(self) -> Dict:
        """
        Identify optimal hyperparameter settings.
        
        Returns:
            {
                "optimal_bias_weight": float,
                "optimal_fairness_threshold": float,
                "optimal_exploration": float,
                "performance_at_optimal": Dict
            }
        """
```

### File Locations:
- Create: `src/ablation/__init__.py`
- Create: `src/ablation/ablation_configs.py`
- Create: `src/ablation/ablation_runner.py`
- Create: `src/ablation/component_analyzer.py`
- Create: `src/ablation/sensitivity_analyzer.py`
```

---

# PROMPT 7: Update Main Pipeline for Sequential Processing

```
## Task: Update Main Pipeline for Memory-Efficient Sequential Processing

Update the existing pipeline to process one model and one language at a time to fit within 24GB VRAM.

### Requirements:

1. Update `src/pipeline/training_pipeline.py`:

```python
class SequentialTrainingPipeline:
    """
    Memory-efficient pipeline that processes one model and one language at a time.
    """
    
    def __init__(
        self,
        models: List[str] = None,
        languages: List[str] = None,
        output_dir: str = "./results"
    ):
        """
        Args:
            models: List of model names to process sequentially
            languages: List of languages to process sequentially
            output_dir: Directory for results
        """
        self.models = models or [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "google/gemma-2-2b-it"
        ]
        self.languages = languages or ["English", "Hindi", "Bengali"]
        self.output_dir = output_dir
        
    def clear_gpu_memory(self):
        """
        Aggressively clear GPU memory between model/language switches.
        """
        import torch
        import gc
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def run_single_experiment(
        self,
        model_name: str,
        language: str,
        bandit_type: str = "linucb"
    ) -> Dict:
        """
        Run complete experiment for one model-language combination.
        
        Steps:
        1. Load model
        2. Load language-specific data
        3. Train bandit
        4. Evaluate on both datasets
        5. Compute all metrics
        6. Clear memory
        
        Returns results dictionary with all metrics.
        """
        
    def run_all_experiments(
        self,
        bandit_types: List[str] = ["linucb", "thompson"],
        n_epochs: int = 3
    ) -> pd.DataFrame:
        """
        Run experiments for all model-language-bandit combinations.
        
        Total combinations: 3 models × 3 languages × 2 bandits = 18 experiments
        
        Returns DataFrame with columns using full names:
        - Model Name (e.g., "Qwen 2.5 (1.5B)")
        - Language
        - Bandit Algorithm
        - CrowS-Pairs Bias Score
        - IndiBias Bias Score
        - India-Specific Bias Score (Caste + Religious)
        - Intersectional Bias Reduction (IBR)
        - Fairness-Aware Regret (FAR)
        - Output Quality Score
        - Training Time (Minutes)
        - GPU Memory Peak (GB)
        """
        
    def save_checkpoint(
        self,
        model_name: str,
        language: str,
        bandit,
        metrics: Dict
    ):
        """Save checkpoint after each experiment."""
        
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume interrupted experiment run."""
```

2. Update `scripts/run_experiment.py`:

```python
#!/usr/bin/env python3
"""
Main experiment script for Fair-CB evaluation.

Usage:
    # Run single model-language combination
    python scripts/run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct --language English
    
    # Run all experiments sequentially
    python scripts/run_experiment.py --run_all
    
    # Resume from checkpoint
    python scripts/run_experiment.py --resume checkpoint_path
"""

import argparse
import os
from src.pipeline.training_pipeline import SequentialTrainingPipeline
from src.data.dataset_loader import UnifiedBiasDataset

def main():
    parser = argparse.ArgumentParser(description="Fair-CB Experiment Runner")
    
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--language", type=str, help="Language to process")
    parser.add_argument("--bandit_type", type=str, default="linucb")
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--run_all", action="store_true", help="Run all combinations")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--hf_token", type=str, default=None)
    
    args = parser.parse_args()
    
    # Get HF token from env if not provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Initialize dataset loader
    dataset = UnifiedBiasDataset(hf_token=hf_token)
    
    # Initialize pipeline
    pipeline = SequentialTrainingPipeline(output_dir=args.output_dir)
    
    if args.run_all:
        # Run all model-language combinations
        results = pipeline.run_all_experiments(n_epochs=args.n_epochs)
        results.to_csv(f"{args.output_dir}/complete_results.csv", index=False)
        print(f"Results saved to {args.output_dir}/complete_results.csv")
        
    elif args.resume:
        # Resume from checkpoint
        pipeline.resume_from_checkpoint(args.resume)
        
    else:
        # Run single experiment
        if not args.model or not args.language:
            parser.error("--model and --language required unless --run_all")
            
        results = pipeline.run_single_experiment(
            model_name=args.model,
            language=args.language,
            bandit_type=args.bandit_type
        )
        print(f"Results: {results}")

if __name__ == "__main__":
    main()
```

3. Create `scripts/generate_all_results.py`:

```python
#!/usr/bin/env python3
"""
Generate all CSV result files for paper.

Creates:
- results/tables/main_results.csv
- results/tables/per_category_results.csv
- results/tables/cross_lingual_transfer.csv
- results/tables/ablation_results.csv
- results/tables/theory_verification.csv
"""

import os
import pandas as pd
from src.metrics.comprehensive_evaluator import ComprehensiveEvaluator
from src.crosslingual.transfer_analyzer import CrossLingualTransferAnalyzer
from src.ablation.ablation_runner import AblationRunner
from src.theory.theorem_verification import TheoremVerifier

def generate_main_results_table(experiment_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 1: Main Results comparing all methods.
    
    Columns (must use full names):
    - Method Name
    - Model Name
    - CrowS-Pairs English Bias Score
    - CrowS-Pairs Hindi Bias Score
    - CrowS-Pairs Bengali Bias Score
    - IndiBias English Bias Score
    - IndiBias Hindi Bias Score
    - IndiBias Bengali Bias Score
    - Intersectional Bias Reduction (IBR)
    - Fairness-Aware Regret (FAR)
    - Output Quality Score
    """
    pass

def generate_per_category_table(experiment_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 2: Per-Category Bias Results.
    
    Columns:
    - Bias Category (full name, e.g., "Gender Identity" not "gender")
    - Language
    - Model Name
    - Baseline Bias Score
    - Fair-CB Bias Score
    - Bias Reduction Percentage
    - Statistical Significance (p-value)
    - Effect Size (Cohen's d)
    """
    pass

def generate_cross_lingual_table(transfer_results: Dict) -> pd.DataFrame:
    """
    Generate Table 3: Cross-Lingual Transfer Analysis.
    
    Columns:
    - Source Language
    - Target Language
    - Debiasing Strategy (full name)
    - Source Language Efficacy
    - Target Language Efficacy
    - Transfer Ratio
    - Transfer Quality (Poor/Moderate/Good)
    """
    pass

def generate_ablation_table(ablation_results: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Table 4: Ablation Study Results.
    
    Columns:
    - Configuration Name
    - Configuration Description
    - CrowS-Pairs Bias Score
    - IndiBias Bias Score
    - Intersectional Bias Reduction (IBR)
    - p-value vs Full System
    - Effect Size (Cohen's d)
    - Is Component Critical
    """
    pass

def generate_theory_table(verification_results: Dict) -> pd.DataFrame:
    """
    Generate Table 5: Theoretical Verification.
    
    Columns:
    - Theorem
    - Theoretical Bound Value
    - Empirical Value
    - Bound Satisfied
    - Confidence Level
    - Verification Method
    """
    pass

def main():
    output_dir = "results/tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment results
    experiment_results = pd.read_csv("results/complete_results.csv")
    
    # Generate all tables
    main_table = generate_main_results_table(experiment_results)
    main_table.to_csv(f"{output_dir}/main_results.csv", index=False)
    
    category_table = generate_per_category_table(experiment_results)
    category_table.to_csv(f"{output_dir}/per_category_results.csv", index=False)
    
    # ... generate other tables
    
    print(f"All result tables saved to {output_dir}/")

if __name__ == "__main__":
    main()
```

### File Locations:
- Update: `src/pipeline/training_pipeline.py`
- Update: `scripts/run_experiment.py`
- Create: `scripts/generate_all_results.py`
```

---

# PROMPT 8: CSV Output Standardization

```
## Task: Create CSV Output Manager with Full-Form Column Names

Create a centralized module to ensure all CSV outputs use consistent, full-form column names.

### Requirements:

1. Create `src/output/csv_manager.py`:

```python
"""
CSV Output Manager for Fair-CB Results.

CRITICAL: All column names must use full forms, not abbreviations.
Examples:
- "English" not "en"
- "Gender Identity" not "gender"
- "Intersectional Bias Reduction (IBR)" not "ibr"
- "Qwen 2.5 (1.5B Parameters)" not "qwen2.5-1.5b"
"""

# Column name mappings
COLUMN_NAMES = {
    # Languages
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    
    # Models
    "qwen": "Qwen 2.5 (1.5B Parameters)",
    "llama": "Llama 3.2 (1B Parameters)",
    "gemma": "Gemma 2 (2B Parameters)",
    
    # Bias categories
    "gender": "Gender Identity",
    "race": "Race and Ethnicity",
    "race-color": "Race and Color",
    "religion": "Religious Belief",
    "caste": "Caste System",
    "socioeconomic": "Socioeconomic Status",
    "age": "Age Group",
    "disability": "Disability Status",
    "sexual-orientation": "Sexual Orientation",
    "physical-appearance": "Physical Appearance",
    "nationality": "National Origin",
    
    # Metrics
    "ibr": "Intersectional Bias Reduction (IBR)",
    "far": "Fairness-Aware Regret (FAR)",
    "bias_score": "Bias Score",
    "quality_score": "Output Quality Score",
    
    # Debiasing arms
    "arm_0": "No Intervention (Baseline)",
    "arm_1": "Gender Steering Vector",
    "arm_2": "Race Steering Vector",
    "arm_3": "Religion Steering Vector",
    "arm_4": "Prompt Prefix Debiasing",
    "arm_5": "Output Adjustment",
    
    # Bandit types
    "linucb": "Linear Upper Confidence Bound (LinUCB)",
    "thompson": "Thompson Sampling",
    "neural": "Neural Contextual Bandit",
    
    # Statistical columns
    "mean": "Mean Value",
    "std": "Standard Deviation",
    "ci_lower": "95% Confidence Interval (Lower)",
    "ci_upper": "95% Confidence Interval (Upper)",
    "p_value": "Statistical Significance (p-value)",
    "cohens_d": "Effect Size (Cohen's d)",
    "is_significant": "Is Statistically Significant (p < 0.05)"
}

class CSVOutputManager:
    """
    Centralized manager for CSV output formatting.
    """
    
    def __init__(self, output_dir: str = "./results/tables"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def format_column_name(self, short_name: str) -> str:
        """Convert short form to full form column name."""
        return COLUMN_NAMES.get(short_name, short_name)
        
    def format_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all column names in DataFrame to full forms.
        """
        new_columns = {}
        for col in df.columns:
            # Handle compound column names like "bias_score_en"
            parts = col.split("_")
            formatted_parts = [self.format_column_name(p) for p in parts]
            new_columns[col] = " - ".join(formatted_parts) if len(formatted_parts) > 1 else formatted_parts[0]
        
        return df.rename(columns=new_columns)
        
    def save_main_results(self, results: pd.DataFrame, filename: str = "main_results.csv"):
        """
        Save main results table with proper formatting.
        
        Expected columns after formatting:
        - Method Name
        - Model Name
        - Language
        - CrowS-Pairs Bias Score
        - IndiBias Bias Score
        - India-Specific Bias Score
        - Intersectional Bias Reduction (IBR)
        - Fairness-Aware Regret (FAR)
        - Output Quality Score
        """
        formatted_df = self.format_dataframe_columns(results)
        
        # Ensure numeric precision
        numeric_cols = formatted_df.select_dtypes(include=['float64']).columns
        for col in numeric_cols:
            formatted_df[col] = formatted_df[col].round(4)
            
        formatted_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        print(f"Saved: {self.output_dir}/{filename}")
        
    def save_per_category_results(
        self,
        results: pd.DataFrame,
        filename: str = "per_category_results.csv"
    ):
        """
        Save per-category breakdown.
        
        Expected columns:
        - Bias Category (full name)
        - Language
        - Model Name
        - Baseline Bias Score
        - Fair-CB Bias Score
        - Bias Reduction Percentage
        - Statistical Significance (p-value)
        """
        formatted_df = self.format_dataframe_columns(results)
        formatted_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
    def save_cross_lingual_results(
        self,
        results: pd.DataFrame,
        filename: str = "cross_lingual_transfer.csv"
    ):
        """
        Save cross-lingual transfer analysis.
        
        Expected columns:
        - Source Language
        - Target Language
        - Debiasing Strategy
        - Source Efficacy
        - Target Efficacy
        - Transfer Ratio
        """
        formatted_df = self.format_dataframe_columns(results)
        formatted_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
    def save_ablation_results(
        self,
        results: pd.DataFrame,
        filename: str = "ablation_results.csv"
    ):
        """
        Save ablation study results.
        """
        formatted_df = self.format_dataframe_columns(results)
        formatted_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
    def save_theory_verification(
        self,
        results: pd.DataFrame,
        filename: str = "theory_verification.csv"
    ):
        """
        Save theoretical verification results.
        """
        formatted_df = self.format_dataframe_columns(results)
        formatted_df.to_csv(f"{self.output_dir}/{filename}", index=False)
        
    def generate_summary_statistics(self, all_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics across all experiments.
        
        Returns DataFrame with:
        - Metric Name
        - Overall Mean
        - Per-Model Breakdown
        - Per-Language Breakdown
        - Best Configuration
        """
        pass
```

2. Update all existing code that saves CSV files to use CSVOutputManager.

3. Create validation function to check column names:

```python
def validate_csv_columns(filepath: str) -> List[str]:
    """
    Validate that CSV file uses full-form column names.
    
    Returns list of columns that violate naming convention.
    """
    df = pd.read_csv(filepath)
    violations = []
    
    short_forms = ["en", "hi", "bn", "ibr", "far", "sv", "ucb"]
    
    for col in df.columns:
        col_lower = col.lower()
        for short in short_forms:
            if short == col_lower or f"_{short}" in col_lower or f"{short}_" in col_lower:
                violations.append(col)
                break
                
    return violations
```

### File Locations:
- Create: `src/output/__init__.py`
- Create: `src/output/csv_manager.py`
```

---

# PROMPT 9: Evaluation Script Updates

```
## Task: Update Evaluation Scripts for Both Datasets

Update `scripts/evaluate_system.py` to evaluate on both Multi-CrowS-Pairs and IndiBias datasets with all new metrics.

### Requirements:

1. Update `scripts/evaluate_system.py`:

```python
#!/usr/bin/env python3
"""
Comprehensive evaluation script for Fair-CB system.

Evaluates on:
1. Multi-CrowS-Pairs dataset (9 bias categories)
2. Indian-Multilingual-Bias-Dataset (4 India-specific categories)

Computes:
- Standard bias scores
- IBR (Intersectional Bias Reduction)
- FAR (Fairness-Aware Regret)
- Per-category breakdown
- Cross-lingual analysis

Usage:
    python scripts/evaluate_system.py \
        --checkpoint checkpoints/bandit_linucb_final.pkl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --language English \
        --output_dir results/evaluation
"""

import argparse
import os
import json
import pandas as pd
from typing import Dict, List
from tqdm import tqdm

from src.data.dataset_loader import UnifiedBiasDataset
from src.metrics.ibr import IntersectionalBiasReduction
from src.metrics.far import FairnessAwareRegret
from src.metrics.comprehensive_evaluator import ComprehensiveEvaluator
from src.output.csv_manager import CSVOutputManager
from config.model_config import get_model_config

def evaluate_on_crows_pairs(
    model,
    bandit,
    dataset_loader,
    language: str
) -> Dict:
    """
    Evaluate on Multi-CrowS-Pairs dataset.
    
    Returns:
        {
            "overall_bias_score": float,
            "per_category_scores": {
                "Race and Color": float,
                "Gender Identity": float,
                ...
            },
            "num_samples": int,
            "stereotypical_preference_rate": float
        }
    """
    pass

def evaluate_on_indibias(
    model,
    bandit,
    dataset_loader,
    language: str
) -> Dict:
    """
    Evaluate on Indian-Multilingual-Bias-Dataset.
    
    Returns:
        {
            "overall_bias_score": float,
            "per_category_scores": {
                "Caste System": float,
                "Gender Identity": float,
                "Religious Belief": float,
                "Race and Ethnicity": float
            },
            "india_specific_score": float,  # Average of Caste + Religious
            "num_samples": int
        }
    """
    pass

def compute_all_metrics(
    crows_results: Dict,
    indibias_results: Dict,
    baseline_crows: Dict,
    baseline_indibias: Dict
) -> Dict:
    """
    Compute IBR, FAR, and other aggregate metrics.
    
    Returns:
        {
            "ibr": float,
            "far": float,
            "crows_pairs_bias": float,
            "indibias_bias": float,
            "quality_score": float,
            "per_category_ibr": Dict[str, float]
        }
    """
    pass

def compare_with_baselines(
    fair_cb_results: Dict,
    baseline_results: Dict,
    static_arm_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Generate comparison table with baselines.
    
    Baselines:
    - No Intervention
    - Gender Steering Vector Only
    - Race Steering Vector Only
    - Religion Steering Vector Only
    - Prompt Prefix Only
    - Best Static Arm
    - Fair-CB (Ours)
    
    Returns DataFrame with full-form column names.
    """
    pass

def main():
    parser = argparse.ArgumentParser(description="Fair-CB Evaluation")
    
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--bandit_type", type=str, default="linucb")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--compare_baselines", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    
    args = parser.parse_args()
    
    # Initialize
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    dataset_loader = UnifiedBiasDataset(hf_token=hf_token)
    csv_manager = CSVOutputManager(args.output_dir)
    
    # Load model and bandit
    model_config = get_model_config(args.model)
    # ... load model and bandit
    
    # Evaluate on both datasets
    print(f"Evaluating on Multi-CrowS-Pairs ({args.language})...")
    crows_results = evaluate_on_crows_pairs(model, bandit, dataset_loader, args.language)
    
    print(f"Evaluating on IndiBias ({args.language})...")
    indibias_results = evaluate_on_indibias(model, bandit, dataset_loader, args.language)
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(crows_results, indibias_results, baseline_crows, baseline_indibias)
    
    # Save results
    results_df = pd.DataFrame([{
        "Model Name": model_config["model_family"],
        "Language": args.language,
        "CrowS-Pairs Bias Score": crows_results["overall_bias_score"],
        "IndiBias Bias Score": indibias_results["overall_bias_score"],
        "India-Specific Bias Score": indibias_results["india_specific_score"],
        "Intersectional Bias Reduction (IBR)": metrics["ibr"],
        "Fairness-Aware Regret (FAR)": metrics["far"],
        "Output Quality Score": metrics["quality_score"]
    }])
    
    csv_manager.save_main_results(results_df, f"evaluation_{args.model.split('/')[-1]}_{args.language}.csv")
    
    # Per-category results
    category_df = pd.DataFrame([
        {
            "Bias Category": cat,
            "Language": args.language,
            "Bias Score": score,
            "Dataset": "Multi-CrowS-Pairs"
        }
        for cat, score in crows_results["per_category_scores"].items()
    ] + [
        {
            "Bias Category": cat,
            "Language": args.language,
            "Bias Score": score,
            "Dataset": "Indian-Multilingual-Bias-Dataset"
        }
        for cat, score in indibias_results["per_category_scores"].items()
    ])
    
    csv_manager.save_per_category_results(category_df, f"per_category_{args.model.split('/')[-1]}_{args.language}.csv")
    
    if args.compare_baselines:
        print("Comparing with baselines...")
        comparison_df = compare_with_baselines(metrics, baseline_results, static_arm_results)
        csv_manager.save_main_results(comparison_df, f"baseline_comparison_{args.language}.csv")
    
    print(f"Results saved to {args.output_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {model_config['model_family']}")
    print(f"Language: {args.language}")
    print(f"CrowS-Pairs Bias Score: {crows_results['overall_bias_score']:.4f}")
    print(f"IndiBias Bias Score: {indibias_results['overall_bias_score']:.4f}")
    print(f"IBR: {metrics['ibr']:.4f}")
    print(f"FAR: {metrics['far']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
```

### File Locations:
- Update: `scripts/evaluate_system.py`
```

---

# PROMPT 10: Complete Requirements and Setup

```
## Task: Update Requirements and Project Setup

Update requirements.txt and setup files for new dependencies.

### Requirements:

1. Update `requirements.txt`:

```
# Core ML
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
safetensors>=0.4.0

# HuggingFace
datasets>=2.14.0
huggingface_hub>=0.19.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Language detection (for code-mixing)
langdetect>=1.0.9
fasttext>=0.9.2

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Experiment tracking
wandb>=0.15.0
tqdm>=4.65.0

# Statistics
statsmodels>=0.14.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.0
```

2. Create `.env.example`:

```
# HuggingFace token for private datasets
HF_TOKEN=your_huggingface_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=fair-cb-debiasing

# CUDA settings
CUDA_VISIBLE_DEVICES=0

# Memory settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

3. Update `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="fair-cb-debiasing",
    version="2.0.0",
    description="Fair-CB: Fairness-Constrained Contextual Bandits for Adaptive Multilingual LLM Debiasing",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "langdetect>=1.0.9",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "statsmodels>=0.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fair-cb-train=scripts.run_experiment:main",
            "fair-cb-eval=scripts.evaluate_system:main",
        ],
    },
)
```

4. Create `README_ENHANCED.md` with updated documentation:

```markdown
# Fair-CB: Fairness-Constrained Contextual Bandits for Adaptive Multilingual LLM Debiasing

## Overview

Fair-CB is a theoretically-grounded framework for adaptive debiasing of multilingual Large Language Models. It uses contextual bandits with fairness constraints to dynamically select optimal debiasing interventions.

## Key Features

- **Theoretical Foundation**: Provable regret bounds under fairness constraints
- **Multiple Models**: Supports Qwen, Llama, and Gemma model families
- **Multilingual**: English, Hindi, and Bengali support
- **Novel Metrics**: Intersectional Bias Reduction (IBR) and Fairness-Aware Regret (FAR)
- **Comprehensive Evaluation**: Multi-CrowS-Pairs and IndiBias datasets

## Quick Start

\`\`\`bash
# Install dependencies
pip install -e .

# Set HuggingFace token
export HF_TOKEN=your_token_here

# Run experiment for single model-language
python scripts/run_experiment.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --language English \
    --n_epochs 3

# Run all experiments
python scripts/run_experiment.py --run_all

# Generate result tables
python scripts/generate_all_results.py
\`\`\`

## Models Supported

| Model | Parameters | Family |
|-------|------------|--------|
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Qwen |
| meta-llama/Llama-3.2-1B-Instruct | 1B | Llama |
| google/gemma-2-2b-it | 2B | Gemma |

## Datasets

1. **Multi-CrowS-Pairs**: 1422 sentence pairs, 9 bias categories, 3 languages
2. **Indian-Multilingual-Bias-Dataset**: 774 sentences, 4 India-specific categories, 3 languages

## Results

All CSV outputs use full-form column names. Results are saved to `results/tables/`:

- `main_results.csv`: Primary comparison table
- `per_category_results.csv`: Per-bias-category breakdown
- `cross_lingual_transfer.csv`: Transfer analysis
- `ablation_results.csv`: Ablation study
- `theory_verification.csv`: Theoretical verification

## Citation

\`\`\`bibtex
@article{fair_cb_2025,
    title={Fair-CB: Fairness-Constrained Contextual Bandits for Adaptive Multilingual LLM Debiasing},
    author={...},
    journal={Transactions of the ACL},
    year={2025}
}
\`\`\`
```

### File Locations:
- Update: `requirements.txt`
- Create: `.env.example`
- Update: `setup.py`
- Create: `README_ENHANCED.md`
```

---

# Summary Checklist

After implementing all prompts, verify:

- [ ] All models run sequentially (one at a time) to fit 24GB VRAM
- [ ] All languages processed sequentially (one at a time)
- [ ] Both datasets (Multi-CrowS-Pairs, IndiBias) are loaded from HuggingFace
- [ ] HF_TOKEN environment variable is used for authentication
- [ ] All CSV column names use full forms (no abbreviations)
- [ ] IBR and FAR metrics are computed and saved
- [ ] Ablation study covers all configurations
- [ ] Cross-lingual transfer analysis is complete
- [ ] Theoretical verification produces verifiable bounds
- [ ] All results are reproducible with random seeds

---

# Execution Order

1. **Prompt 1**: Update model configuration
2. **Prompt 2**: Implement dataset loaders
3. **Prompt 10**: Update requirements and setup
4. **Prompt 3**: Implement theory module
5. **Prompt 4**: Implement metrics (IBR, FAR)
6. **Prompt 8**: Create CSV output manager
7. **Prompt 5**: Implement cross-lingual analysis
8. **Prompt 6**: Implement ablation framework
9. **Prompt 7**: Update main pipeline
10. **Prompt 9**: Update evaluation scripts

This order ensures dependencies are available before they're needed.
