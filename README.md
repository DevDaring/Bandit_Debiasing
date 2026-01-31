# Fair-CB: Fairness-Aware Contextual Bandits for Multilingual LLM Debiasing

<div align="center">

**Adaptive Multi-Armed Bandit Debiasing Strategy Selection for Multilingual Large Language Models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## 🎯 Overview

Fair-CB is a research framework that dynamically selects optimal debiasing interventions for multilingual LLMs using contextual bandit algorithms. The system learns from fairness and quality feedback signals to adaptively choose the best debiasing strategy for each input.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | Qwen 2.5 7B, Aya Expanse 8B, Llama 3.1 8B |
| **Multilingual** | English, Hindi, Bengali + code-mixing detection |
| **Novel Metrics** | IBR (Intersectional Bias Reduction), FAR (Fairness-Aware Regret) |
| **Theoretical Guarantees** | Sublinear regret bounds with proof verification |
| **Publication-Ready** | LaTeX table generation, standardized CSV output |

### Debiasing Arms (6 Strategies)

| Arm | Strategy | Description |
|-----|----------|-------------|
| 0 | No Intervention | Baseline (no debiasing) |
| 1 | Gender Steering | Steering vector for gender bias |
| 2 | Race Steering | Steering vector for race/ethnicity bias |
| 3 | Religion Steering | Steering vector for religious bias |
| 4 | Prompt Prefix | Fairness-aware prompt modification |
| 5 | Output Adjustment | Post-hoc output debiasing |

---

## 🚀 Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/Fair-CB.git
cd Fair-CB
pip install -r requirements.txt
python setup.py develop

# Create directories
mkdir -p logs results checkpoints data/steering_vectors data/bias_evaluation_sets

# Configure environment
cp .env.example .env
# Edit .env to add your HF_TOKEN
```

### Run Experiment

```bash
# Quick test (subset of data)
python scripts/generate_all_results.py --quick

# Full TACL experiment suite
python scripts/generate_all_results.py

# Single model/dataset run
python scripts/run_experiment.py --model qwen --dataset multi_crows --epochs 3
```

### Evaluate with Novel Metrics

```bash
python scripts/evaluate_with_metrics.py \
    --dataset both \
    --generate-latex \
    --output-csv ./results
```

---

## 📊 Novel Metrics

### IBR (Intersectional Bias Reduction)

Measures bias reduction across ALL categories using **harmonic mean** (penalizes methods that fail in any category):

```
IBR = HarmonicMean({reduction_gender, reduction_race, reduction_religion, ...})
```

- **Range**: [0, 1] where 1 = perfect reduction across all categories
- **Why harmonic mean**: Penalizes methods that underperform in any single category

### FAR (Fairness-Aware Regret)

Combines regret and fairness violations:

```
FAR = R(T) + λ·V(T)
```

Where:
- `R(T)` = Cumulative regret at time T
- `V(T)` = Cumulative fairness violations
- `λ` = Fairness weight (default: 0.5)

---

## 📁 Project Structure

```
Fair-CB/
├── config/                          # Configuration
│   ├── model_config.py              # Model registry (Qwen, Aya, Llama)
│   ├── bandit_config.py             # Bandit hyperparameters
│   └── steering_vectors.py          # Steering vector paths
│
├── src/                             # Source code
│   ├── bandit/                      # Bandit algorithms (LinUCB, Thompson, Neural)
│   ├── context_extractor/           # 128-dim feature extraction
│   ├── debiasing_arms/              # 6 debiasing strategies
│   ├── llm/                         # Model loading and generation
│   ├── reward/                      # Bias and quality scoring
│   ├── pipeline/                    # Training and inference pipelines
│   │   ├── training_pipeline.py     # Base training pipeline
│   │   ├── sequential_training_pipeline.py  # Enhanced with theory tracking
│   │   └── inference_pipeline.py    # Inference pipeline
│   │
│   ├── theory/                      # Theoretical analysis (NEW)
│   │   ├── regret_tracker.py        # R(T) tracking
│   │   ├── fairness_tracker.py      # V(T) tracking
│   │   ├── bounds.py                # O(d√(KT log(T/δ))) bounds
│   │   ├── adaptive_vs_static.py    # Proves R_adaptive/R_static → 0
│   │   └── theorem_verification.py  # Monte Carlo verification
│   │
│   ├── metrics/                     # Evaluation metrics (NEW)
│   │   ├── ibr.py                   # Intersectional Bias Reduction
│   │   ├── far.py                   # Fairness-Aware Regret
│   │   └── comprehensive_evaluator.py
│   │
│   ├── output/                      # Output standardization (NEW)
│   │   └── csv_manager.py           # Full-form column names
│   │
│   ├── crosslingual/                # Cross-lingual analysis (NEW)
│   │   ├── transfer_analyzer.py     # EN→HI, EN→BN transfer
│   │   ├── code_mixing_handler.py   # Hinglish/Benglish detection
│   │   └── parallel_evaluator.py    # Parallel sample evaluation
│   │
│   ├── ablation/                    # Ablation framework (NEW)
│   │   ├── config_generator.py      # 14+ ablation configurations
│   │   ├── ablation_runner.py       # Automated experiment runner
│   │   └── results_analyzer.py      # Component importance
│   │
│   └── data/                        # Data handling
│       ├── dataset_loader.py        # Multi-CrowS-Pairs, IndiBias
│       └── bias_categories.py       # Full-form bias category mappings
│
├── scripts/                         # Executable scripts
│   ├── run_experiment.py            # Main experiment orchestrator
│   ├── generate_all_results.py      # TACL publication suite
│   ├── evaluate_with_metrics.py     # IBR/FAR evaluation
│   ├── train_bandit.py              # Train specific bandit
│   ├── evaluate_system.py           # Evaluate trained system
│   ├── create_steering_vectors.py   # Create steering vectors
│   └── prepare_evaluation_data.py   # Prepare datasets
│
├── tests/                           # Unit tests
├── results/                         # Output results
├── logs/                            # Log files
├── checkpoints/                     # Model checkpoints
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup
├── .env.example                     # Environment template
└── README.md                        # This file
```

---

## 🔬 Theoretical Guarantees

### Regret Bound

LinUCB achieves sublinear regret:

```
R(T) ≤ O(d√(KT log(T/δ)))
```

Where:
- `d` = context dimension (128)
- `K` = number of arms (6)
- `T` = number of rounds
- `δ` = confidence parameter

### Adaptive vs Static

The framework proves that adaptive selection outperforms any static arm:

```
lim(T→∞) R_adaptive(T) / R_static(T) → 0
```

### Verification

Run Monte Carlo simulations to verify theoretical claims:

```python
from src.theory import TheoremVerifier

verifier = TheoremVerifier(n_arms=6, context_dim=128, n_simulations=1000)
results = verifier.run_all_verifications(T=1000)
print(verifier.get_summary())
```

---

## 🧪 Ablation Studies

### Standard Configurations (14+)

| Category | Configurations |
|----------|----------------|
| **Full System** | `full` |
| **Baselines** | `random`, `static_baseline`, `static_gender`, `static_prompt` |
| **Component Ablations** | `no_context`, `no_steering`, `no_prompt`, `no_output_adjust` |
| **Bandit Algorithms** | `linucb`, `thompson`, `neural` |
| **Hyperparameter Sensitivity** | `alpha_0.5`, `alpha_2.0`, `lambda_0.0`, `lambda_1.0` |

### Run Ablation Study

```python
from src.ablation import AblationConfigGenerator, AblationRunner

# Generate configurations
generator = AblationConfigGenerator()
configs = generator.generate_all()

# Run experiments
runner = AblationRunner(results_dir='./ablation_results')
runner.run_all(configs)

# Analyze results
from src.ablation import AblationResultsAnalyzer
analyzer = AblationResultsAnalyzer(runner.load_results())
print(analyzer.generate_summary())
print(analyzer.generate_latex_table())
```

---

## 🌐 Cross-Lingual Transfer

### Transfer Analysis

Analyze how debiasing transfers across languages:

```python
from src.crosslingual import TransferAnalyzer

analyzer = TransferAnalyzer(
    source_languages=['en'],
    target_languages=['hi', 'bn']
)

# Add observations from experiments
analyzer.add_observation(language='en', category='gender', baseline_bias=0.8, method_bias=0.3)
analyzer.add_observation(language='hi', category='gender', baseline_bias=0.7, method_bias=0.4)

# Compute transfer ratios
transfers = analyzer.compute_all_transfers()
print(transfers['en->hi'].transfer_ratio)  # 1.0 = perfect transfer
```

### Code-Mixing Detection

Handle Hindi-English (Hinglish) and Bengali-English input:

```python
from src.crosslingual import CodeMixingDetector

detector = CodeMixingDetector()
result = detector.detect("Mujhe lagta hai this is a good idea")
print(result.is_code_mixed)  # True
print(result.languages_detected)  # ['hi', 'en']
```

---

## 📈 Usage Examples

### Training with Enhanced Tracking

```python
from src.pipeline import SequentialTrainingPipeline

pipeline = SequentialTrainingPipeline(
    inference_pipeline=inference_pipeline,
    n_arms=6,
    context_dim=128,
    lambda_fairness=0.5,
    enable_wandb=True
)

results = pipeline.train_sequential(
    train_data=train_data,
    eval_data=eval_data,
    n_epochs=3,
    warmup_samples=50
)

print(f"IBR: {results['ibr']:.4f}")
print(f"FAR: {results['far']:.4f}")
print(f"Regret Bound Satisfied: {results['regret_bound_satisfied']}")
```

### Evaluation with IBR/FAR

```python
from src.metrics import ComprehensiveMetricsEvaluator

evaluator = ComprehensiveMetricsEvaluator(
    lambda_weight=0.5,
    bias_threshold=0.3
)

# Add observations
for sample in test_data:
    evaluator.add_observation(
        bias_score=sample['bias'],
        reward=sample['reward'],
        category=sample['category'],
        language=sample['language']
    )

# Evaluate
result = evaluator.evaluate()
print(f"IBR: {result.ibr.ibr_score:.4f}")
print(f"FAR: {result.far.far_score:.4f}")
print(f"Worst category: {result.ibr.worst_category}")
```

### Standardized CSV Output

```python
from src.output import CSVOutputManager

manager = CSVOutputManager(output_dir='./results', timestamp_files=True)
manager.save_main_results(df)  # Automatically uses full-form column names
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# Required for HuggingFace models
HF_TOKEN=hf_xxxxxxxxxxxx

# Optional
WANDB_PROJECT=fair-cb
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration (`config/model_config.py`)

```python
from config.model_config import get_model_config, get_all_models

# Get specific model
config = get_model_config('qwen')
print(config['model_id'])  # Qwen/Qwen2.5-7B-Instruct

# List all supported models
models = get_all_models()
print(models)  # ['qwen', 'aya', 'llama']
```

### Bandit Configuration

```python
from config.bandit_config import get_bandit_config

config = get_bandit_config('linucb')
print(config['alpha'])  # 1.0
print(config['context_dim'])  # 128
```

---

## 🧹 Memory Management

The system is optimized for 24GB VRAM:

- 4-bit quantization (NF4) for LLMs
- Sequential model loading
- Aggressive memory cleanup
- Neural bandit on CPU (avoids GPU conflicts)

---

## 📋 Expected Results

### Runtime (24GB GPU)

| Task | Duration |
|------|----------|
| Dataset preparation | 10-15 min |
| Steering vector creation | 30-45 min |
| Training (1 epoch, 1000 samples) | 2-3 hours/algorithm |
| Evaluation (200 samples) | 15-20 min/algorithm |
| Complete experiment (3 epochs, 3 algorithms) | 20-24 hours |

### Disk Usage

| Component | Size |
|-----------|------|
| Steering vectors | ~450MB |
| Checkpoints | ~50-150MB |
| Datasets | ~100MB |
| Results/logs | ~200MB |
| **Total** | ~1-2GB |

---

## 🔧 Troubleshooting

<details>
<summary><b>Out of Memory (OOM) Errors</b></summary>

1. Reduce batch size in neural bandit config
2. Decrease `--max_train_samples` during training
3. Use fewer warmup samples
4. Ensure previous models are unloaded

</details>

<details>
<summary><b>Slow Training</b></summary>

1. Reduce `--eval-every` to evaluate less frequently
2. Use smaller evaluation set with `--max_eval_samples`
3. Start with LinUCB (fastest) before Neural Bandit
4. Use `--max_train_samples` for quick testing

</details>

<details>
<summary><b>Missing Steering Vectors</b></summary>

```bash
python scripts/create_steering_vectors.py
```

</details>

<details>
<summary><b>W&B Login Issues</b></summary>

```bash
wandb login
# Or disable: python scripts/train_bandit.py --no_wandb
```

</details>

---

## 📜 Citation

```bibtex
@article{fair_cb_2026,
  title={Fair-CB: Fairness-Aware Contextual Bandits for Adaptive Multilingual LLM Debiasing},
  author={Your Name},
  journal={Transactions of the Association for Computational Linguistics},
  year={2026},
  note={Under Review}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Documentation](docs/) · [Issues](https://github.com/yourusername/Fair-CB/issues) · [Contributing](CONTRIBUTING.md)**

</div>
