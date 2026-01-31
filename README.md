# Fair-CB: Fairness-Aware Contextual Bandits for Multilingual LLM Debiasing

<div align="center">

**Adaptive Multi-Armed Bandit Debiasing Strategy Selection for Multilingual Large Language Models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 🎯 Overview

Fair-CB is a research framework that dynamically selects optimal debiasing interventions for multilingual LLMs using contextual bandit algorithms.

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | Qwen2.5-1.5B, Llama-3.2-1B, Gemma-2-2B |
| **Multilingual** | English, Hindi, Bengali + code-mixing |
| **Novel Metrics** | IBR (Intersectional Bias Reduction), FAR (Fairness-Aware Regret) |
| **Statistical Rigor** | Bootstrap CIs, paired t-tests, effect sizes |
| **Counterfactual Evaluation** | True regret via all-arm evaluation |

### Models (24GB VRAM Constraint)

| Model | Parameters | VRAM (4-bit) |
|-------|------------|--------------|
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | ~2.5GB |
| meta-llama/Llama-3.2-1B-Instruct | 1B | ~1.5GB |
| google/gemma-2-2b-it | 2B | ~3.0GB |

### Debiasing Arms (6 Strategies)

| Arm | Strategy | Description |
|-----|----------|-------------|
| 0 | No Intervention | Baseline |
| 1 | Gender Steering | Steering vector for gender bias |
| 2 | Race Steering | Steering vector for race/ethnicity |
| 3 | Religion Steering | Steering vector for religious bias |
| 4 | Prompt Prefix | Fairness-aware prompt modification |
| 5 | Output Adjustment | Post-hoc output debiasing |

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt
python setup.py develop

# Configure
cp .env.example .env
# Add HF_TOKEN to .env

# Run experiment
python scripts/generate_all_results.py --quick  # Test
python scripts/generate_all_results.py          # Full

# Evaluate with metrics
python scripts/evaluate_with_metrics.py --dataset both --generate-latex
```

---

## 📊 Novel Metrics

### IBR (Intersectional Bias Reduction)

```
IBR = HarmonicMean({reduction_per_category})
Signed_IBR = IBR * (1 - n_worsened/n_total)  # Penalizes bias increases
```

### FAR (Fairness-Aware Regret)

```
FAR = R(T) + λ·V(T)
```
Where R(T) = cumulative regret, V(T) = fairness violations, λ = weight (default 0.5)

---

## 📁 Project Structure

```
Fair-CB/
├── config/                      # Configuration
│   └── model_config.py          # Model registry (1.5B/1B/2B)
├── src/
│   ├── theory/                  # Theoretical analysis
│   ├── metrics/                 # IBR, FAR, statistical tests
│   ├── evaluation/              # Counterfactual evaluator
│   ├── crosslingual/            # Transfer analysis
│   ├── ablation/                # Ablation framework
│   └── pipeline/                # Training pipelines
├── scripts/
│   ├── generate_all_results.py  # Full experiment suite
│   └── evaluate_with_metrics.py # IBR/FAR evaluation
└── docs/
    └── theoretical_limitations.md  # Documented limitations
```

---

## 🔬 Theoretical Guarantees

### Regret Bound (LinUCB)

```
R(T) ≤ O(d√(KT log(T/δ)))
```

### Counterfactual Evaluation

For true regret computation, use `CounterfactualEvaluator`:

```python
from src.evaluation import CounterfactualEvaluator

evaluator = CounterfactualEvaluator(n_arms=6)
summary = evaluator.evaluate_dataset(
    samples=test_data,
    arm_executor=run_arm,
    bandit_selector=bandit.select_arm
)
print(f"True Regret: {summary.true_cumulative_regret}")
print(f"Adaptive/Static Ratio: {summary.adaptive_vs_static_ratio}")
```

---

## ⚠️ Known Limitations

See [docs/theoretical_limitations.md](docs/theoretical_limitations.md) for full details:

1. **LinUCB realizability**: Reward function is not perfectly linear
2. **Non-i.i.d. contexts**: Data is grouped by bias category
3. **24GB VRAM constraint**: Limited to 1-2B parameter models
4. **Regret estimation**: Training uses estimated (not counterfactual) regret

---

## 📋 Statistical Reporting

All results include:
- **95% Bootstrap CIs**: `compute_bootstrap_ci()`
- **Paired t-tests**: `compute_paired_ttest()`
- **Effect sizes**: Cohen's d with interpretation

```python
from src.metrics import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
analyzer.add_results('Fair-CB', scores)
ci = analyzer.get_confidence_interval('Fair-CB')
print(f"Mean: {ci.mean:.4f} [{ci.lower:.4f}, {ci.upper:.4f}]")
```

---

## 📜 Citation

```bibtex
@article{fair_cb_2026,
  title={Fair-CB: Fairness-Aware Contextual Bandits for Adaptive Multilingual LLM Debiasing},
  author={Your Name},
  journal={Transactions of the Association for Computational Linguistics},
  year={2026}
}
```

---

## 📄 License

MIT License
