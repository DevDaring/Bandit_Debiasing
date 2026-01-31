# Theoretical Limitations

This document acknowledges the theoretical assumptions and limitations of Fair-CB as required for scientific transparency.

---

## LinUCB Assumptions

The theoretical regret bound `R(T) ≤ O(d√(KT log(T/δ)))` requires assumptions that may not perfectly hold in our LLM debiasing setting:

### 1. Realizability (Violated)
- **Assumption**: True reward function is linear in context features
- **Reality**: Bias scores are outputs of neural embedding models, not linear functions
- **Mitigation**: Our experiments show empirical regret still follows sublinear trends, suggesting the linear approximation is useful even if not exact

### 2. I.I.D. Contexts (Partially Violated)
- **Assumption**: Contexts are drawn i.i.d.
- **Reality**: Multi-CrowS-Pairs and IndiBias have grouped structure by bias category
- **Mitigation**: We shuffle data and report per-category performance separately

### 3. Sub-Gaussian Noise (Unknown)
- **Assumption**: Reward noise is sub-Gaussian
- **Reality**: Bias scoring involves neural models with unknown noise distribution
- **Mitigation**: We use robust statistics and report confidence intervals

---

## Regret Computation

### Counterfactual Limitation
- Full counterfactual evaluation (running all 6 arms on every sample) is computationally expensive
- We provide `CounterfactualEvaluator` for rigorous evaluation on held-out test sets
- Training uses estimated regret; evaluation can use true counterfactual regret

### Oracle Regret Estimation
- During online learning, the true optimal arm is unknown
- We estimate optimal reward from arm means, which is biased for rarely-selected arms
- For publication results, use `CounterfactualEvaluator` on a test subset

---

## Metric Limitations

### IBR (Intersectional Bias Reduction)
- Harmonic mean is sensitive to any category with low reduction
- If baseline bias ≈ 0 for a category, reduction ratio is undefined
- We now report both `ibr_score` and `signed_ibr` (latter penalizes worsened categories)

### FAR (Fairness-Aware Regret)
- λ (fairness weight) is a hyperparameter with no principled selection method
- We report results for λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}

---

## Model Selection

The models used (1.5B-2B parameters) are constrained by 24GB VRAM:
- Qwen2.5-1.5B-Instruct
- Llama-3.2-1B-Instruct
- Gemma-2-2B-it

Larger models may exhibit different bias patterns and debiasing effectiveness.

---

## Statistical Considerations

All reported results include:
- 95% bootstrap confidence intervals
- Paired t-tests for method comparisons
- Effect sizes (Cohen's d) for practical significance

This document will be updated as additional limitations are identified.
