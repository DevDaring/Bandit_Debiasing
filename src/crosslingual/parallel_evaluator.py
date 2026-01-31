"""
Parallel evaluation across languages.

Evaluates debiasing on parallel samples (same content in different languages)
to enable fair cross-lingual comparison.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CrossLingualResult:
    """Result from parallel cross-lingual evaluation."""
    parallel_id: str
    languages: List[str]
    baseline_scores: Dict[str, float]
    method_scores: Dict[str, float]
    reductions: Dict[str, float]
    best_language: str
    worst_language: str
    variance_across_languages: float


class ParallelEvaluator:
    """
    Evaluate debiasing on parallel samples across languages.

    Uses parallel sentence pairs to enable fair comparison:
    - Same content in English, Hindi, Bengali
    - Measures per-language bias reduction
    - Computes cross-lingual consistency
    """

    def __init__(
        self,
        languages: List[str] = None
    ):
        """
        Initialize parallel evaluator.

        Args:
            languages: Languages to track
        """
        self.languages = languages or ['en', 'hi', 'bn']

        # Tracking: parallel_id -> lang -> (baseline, method)
        self.parallel_results: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)

    def add_result(
        self,
        parallel_id: str,
        language: str,
        baseline_bias: float,
        method_bias: float
    ):
        """
        Add result for a parallel sample.

        Args:
            parallel_id: Identifier linking parallel samples
            language: Language of this sample
            baseline_bias: Bias without intervention
            method_bias: Bias with debiasing method
        """
        self.parallel_results[parallel_id][language] = (baseline_bias, method_bias)

    def add_parallel_batch(
        self,
        parallel_id: str,
        results: Dict[str, Tuple[float, float]]
    ):
        """
        Add results for all languages of a parallel sample.

        Args:
            parallel_id: Parallel sample identifier
            results: Dict[language] -> (baseline_bias, method_bias)
        """
        for lang, (baseline, method) in results.items():
            self.add_result(parallel_id, lang, baseline, method)

    def evaluate_parallel(self, parallel_id: str) -> CrossLingualResult:
        """
        Evaluate a single parallel sample set.

        Args:
            parallel_id: Parallel sample identifier

        Returns:
            CrossLingualResult with per-language breakdown
        """
        results = self.parallel_results.get(parallel_id, {})

        if not results:
            return CrossLingualResult(
                parallel_id=parallel_id,
                languages=[],
                baseline_scores={},
                method_scores={},
                reductions={},
                best_language='',
                worst_language='',
                variance_across_languages=0.0
            )

        baseline_scores = {}
        method_scores = {}
        reductions = {}

        for lang, (baseline, method) in results.items():
            baseline_scores[lang] = baseline
            method_scores[lang] = method
            if baseline > 0:
                reductions[lang] = (baseline - method) / baseline
            else:
                reductions[lang] = 0.0

        # Find best and worst
        if reductions:
            best_language = max(reductions, key=reductions.get)
            worst_language = min(reductions, key=reductions.get)
            variance = np.var(list(reductions.values()))
        else:
            best_language = ''
            worst_language = ''
            variance = 0.0

        return CrossLingualResult(
            parallel_id=parallel_id,
            languages=list(results.keys()),
            baseline_scores=baseline_scores,
            method_scores=method_scores,
            reductions=reductions,
            best_language=best_language,
            worst_language=worst_language,
            variance_across_languages=variance
        )

    def evaluate_all(self) -> List[CrossLingualResult]:
        """
        Evaluate all parallel samples.

        Returns:
            List of CrossLingualResult for each parallel sample
        """
        return [
            self.evaluate_parallel(pid)
            for pid in self.parallel_results.keys()
        ]

    def get_aggregate_statistics(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all parallel samples.

        Returns:
            Dict with per-language and cross-lingual statistics
        """
        all_results = self.evaluate_all()

        if not all_results:
            return {'status': 'No parallel results available'}

        # Aggregate per-language reductions
        per_lang_reductions = defaultdict(list)
        for result in all_results:
            for lang, reduction in result.reductions.items():
                per_lang_reductions[lang].append(reduction)

        per_language_stats = {
            lang: {
                'mean_reduction': np.mean(reductions),
                'std_reduction': np.std(reductions),
                'n_samples': len(reductions)
            }
            for lang, reductions in per_lang_reductions.items()
        }

        # Cross-lingual consistency
        variances = [r.variance_across_languages for r in all_results if r.languages]
        mean_variance = np.mean(variances) if variances else 0.0

        # Count best/worst language frequency
        best_counts = defaultdict(int)
        worst_counts = defaultdict(int)
        for result in all_results:
            if result.best_language:
                best_counts[result.best_language] += 1
            if result.worst_language:
                worst_counts[result.worst_language] += 1

        return {
            'n_parallel_samples': len(all_results),
            'per_language': per_language_stats,
            'mean_cross_lingual_variance': mean_variance,
            'best_language_counts': dict(best_counts),
            'worst_language_counts': dict(worst_counts),
            'most_consistent_language': min(best_counts, key=best_counts.get) if best_counts else '',
            'overall_consistency_score': 1.0 - mean_variance  # Higher = more consistent
        }

    def get_parity_analysis(self) -> Dict[str, Any]:
        """
        Analyze parity of debiasing across languages.

        Returns:
            Parity statistics (how equally method performs across languages)
        """
        all_results = self.evaluate_all()

        if not all_results:
            return {}

        # For each parallel sample, compute parity (1 - normalized variance)
        parities = []
        for result in all_results:
            if len(result.reductions) >= 2:
                reductions = list(result.reductions.values())
                range_val = max(reductions) - min(reductions)
                mean_val = np.mean(reductions)
                if mean_val > 0:
                    parity = 1.0 - (range_val / mean_val)
                    parities.append(max(0, parity))

        if not parities:
            return {'mean_parity': 0.0, 'n_samples': 0}

        return {
            'mean_parity': np.mean(parities),
            'std_parity': np.std(parities),
            'min_parity': np.min(parities),
            'max_parity': np.max(parities),
            'n_samples': len(parities),
            'high_parity_rate': sum(1 for p in parities if p > 0.8) / len(parities)
        }

    def reset(self):
        """Clear all accumulated results."""
        self.parallel_results = defaultdict(dict)

    def __repr__(self) -> str:
        n_samples = len(self.parallel_results)
        return f"ParallelEvaluator(n_parallel_samples={n_samples}, languages={self.languages})"
