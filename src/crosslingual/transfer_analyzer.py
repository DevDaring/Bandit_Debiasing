"""
Cross-lingual transfer analysis for steering vectors.

Analyzes how steering vectors trained on one language
transfer to other languages.

Transfer Ratio = target_reduction / source_reduction
- 1.0 = perfect transfer
- < 1.0 = reduced efficacy in target language
- > 1.0 = improved efficacy (rare, but possible)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result for a single transfer direction."""
    source_language: str
    target_language: str
    source_bias_reduction: float
    target_bias_reduction: float
    transfer_ratio: float
    n_source_samples: int
    n_target_samples: int
    is_positive_transfer: bool


def compute_transfer_ratio(
    source_reduction: float,
    target_reduction: float,
    epsilon: float = 1e-10
) -> float:
    """
    Compute transfer ratio.

    transfer_ratio = target_reduction / source_reduction

    Args:
        source_reduction: Bias reduction in source language
        target_reduction: Bias reduction in target language
        epsilon: Small value to avoid division by zero

    Returns:
        Transfer ratio (1.0 = perfect transfer)
    """
    if source_reduction <= epsilon:
        return 0.0 if target_reduction <= epsilon else float('inf')

    return target_reduction / source_reduction


class TransferAnalyzer:
    """
    Analyze cross-lingual transfer of debiasing strategies.

    Tracks:
    1. Bias reduction in source language (where steering vector trained)
    2. Bias reduction in target languages
    3. Transfer ratio for each direction
    4. Per-category transfer efficacy
    """

    def __init__(
        self,
        source_languages: List[str] = None,
        target_languages: List[str] = None,
        bias_categories: List[str] = None
    ):
        """
        Initialize transfer analyzer.

        Args:
            source_languages: Languages used for training (default: ['en'])
            target_languages: Languages to evaluate transfer on
            bias_categories: Bias categories to track
        """
        self.source_languages = source_languages or ['en']
        self.target_languages = target_languages or ['hi', 'bn']
        self.bias_categories = bias_categories or [
            'gender', 'race', 'religion', 'caste'
        ]

        # Tracking: lang -> category -> list of (baseline, method) scores
        self.scores: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_observation(
        self,
        language: str,
        category: str,
        baseline_bias: float,
        method_bias: float
    ):
        """
        Add a single observation.

        Args:
            language: Language code
            category: Bias category
            baseline_bias: Bias without intervention
            method_bias: Bias with debiasing method
        """
        self.scores[language][category].append((baseline_bias, method_bias))

    def add_batch(self, observations: List[Dict[str, Any]]):
        """
        Add batch of observations.

        Each observation should have:
        - language, category, baseline_bias, method_bias
        """
        for obs in observations:
            self.add_observation(
                language=obs['language'],
                category=obs['category'],
                baseline_bias=obs['baseline_bias'],
                method_bias=obs['method_bias']
            )

    def compute_bias_reduction(self, language: str, category: str = None) -> float:
        """
        Compute mean bias reduction for a language.

        Args:
            language: Language code
            category: Optional category filter

        Returns:
            Mean bias reduction ratio
        """
        reductions = []

        categories = [category] if category else self.scores[language].keys()

        for cat in categories:
            for baseline, method in self.scores[language].get(cat, []):
                if baseline > 0:
                    reduction = (baseline - method) / baseline
                    reductions.append(reduction)

        return np.mean(reductions) if reductions else 0.0

    def compute_transfer(
        self,
        source_lang: str,
        target_lang: str,
        category: str = None
    ) -> TransferResult:
        """
        Compute transfer from source to target language.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            category: Optional category filter

        Returns:
            TransferResult with transfer details
        """
        source_reduction = self.compute_bias_reduction(source_lang, category)
        target_reduction = self.compute_bias_reduction(target_lang, category)

        transfer_ratio = compute_transfer_ratio(source_reduction, target_reduction)

        # Count samples
        if category:
            n_source = len(self.scores[source_lang].get(category, []))
            n_target = len(self.scores[target_lang].get(category, []))
        else:
            n_source = sum(len(v) for v in self.scores[source_lang].values())
            n_target = sum(len(v) for v in self.scores[target_lang].values())

        return TransferResult(
            source_language=source_lang,
            target_language=target_lang,
            source_bias_reduction=source_reduction,
            target_bias_reduction=target_reduction,
            transfer_ratio=transfer_ratio,
            n_source_samples=n_source,
            n_target_samples=n_target,
            is_positive_transfer=transfer_ratio > 0.5
        )

    def compute_all_transfers(self) -> Dict[str, TransferResult]:
        """
        Compute transfer for all source-target pairs.

        Returns:
            Dict mapping 'source->target' to TransferResult
        """
        results = {}

        for source in self.source_languages:
            for target in self.target_languages:
                if source != target:
                    key = f"{source}->{target}"
                    results[key] = self.compute_transfer(source, target)

        return results

    def compute_per_category_transfers(
        self,
        source_lang: str = 'en'
    ) -> Dict[str, Dict[str, TransferResult]]:
        """
        Compute per-category transfer from source language.

        Args:
            source_lang: Source language

        Returns:
            Dict[target_lang][category] -> TransferResult
        """
        results = defaultdict(dict)

        for target in self.target_languages:
            if source_lang == target:
                continue

            for category in self.bias_categories:
                results[target][category] = self.compute_transfer(
                    source_lang, target, category
                )

        return dict(results)

    def get_best_transfer_direction(self) -> Tuple[str, str, float]:
        """
        Find the source-target pair with best transfer.

        Returns:
            Tuple of (source, target, transfer_ratio)
        """
        all_transfers = self.compute_all_transfers()

        if not all_transfers:
            return ('', '', 0.0)

        best_key = max(all_transfers, key=lambda k: all_transfers[k].transfer_ratio)
        result = all_transfers[best_key]

        return (result.source_language, result.target_language, result.transfer_ratio)

    def get_worst_transfer_direction(self) -> Tuple[str, str, float]:
        """
        Find the source-target pair with worst transfer.

        Returns:
            Tuple of (source, target, transfer_ratio)
        """
        all_transfers = self.compute_all_transfers()

        if not all_transfers:
            return ('', '', 0.0)

        worst_key = min(all_transfers, key=lambda k: all_transfers[k].transfer_ratio)
        result = all_transfers[worst_key]

        return (result.source_language, result.target_language, result.transfer_ratio)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive transfer analysis summary."""
        all_transfers = self.compute_all_transfers()

        if not all_transfers:
            return {'status': 'No transfer data available'}

        ratios = [t.transfer_ratio for t in all_transfers.values()]

        return {
            'n_directions': len(all_transfers),
            'mean_transfer_ratio': np.mean(ratios),
            'min_transfer_ratio': np.min(ratios),
            'max_transfer_ratio': np.max(ratios),
            'directions_with_positive_transfer': sum(1 for t in all_transfers.values() if t.is_positive_transfer),
            'per_direction': {
                key: {
                    'source': r.source_language,
                    'target': r.target_language,
                    'source_reduction': r.source_bias_reduction,
                    'target_reduction': r.target_bias_reduction,
                    'transfer_ratio': r.transfer_ratio,
                    'positive': r.is_positive_transfer
                }
                for key, r in all_transfers.items()
            }
        }

    def generate_transfer_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Generate transfer ratio matrix for visualization.

        Returns:
            Dict[source][target] -> transfer_ratio
        """
        all_langs = set(self.source_languages) | set(self.target_languages)
        matrix = {lang: {} for lang in all_langs}

        for source in all_langs:
            for target in all_langs:
                if source == target:
                    matrix[source][target] = 1.0  # Perfect transfer to self
                else:
                    result = self.compute_transfer(source, target)
                    matrix[source][target] = result.transfer_ratio

        return matrix

    def reset(self):
        """Clear all accumulated data."""
        self.scores = defaultdict(lambda: defaultdict(list))

    def __repr__(self) -> str:
        summary = self.get_summary()
        if 'mean_transfer_ratio' in summary:
            return (f"TransferAnalyzer(mean_ratio={summary['mean_transfer_ratio']:.3f}, "
                    f"directions={summary['n_directions']})")
        return "TransferAnalyzer(no data)"
