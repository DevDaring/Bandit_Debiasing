"""
Ablation results analyzer.

Analyzes ablation study results to determine component importance.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from collections import defaultdict

from .ablation_runner import AblationResult
from .config_generator import AblationConfig

logger = logging.getLogger(__name__)


@dataclass
class ComponentImportance:
    """Importance score for a component."""
    component_name: str
    importance_score: float
    delta_ibr: float
    delta_far: float
    rank: int
    is_critical: bool  # True if removal causes >20% performance drop


class AblationResultsAnalyzer:
    """
    Analyze ablation study results.

    Computes:
    1. Component importance scores
    2. Performance deltas (full - ablated)
    3. Critical component identification
    4. Hyperparameter sensitivity
    """

    def __init__(
        self,
        results: List[AblationResult],
        primary_metric: str = 'ibr'
    ):
        """
        Initialize analyzer.

        Args:
            results: List of ablation results
            primary_metric: Primary metric for importance ranking
        """
        self.results = {r.config.name: r for r in results if r.success}
        self.primary_metric = primary_metric

    def get_full_system_performance(self) -> Dict[str, float]:
        """Get performance of full system (baseline for comparison)."""
        if 'full' in self.results:
            return self.results['full'].metrics
        return {}

    def compute_component_importance(self) -> List[ComponentImportance]:
        """
        Compute importance scores for all components.

        Importance = performance_full - performance_ablated

        Returns:
            List of ComponentImportance sorted by importance
        """
        full_metrics = self.get_full_system_performance()
        if not full_metrics:
            logger.warning("No 'full' configuration found in results")
            return []

        full_primary = full_metrics.get(self.primary_metric, 0)
        full_ibr = full_metrics.get('ibr', 0)
        full_far = full_metrics.get('far', 0)

        # Component ablation mapping
        component_ablations = {
            'Context Extractor': 'no_context',
            'Steering Vectors': 'no_steering',
            'Prompt Prefix': 'no_prompt',
            'Output Adjustment': 'no_output_adjust',
        }

        importances = []

        for component, ablation_name in component_ablations.items():
            if ablation_name not in self.results:
                continue

            ablated = self.results[ablation_name]
            ablated_primary = ablated.metrics.get(self.primary_metric, 0)
            ablated_ibr = ablated.metrics.get('ibr', 0)
            ablated_far = ablated.metrics.get('far', 0)

            # For IBR, higher is better → importance = full - ablated
            # For FAR, lower is better → importance = ablated - full
            if self.primary_metric == 'ibr':
                importance = full_primary - ablated_primary
            else:  # far
                importance = ablated_primary - full_primary

            delta_ibr = full_ibr - ablated_ibr
            delta_far = ablated_far - full_far  # Positive if ablated is worse

            # Critical: >20% relative drop
            is_critical = False
            if full_ibr > 0:
                relative_drop = delta_ibr / full_ibr
                is_critical = relative_drop > 0.2

            importances.append(ComponentImportance(
                component_name=component,
                importance_score=importance,
                delta_ibr=delta_ibr,
                delta_far=delta_far,
                rank=0,  # Set later
                is_critical=is_critical
            ))

        # Sort by importance and assign ranks
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        for i, imp in enumerate(importances):
            imp.rank = i + 1

        return importances

    def compute_bandit_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Compare bandit algorithms.

        Returns:
            Dict[algorithm] -> metrics
        """
        algorithms = ['linucb', 'thompson', 'neural', 'random']
        comparison = {}

        for algo in algorithms:
            # Try different naming conventions
            possible_names = [algo, f'bandit_{algo}']
            for name in possible_names:
                if name in self.results:
                    comparison[algo] = self.results[name].metrics
                    break

        return comparison

    def compute_static_vs_adaptive(self) -> Dict[str, Any]:
        """
        Compare static arm strategies vs adaptive (full system).

        Returns:
            Comparison statistics
        """
        full_metrics = self.get_full_system_performance()
        if not full_metrics:
            return {}

        static_results = {}
        for result_name, result in self.results.items():
            if result_name.startswith('static'):
                static_results[result_name] = result.metrics

        if not static_results:
            return {}

        # Find best static arm
        best_static_name = max(
            static_results.keys(),
            key=lambda k: static_results[k].get('ibr', 0)
        )
        best_static_metrics = static_results[best_static_name]

        return {
            'full_ibr': full_metrics.get('ibr', 0),
            'best_static_ibr': best_static_metrics.get('ibr', 0),
            'best_static_name': best_static_name,
            'adaptive_advantage': full_metrics.get('ibr', 0) - best_static_metrics.get('ibr', 0),
            'static_results': static_results
        }

    def compute_hyperparameter_sensitivity(
        self,
        param_name: str
    ) -> Dict[Any, Dict[str, float]]:
        """
        Compute sensitivity to a hyperparameter.

        Args:
            param_name: Hyperparameter name (e.g., 'alpha', 'lambda_fairness')

        Returns:
            Dict[param_value] -> metrics
        """
        sensitivity = {}

        for name, result in self.results.items():
            param_value = getattr(result.config, param_name, None)
            if param_value is not None:
                sensitivity[param_value] = result.metrics

        return sensitivity

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive ablation summary.

        Returns:
            Summary dictionary
        """
        full_metrics = self.get_full_system_performance()
        importances = self.compute_component_importance()
        static_comparison = self.compute_static_vs_adaptive()
        bandit_comparison = self.compute_bandit_comparison()

        # Critical components
        critical = [imp.component_name for imp in importances if imp.is_critical]

        # Most important component
        most_important = importances[0].component_name if importances else ''

        # Best bandit
        best_bandit = ''
        if bandit_comparison:
            best_bandit = max(
                bandit_comparison.keys(),
                key=lambda k: bandit_comparison[k].get('ibr', 0)
            )

        return {
            'n_experiments': len(self.results),
            'full_system': full_metrics,
            'component_importance': [
                {
                    'rank': imp.rank,
                    'component': imp.component_name,
                    'importance': imp.importance_score,
                    'delta_ibr': imp.delta_ibr,
                    'delta_far': imp.delta_far,
                    'is_critical': imp.is_critical
                }
                for imp in importances
            ],
            'most_important_component': most_important,
            'critical_components': critical,
            'adaptive_vs_static': static_comparison.get('adaptive_advantage', 0),
            'best_bandit_algorithm': best_bandit,
            'bandit_comparison': bandit_comparison
        }

    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for publication.

        Returns:
            LaTeX table string
        """
        summary = self.generate_summary()

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & IBR $\uparrow$ & FAR $\downarrow$ & $\Delta$IBR & Critical \\",
            r"\midrule",
        ]

        # Full system first
        full_metrics = summary.get('full_system', {})
        lines.append(
            f"Full System & {full_metrics.get('ibr', 0):.3f} & "
            f"{full_metrics.get('far', 0):.3f} & -- & -- \\\\"
        )

        lines.append(r"\midrule")

        # Component ablations
        for imp in summary.get('component_importance', []):
            # Get ablated metrics
            critical_mark = r"\checkmark" if imp['is_critical'] else "--"
            lines.append(
                f"w/o {imp['component']} & -- & -- & "
                f"{imp['delta_ibr']:+.3f} & {critical_mark} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        n = len(self.results)
        return f"AblationResultsAnalyzer(n_results={n}, primary_metric={self.primary_metric})"
