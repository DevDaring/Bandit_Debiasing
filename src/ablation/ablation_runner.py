"""
Ablation study runner.

Runs ablation experiments with different configurations and collects results.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import time
from datetime import datetime

from .config_generator import AblationConfig

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result from a single ablation experiment."""
    config: AblationConfig
    metrics: Dict[str, float]
    per_category_metrics: Dict[str, Dict[str, float]]
    per_language_metrics: Dict[str, Dict[str, float]]
    runtime_seconds: float
    timestamp: str
    n_samples: int
    success: bool
    error_message: Optional[str] = None


class AblationRunner:
    """
    Run ablation experiments systematically.

    Provides:
    1. Automated experiment execution
    2. Result collection and storage
    3. Progress tracking
    4. Error handling with resume capability
    """

    def __init__(
        self,
        experiment_fn: Optional[Callable[[AblationConfig], Dict[str, Any]]] = None,
        results_dir: str = './ablation_results',
        save_intermediate: bool = True
    ):
        """
        Initialize ablation runner.

        Args:
            experiment_fn: Function that takes AblationConfig and returns metrics dict
            results_dir: Directory to save results
            save_intermediate: Whether to save after each experiment
        """
        self.experiment_fn = experiment_fn
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.save_intermediate = save_intermediate

        self.results: List[AblationResult] = []
        self.failed_configs: List[AblationConfig] = []

    def set_experiment_function(self, fn: Callable[[AblationConfig], Dict[str, Any]]):
        """
        Set the experiment function.

        The function should take an AblationConfig and return a dict with:
        - 'metrics': Dict[str, float] of overall metrics
        - 'per_category': Optional Dict[str, Dict[str, float]]
        - 'per_language': Optional Dict[str, Dict[str, float]]
        - 'n_samples': int
        """
        self.experiment_fn = fn

    def run_single(self, config: AblationConfig) -> AblationResult:
        """
        Run a single ablation experiment.

        Args:
            config: Ablation configuration

        Returns:
            AblationResult
        """
        if self.experiment_fn is None:
            return AblationResult(
                config=config,
                metrics={},
                per_category_metrics={},
                per_language_metrics={},
                runtime_seconds=0,
                timestamp=datetime.now().isoformat(),
                n_samples=0,
                success=False,
                error_message="No experiment function set"
            )

        logger.info(f"Running ablation: {config.name}")
        start_time = time.time()

        try:
            result_dict = self.experiment_fn(config)

            runtime = time.time() - start_time

            result = AblationResult(
                config=config,
                metrics=result_dict.get('metrics', {}),
                per_category_metrics=result_dict.get('per_category', {}),
                per_language_metrics=result_dict.get('per_language', {}),
                runtime_seconds=runtime,
                timestamp=datetime.now().isoformat(),
                n_samples=result_dict.get('n_samples', 0),
                success=True
            )

            self.results.append(result)

            if self.save_intermediate:
                self._save_result(result)

            logger.info(f"Completed {config.name} in {runtime:.2f}s")

            return result

        except Exception as e:
            runtime = time.time() - start_time
            logger.error(f"Failed ablation {config.name}: {str(e)}")

            result = AblationResult(
                config=config,
                metrics={},
                per_category_metrics={},
                per_language_metrics={},
                runtime_seconds=runtime,
                timestamp=datetime.now().isoformat(),
                n_samples=0,
                success=False,
                error_message=str(e)
            )

            self.failed_configs.append(config)

            return result

    def run_all(
        self,
        configs: List[AblationConfig],
        skip_completed: bool = True
    ) -> List[AblationResult]:
        """
        Run all ablation experiments.

        Args:
            configs: List of configurations to run
            skip_completed: Whether to skip already-completed experiments

        Returns:
            List of AblationResult
        """
        completed_names = set()
        if skip_completed:
            completed_names = self._get_completed_names()
            logger.info(f"Skipping {len(completed_names)} completed experiments")

        results = []
        total = len(configs)

        for i, config in enumerate(configs):
            if config.name in completed_names:
                logger.info(f"Skipping completed: {config.name}")
                continue

            logger.info(f"Progress: {i+1}/{total} - {config.name}")
            result = self.run_single(config)
            results.append(result)

        return results

    def _save_result(self, result: AblationResult):
        """Save a single result to disk."""
        filepath = self.results_dir / f"{result.config.name}.json"

        data = {
            'config': result.config.to_dict(),
            'metrics': result.metrics,
            'per_category_metrics': result.per_category_metrics,
            'per_language_metrics': result.per_language_metrics,
            'runtime_seconds': result.runtime_seconds,
            'timestamp': result.timestamp,
            'n_samples': result.n_samples,
            'success': result.success,
            'error_message': result.error_message
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_completed_names(self) -> set:
        """Get names of completed experiments from disk."""
        completed = set()

        for filepath in self.results_dir.glob('*.json'):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if data.get('success', False):
                    completed.add(data['config']['name'])
            except Exception:
                pass

        return completed

    def load_results(self) -> List[AblationResult]:
        """Load all results from disk."""
        results = []

        for filepath in sorted(self.results_dir.glob('*.json')):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                result = AblationResult(
                    config=AblationConfig.from_dict(data['config']),
                    metrics=data['metrics'],
                    per_category_metrics=data.get('per_category_metrics', {}),
                    per_language_metrics=data.get('per_language_metrics', {}),
                    runtime_seconds=data['runtime_seconds'],
                    timestamp=data['timestamp'],
                    n_samples=data['n_samples'],
                    success=data['success'],
                    error_message=data.get('error_message')
                )
                results.append(result)

            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

        self.results = results
        return results

    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame.

        Returns:
            DataFrame with all results
        """
        import pandas as pd

        rows = []
        for result in self.results:
            row = {
                'config_name': result.config.name,
                'success': result.success,
                'runtime_seconds': result.runtime_seconds,
                'n_samples': result.n_samples,
                **result.metrics
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save_summary(self, filepath: str = None):
        """Save summary of all results."""
        if filepath is None:
            filepath = self.results_dir / 'summary.json'
        else:
            filepath = Path(filepath)

        summary = {
            'n_total': len(self.results),
            'n_success': sum(1 for r in self.results if r.success),
            'n_failed': len(self.failed_configs),
            'total_runtime': sum(r.runtime_seconds for r in self.results),
            'results': [
                {
                    'name': r.config.name,
                    'success': r.success,
                    'metrics': r.metrics,
                    'runtime': r.runtime_seconds
                }
                for r in self.results
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved ablation summary to {filepath}")

    def __repr__(self) -> str:
        n_complete = len(self.results)
        n_failed = len(self.failed_configs)
        return f"AblationRunner(completed={n_complete}, failed={n_failed})"
