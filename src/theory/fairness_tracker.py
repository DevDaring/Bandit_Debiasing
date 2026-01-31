"""
Fairness violation tracker for Fair-CB.

Implements:
- Violation: V(t) = max(0, bias_score(t) - τ)
- Cumulative violations: V(T) = Σ_{t=1}^T V(t)
- Per-category violation tracking
- Violation rate monitoring

Key property: With fairness constraint, E[V(T)] ≤ O(√T)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ViolationEntry:
    """Single fairness violation observation."""
    timestep: int
    bias_score: float
    threshold: float
    violation: float
    cumulative_violation: float
    bias_category: Optional[str] = None
    language: Optional[str] = None
    selected_arm: Optional[int] = None


class FairnessViolationTracker:
    """
    Track fairness violations in the Fair-CB framework.

    A violation occurs when the bias score exceeds the threshold τ:
        V(t) = max(0, B(t) - τ)

    The fairness constraint requires cumulative violations to be sublinear:
        E[V(T)] ≤ O(√T)

    This tracker monitors:
    1. Per-timestep violations
    2. Cumulative violations
    3. Violation rates by category and language
    4. Constraint satisfaction verification
    """

    def __init__(
        self,
        threshold: float = 0.3,
        track_by_category: bool = True,
        track_by_language: bool = True
    ):
        """
        Initialize fairness violation tracker.

        Args:
            threshold: Bias threshold τ (violations occur when bias > τ)
            track_by_category: Track violations per bias category
            track_by_language: Track violations per language
        """
        self.threshold = threshold
        self.track_by_category = track_by_category
        self.track_by_language = track_by_language

        # Core tracking
        self.history: List[ViolationEntry] = []
        self.cumulative_violation: float = 0.0
        self.timestep: int = 0

        # Per-group tracking
        self.category_violations: Dict[str, List[float]] = defaultdict(list)
        self.language_violations: Dict[str, List[float]] = defaultdict(list)
        self.arm_violations: Dict[int, List[float]] = defaultdict(list)

        # Violation events (timesteps where violation > 0)
        self.violation_events: List[int] = []

    def update(
        self,
        bias_score: float,
        bias_category: Optional[str] = None,
        language: Optional[str] = None,
        selected_arm: Optional[int] = None
    ) -> float:
        """
        Record a new observation and update cumulative violation.

        Args:
            bias_score: Current bias score B(t) ∈ [0, 1]
            bias_category: Optional category (e.g., 'gender', 'race')
            language: Optional language code (e.g., 'en', 'hi')
            selected_arm: Optional arm that was selected

        Returns:
            Violation amount V(t) = max(0, B(t) - τ)
        """
        self.timestep += 1

        # Compute violation
        violation = max(0.0, bias_score - self.threshold)
        self.cumulative_violation += violation

        # Record violation event
        if violation > 0:
            self.violation_events.append(self.timestep)

        # Per-group tracking
        if self.track_by_category and bias_category:
            self.category_violations[bias_category].append(violation)

        if self.track_by_language and language:
            self.language_violations[language].append(violation)

        if selected_arm is not None:
            self.arm_violations[selected_arm].append(violation)

        # Create entry
        entry = ViolationEntry(
            timestep=self.timestep,
            bias_score=bias_score,
            threshold=self.threshold,
            violation=violation,
            cumulative_violation=self.cumulative_violation,
            bias_category=bias_category,
            language=language,
            selected_arm=selected_arm
        )

        self.history.append(entry)

        return violation

    def get_cumulative_violation(self) -> float:
        """Get cumulative fairness violation V(T)."""
        return self.cumulative_violation

    def get_violation_rate(self) -> float:
        """Get violation rate (proportion of timesteps with violation > 0)."""
        if self.timestep == 0:
            return 0.0
        return len(self.violation_events) / self.timestep

    def get_average_violation(self) -> float:
        """Get average violation V(T)/T."""
        if self.timestep == 0:
            return 0.0
        return self.cumulative_violation / self.timestep

    def get_violation_over_time(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get cumulative violation trajectory.

        Returns:
            Tuple of (timesteps, cumulative_violations)
        """
        if not self.history:
            return np.array([]), np.array([])

        timesteps = np.array([e.timestep for e in self.history])
        cumulative = np.array([e.cumulative_violation for e in self.history])

        return timesteps, cumulative

    def get_category_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get violation statistics per bias category."""
        stats = {}
        for category, violations in self.category_violations.items():
            if violations:
                stats[category] = {
                    'count': len(violations),
                    'total_violation': sum(violations),
                    'mean_violation': np.mean(violations),
                    'violation_rate': sum(1 for v in violations if v > 0) / len(violations),
                    'max_violation': max(violations)
                }
        return stats

    def get_language_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get violation statistics per language."""
        stats = {}
        for language, violations in self.language_violations.items():
            if violations:
                stats[language] = {
                    'count': len(violations),
                    'total_violation': sum(violations),
                    'mean_violation': np.mean(violations),
                    'violation_rate': sum(1 for v in violations if v > 0) / len(violations),
                    'max_violation': max(violations)
                }
        return stats

    def get_arm_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get violation statistics per arm."""
        stats = {}
        for arm, violations in self.arm_violations.items():
            if violations:
                stats[arm] = {
                    'count': len(violations),
                    'total_violation': sum(violations),
                    'mean_violation': np.mean(violations),
                    'violation_rate': sum(1 for v in violations if v > 0) / len(violations)
                }
        return stats

    def is_constraint_satisfied(self, c: float = 1.0) -> bool:
        """
        Check if fairness constraint is satisfied.

        The constraint V(T) ≤ c√T should hold for sublinear violations.

        Args:
            c: Constant multiplier for the bound

        Returns:
            True if constraint is satisfied
        """
        if self.timestep < 10:
            return True  # Not enough data

        bound = c * np.sqrt(self.timestep)
        return self.cumulative_violation <= bound

    def get_constraint_margin(self, c: float = 1.0) -> float:
        """
        Get margin to the constraint bound.

        Positive means under the bound (good), negative means over (bad).
        """
        if self.timestep == 0:
            return float('inf')

        bound = c * np.sqrt(self.timestep)
        return bound - self.cumulative_violation

    def get_fairness_statistics(self) -> Dict[str, float]:
        """Get comprehensive fairness statistics."""
        if not self.history:
            return {}

        violations = np.array([e.violation for e in self.history])
        bias_scores = np.array([e.bias_score for e in self.history])

        return {
            'threshold': self.threshold,
            'timesteps': self.timestep,
            'cumulative_violation': self.cumulative_violation,
            'average_violation': self.get_average_violation(),
            'violation_rate': self.get_violation_rate(),
            'total_violation_events': len(self.violation_events),
            'mean_bias_score': float(np.mean(bias_scores)),
            'max_bias_score': float(np.max(bias_scores)),
            'constraint_satisfied_c1': self.is_constraint_satisfied(c=1.0),
            'constraint_satisfied_c2': self.is_constraint_satisfied(c=2.0),
            'constraint_margin_c1': self.get_constraint_margin(c=1.0),
            'violation_per_sqrt_t': self.cumulative_violation / np.sqrt(self.timestep) if self.timestep > 0 else 0,
        }

    def get_worst_categories(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get categories with highest violation rates."""
        cat_stats = self.get_category_statistics()
        sorted_cats = sorted(
            cat_stats.items(),
            key=lambda x: x[1]['mean_violation'],
            reverse=True
        )
        return [(cat, stats['mean_violation']) for cat, stats in sorted_cats[:top_k]]

    def save(self, filepath: str):
        """Save violation history to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'threshold': self.threshold,
            'timestep': self.timestep,
            'cumulative_violation': self.cumulative_violation,
            'statistics': self.get_fairness_statistics(),
            'category_statistics': self.get_category_statistics(),
            'language_statistics': self.get_language_statistics(),
            'history': [
                {
                    'timestep': e.timestep,
                    'bias_score': e.bias_score,
                    'violation': e.violation,
                    'cumulative_violation': e.cumulative_violation,
                    'bias_category': e.bias_category,
                    'language': e.language,
                }
                for e in self.history
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved violation history to {path}")

    @classmethod
    def load(cls, filepath: str) -> 'FairnessViolationTracker':
        """Load tracker from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        tracker = cls(threshold=data['threshold'])
        tracker.timestep = data['timestep']
        tracker.cumulative_violation = data['cumulative_violation']

        for entry_data in data['history']:
            entry = ViolationEntry(
                timestep=entry_data['timestep'],
                bias_score=entry_data['bias_score'],
                threshold=data['threshold'],
                violation=entry_data['violation'],
                cumulative_violation=entry_data['cumulative_violation'],
                bias_category=entry_data.get('bias_category'),
                language=entry_data.get('language')
            )
            tracker.history.append(entry)

            if entry.bias_category:
                tracker.category_violations[entry.bias_category].append(entry.violation)
            if entry.language:
                tracker.language_violations[entry.language].append(entry.violation)

        return tracker

    def __repr__(self) -> str:
        satisfied = "✓" if self.is_constraint_satisfied() else "✗"
        return (f"FairnessViolationTracker(τ={self.threshold}, T={self.timestep}, "
                f"V(T)={self.cumulative_violation:.4f}, rate={self.get_violation_rate():.2%} {satisfied})")
