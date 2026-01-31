"""
Counterfactual evaluation for Fair-CB.

Provides true regret computation by running ALL arms on evaluation samples.
This addresses Concern 1.1 from Concerns.md about oracle regret computation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of running all arms on a single sample."""
    sample_id: int
    text: str
    category: str
    language: str
    arm_results: Dict[int, Dict[str, float]]  # arm_id -> {bias, reward, quality}
    optimal_arm: int
    optimal_reward: float
    selected_arm: int
    selected_reward: float
    instantaneous_regret: float


@dataclass
class CounterfactualSummary:
    """Summary of counterfactual evaluation."""
    n_samples: int
    true_cumulative_regret: float
    true_average_regret: float
    arm_optimality_count: Dict[int, int]  # How often each arm was optimal
    best_static_arm: int
    best_static_regret: float
    adaptive_vs_static_ratio: float


class CounterfactualEvaluator:
    """
    Evaluator that runs ALL debiasing arms on each sample.
    
    This is required for:
    1. True regret computation (not estimated)
    2. Verifying adaptive vs static claims
    3. Finding the true best static arm
    
    WARNING: This is 6x slower than normal evaluation since all arms are run.
    Use on a subset of eval data for publication results.
    """
    
    def __init__(
        self,
        n_arms: int = 6,
        arm_names: Optional[Dict[int, str]] = None
    ):
        self.n_arms = n_arms
        self.arm_names = arm_names or {
            0: "No Intervention",
            1: "Gender Steering",
            2: "Race Steering",
            3: "Religion Steering",
            4: "Prompt Prefix",
            5: "Output Adjustment"
        }
        
        self.results: List[CounterfactualResult] = []
    
    def evaluate_sample_all_arms(
        self,
        sample_id: int,
        text: str,
        category: str,
        language: str,
        arm_executor: callable,
        selected_arm: int
    ) -> CounterfactualResult:
        """
        Run all arms on a single sample and compute true regret.
        
        Args:
            sample_id: Unique identifier for sample
            text: Input text to evaluate
            category: Bias category
            language: Language of text
            arm_executor: Function that takes (text, arm_id) and returns
                          {'bias_score': float, 'reward': float, 'quality': float}
            selected_arm: The arm that was actually selected by bandit
            
        Returns:
            CounterfactualResult with all arm performances
        """
        arm_results = {}
        
        for arm_id in range(self.n_arms):
            try:
                result = arm_executor(text, arm_id)
                arm_results[arm_id] = {
                    'bias_score': result.get('bias_score', 0.5),
                    'reward': result.get('reward', 0.5),
                    'quality': result.get('quality', 0.5)
                }
            except Exception as e:
                logger.warning(f"Arm {arm_id} failed on sample {sample_id}: {e}")
                arm_results[arm_id] = {
                    'bias_score': 1.0,  # Worst bias
                    'reward': 0.0,      # No reward
                    'quality': 0.0
                }
        
        # Find optimal arm (highest reward)
        optimal_arm = max(arm_results.keys(), key=lambda a: arm_results[a]['reward'])
        optimal_reward = arm_results[optimal_arm]['reward']
        
        selected_reward = arm_results[selected_arm]['reward']
        instantaneous_regret = optimal_reward - selected_reward
        
        result = CounterfactualResult(
            sample_id=sample_id,
            text=text[:100] + '...' if len(text) > 100 else text,
            category=category,
            language=language,
            arm_results=arm_results,
            optimal_arm=optimal_arm,
            optimal_reward=optimal_reward,
            selected_arm=selected_arm,
            selected_reward=selected_reward,
            instantaneous_regret=instantaneous_regret
        )
        
        self.results.append(result)
        return result
    
    def evaluate_dataset(
        self,
        samples: List[Dict],
        arm_executor: callable,
        bandit_selector: callable,
        show_progress: bool = True
    ) -> CounterfactualSummary:
        """
        Evaluate entire dataset with counterfactual arm execution.
        
        Args:
            samples: List of samples with 'text', 'category', 'language' keys
            arm_executor: Function(text, arm_id) -> {'reward': float, ...}
            bandit_selector: Function(text) -> selected_arm
            show_progress: Show tqdm progress bar
            
        Returns:
            CounterfactualSummary
        """
        iterator = tqdm(samples, desc="Counterfactual Evaluation") if show_progress else samples
        
        for i, sample in enumerate(iterator):
            text = sample.get('text', sample.get('sentence', ''))
            category = sample.get('category', sample.get('bias_type', 'unknown'))
            language = sample.get('language', 'en')
            
            selected_arm = bandit_selector(text)
            
            self.evaluate_sample_all_arms(
                sample_id=i,
                text=text,
                category=category,
                language=language,
                arm_executor=arm_executor,
                selected_arm=selected_arm
            )
        
        return self.get_summary()
    
    def get_summary(self) -> CounterfactualSummary:
        """Compute summary statistics from all results."""
        if not self.results:
            return CounterfactualSummary(
                n_samples=0,
                true_cumulative_regret=0.0,
                true_average_regret=0.0,
                arm_optimality_count={i: 0 for i in range(self.n_arms)},
                best_static_arm=0,
                best_static_regret=0.0,
                adaptive_vs_static_ratio=0.0
            )
        
        n_samples = len(self.results)
        
        # True cumulative regret
        true_cumulative_regret = sum(r.instantaneous_regret for r in self.results)
        true_average_regret = true_cumulative_regret / n_samples
        
        # How often each arm was optimal
        arm_optimality_count = {i: 0 for i in range(self.n_arms)}
        for r in self.results:
            arm_optimality_count[r.optimal_arm] += 1
        
        # Best static arm (highest average reward across all samples)
        arm_avg_rewards = {}
        for arm_id in range(self.n_arms):
            rewards = [r.arm_results[arm_id]['reward'] for r in self.results]
            arm_avg_rewards[arm_id] = np.mean(rewards)
        
        best_static_arm = max(arm_avg_rewards.keys(), key=lambda a: arm_avg_rewards[a])
        
        # Static regret (regret of always using best static arm)
        best_static_regret = sum(
            r.optimal_reward - r.arm_results[best_static_arm]['reward']
            for r in self.results
        )
        
        # Adaptive vs static ratio
        # R_adaptive / R_static (should be < 1 if adaptive is better)
        adaptive_vs_static_ratio = (
            true_cumulative_regret / best_static_regret
            if best_static_regret > 0 else 0.0
        )
        
        return CounterfactualSummary(
            n_samples=n_samples,
            true_cumulative_regret=true_cumulative_regret,
            true_average_regret=true_average_regret,
            arm_optimality_count=arm_optimality_count,
            best_static_arm=best_static_arm,
            best_static_regret=best_static_regret,
            adaptive_vs_static_ratio=adaptive_vs_static_ratio
        )
    
    def get_arm_performance_table(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-arm performance statistics.
        
        Returns:
            Dict[arm_name] -> {avg_reward, avg_bias, times_optimal}
        """
        table = {}
        
        for arm_id in range(self.n_arms):
            arm_name = self.arm_names.get(arm_id, f"Arm {arm_id}")
            
            rewards = [r.arm_results[arm_id]['reward'] for r in self.results]
            biases = [r.arm_results[arm_id]['bias_score'] for r in self.results]
            times_optimal = sum(1 for r in self.results if r.optimal_arm == arm_id)
            times_selected = sum(1 for r in self.results if r.selected_arm == arm_id)
            
            table[arm_name] = {
                'Average Reward': np.mean(rewards) if rewards else 0.0,
                'Average Bias Score': np.mean(biases) if biases else 0.0,
                'Times Optimal': times_optimal,
                'Times Selected': times_selected,
                'Optimality Rate': times_optimal / len(self.results) if self.results else 0.0
            }
        
        return table
    
    def update_regret_tracker(self, regret_tracker):
        """
        Update a RegretTracker with true counterfactual data.
        
        Args:
            regret_tracker: RegretTracker instance to update
        """
        for r in self.results:
            all_arm_rewards = {arm: data['reward'] for arm, data in r.arm_results.items()}
            regret_tracker.update(
                selected_arm=r.selected_arm,
                reward=r.selected_reward,
                all_arm_rewards=all_arm_rewards
            )
    
    def reset(self):
        """Clear all stored results."""
        self.results = []
    
    def __len__(self) -> int:
        return len(self.results)
