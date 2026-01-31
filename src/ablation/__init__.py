"""
Ablation study framework for Fair-CB.

Provides systematic testing of component contributions:
1. Configuration generator for ablation settings
2. Ablation runner for automated experiments
3. Results analyzer for component importance
"""

from .config_generator import (
    AblationConfig,
    AblationConfigGenerator,
    STANDARD_ABLATION_CONFIGS,
)
from .ablation_runner import (
    AblationRunner,
    AblationResult,
)
from .results_analyzer import (
    AblationResultsAnalyzer,
    ComponentImportance,
)

__all__ = [
    'AblationConfig',
    'AblationConfigGenerator',
    'STANDARD_ABLATION_CONFIGS',
    'AblationRunner',
    'AblationResult',
    'AblationResultsAnalyzer',
    'ComponentImportance',
]
