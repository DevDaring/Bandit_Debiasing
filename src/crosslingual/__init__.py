"""
Cross-lingual analysis module for Fair-CB.

Analyzes how debiasing transfers across languages:
- English → Hindi transfer
- English → Bengali transfer
- Cross-lingual steering vector efficacy
- Code-mixing handling
"""

from .transfer_analyzer import (
    TransferAnalyzer,
    TransferResult,
    compute_transfer_ratio,
)
from .code_mixing_handler import (
    CodeMixingDetector,
    CodeMixingHandler,
)
from .parallel_evaluator import (
    ParallelEvaluator,
    CrossLingualResult,
)

__all__ = [
    'TransferAnalyzer',
    'TransferResult',
    'compute_transfer_ratio',
    'CodeMixingDetector',
    'CodeMixingHandler',
    'ParallelEvaluator',
    'CrossLingualResult',
]
