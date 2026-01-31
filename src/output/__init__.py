"""
Output standardization module for Fair-CB.

Provides:
- CSV output manager with full-form column names
- Consistent formatting across all result files
- Validation utilities
"""

from .csv_manager import (
    CSVOutputManager,
    format_column_name,
    validate_csv_columns,
    STANDARD_COLUMNS,
)

__all__ = [
    'CSVOutputManager',
    'format_column_name',
    'validate_csv_columns',
    'STANDARD_COLUMNS',
]
