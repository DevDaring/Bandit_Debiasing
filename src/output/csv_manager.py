"""
CSV Output Manager with standardized full-form column names.

CRITICAL: All CSV outputs MUST use full-form column names.
Examples:
- "English" not "en"
- "Gender Identity" not "gender"
- "Intersectional Bias Reduction (IBR)" not "ibr"

This ensures publication-ready CSV files that don't require
post-processing for figures and tables.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# STANDARD COLUMN DEFINITIONS
# ============================================================================

STANDARD_COLUMNS = {
    # Model columns
    'model': 'Model Name',
    'model_id': 'Model Identifier',
    'model_params': 'Model Parameters',
    'model_family': 'Model Family',

    # Language columns
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'language': 'Language',
    'lang': 'Language',

    # Bias category columns
    'gender': 'Gender Identity',
    'race': 'Race and Ethnicity',
    'race-color': 'Race and Color',
    'caste': 'Caste System',
    'religion': 'Religious Belief',
    'socioeconomic': 'Socioeconomic Status',
    'nationality': 'National Origin',
    'age': 'Age Group',
    'sexual_orientation': 'Sexual Orientation',
    'sexual-orientation': 'Sexual Orientation',
    'physical_appearance': 'Physical Appearance',
    'physical-appearance': 'Physical Appearance',
    'disability': 'Disability Status',
    'bias_type': 'Bias Category',
    'category': 'Bias Category',

    # Metric columns
    'ibr': 'Intersectional Bias Reduction (IBR)',
    'far': 'Fairness-Aware Regret (FAR)',
    'bias_score': 'Bias Score',
    'quality_score': 'Output Quality Score',
    'reward': 'Reward',
    'regret': 'Cumulative Regret',
    'violation': 'Fairness Violation',
    'avg_bias': 'Average Bias Score',
    'avg_reward': 'Average Reward',

    # Arm columns
    'arm': 'Selected Arm',
    'arm_0': 'No Intervention (Baseline)',
    'arm_1': 'Gender Steering Vector',
    'arm_2': 'Race Steering Vector',
    'arm_3': 'Religion Steering Vector',
    'arm_4': 'Prompt Prefix Debiasing',
    'arm_5': 'Output Adjustment',
    'selected_arm': 'Selected Debiasing Strategy',

    # Bandit columns
    'bandit': 'Bandit Algorithm',
    'linucb': 'Linear Upper Confidence Bound (LinUCB)',
    'thompson': 'Thompson Sampling',
    'neural': 'Neural Contextual Bandit',

    # Statistical columns
    'mean': 'Mean',
    'std': 'Standard Deviation',
    'ci_lower': '95% Confidence Interval (Lower)',
    'ci_upper': '95% Confidence Interval (Upper)',
    'p_value': 'Statistical Significance (p-value)',
    'cohens_d': 'Effect Size (Cohen\'s d)',
    'is_significant': 'Is Statistically Significant (p < 0.05)',

    # Experiment columns
    'seed': 'Random Seed',
    'epoch': 'Training Epoch',
    'step': 'Training Step',
    'timestep': 'Timestep',
    'n_samples': 'Number of Samples',
    'dataset': 'Dataset Name',
    'split': 'Data Split',

    # Cross-lingual columns
    'source_lang': 'Source Language',
    'target_lang': 'Target Language',
    'transfer_ratio': 'Transfer Efficacy Ratio',

    # Ablation columns
    'config': 'Configuration',
    'ablation': 'Ablation Setting',
    'component': 'Component Name',
    'enabled': 'Component Enabled',
}


def format_column_name(short_name: str) -> str:
    """
    Convert short column name to full form.

    Args:
        short_name: Short form like 'ibr', 'en', 'gender'

    Returns:
        Full form like 'Intersectional Bias Reduction (IBR)', 'English'
    """
    # Check direct mapping
    if short_name.lower() in STANDARD_COLUMNS:
        return STANDARD_COLUMNS[short_name.lower()]

    # Check with underscores converted to hyphens
    hyphenated = short_name.replace('_', '-').lower()
    if hyphenated in STANDARD_COLUMNS:
        return STANDARD_COLUMNS[hyphenated]

    # Handle compound names (e.g., 'bias_score_en' -> 'Bias Score - English')
    parts = short_name.replace('-', '_').split('_')
    formatted_parts = []
    for part in parts:
        if part.lower() in STANDARD_COLUMNS:
            formatted_parts.append(STANDARD_COLUMNS[part.lower()])
        else:
            formatted_parts.append(part.capitalize())

    if len(formatted_parts) > 1:
        return ' - '.join(formatted_parts)

    # If no mapping found, capitalize first letter of each word
    return ' '.join(word.capitalize() for word in short_name.split('_'))


def validate_csv_columns(df: pd.DataFrame) -> List[str]:
    """
    Check for column names that violate the full-form convention.

    Args:
        df: DataFrame to validate

    Returns:
        List of problematic column names
    """
    violations = []

    for col in df.columns:
        col_lower = col.lower().replace('-', '_')

        # Check if it's a known short form
        if col_lower in STANDARD_COLUMNS and col != STANDARD_COLUMNS[col_lower]:
            violations.append(col)

        # Check for common abbreviation patterns
        if col in ['en', 'hi', 'bn', 'ibr', 'far', 'std', 'avg']:
            violations.append(col)

    return violations


class CSVOutputManager:
    """
    Centralized CSV output manager with standardized column names.

    Ensures all output CSVs use full-form column names for
    publication-ready results.
    """

    def __init__(
        self,
        output_dir: str = './results',
        timestamp_files: bool = True,
        validate_columns: bool = True
    ):
        """
        Initialize CSV output manager.

        Args:
            output_dir: Directory for output files
            timestamp_files: Whether to add timestamps to filenames
            validate_columns: Whether to validate column names before saving
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp_files = timestamp_files
        self.validate_columns = validate_columns

        self.saved_files: List[Path] = []

    def format_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all column names to full form.

        Args:
            df: DataFrame with possibly short column names

        Returns:
            DataFrame with full-form column names
        """
        new_columns = {}
        for col in df.columns:
            new_columns[col] = format_column_name(col)

        return df.rename(columns=new_columns)

    def save_main_results(
        self,
        df: pd.DataFrame,
        filename: str = 'main_results'
    ) -> Path:
        """
        Save main experiment results.

        Args:
            df: Results DataFrame
            filename: Base filename (without extension)

        Returns:
            Path to saved file
        """
        return self._save_dataframe(df, filename, 'main_results')

    def save_per_category_results(
        self,
        df: pd.DataFrame,
        filename: str = 'per_category_results'
    ) -> Path:
        """
        Save per-bias-category results.

        Args:
            df: Per-category results DataFrame
            filename: Base filename

        Returns:
            Path to saved file
        """
        return self._save_dataframe(df, filename, 'per_category')

    def save_per_language_results(
        self,
        df: pd.DataFrame,
        filename: str = 'per_language_results'
    ) -> Path:
        """
        Save per-language results.

        Args:
            df: Per-language results DataFrame
            filename: Base filename

        Returns:
            Path to saved file
        """
        return self._save_dataframe(df, filename, 'per_language')

    def save_ablation_results(
        self,
        df: pd.DataFrame,
        filename: str = 'ablation_results'
    ) -> Path:
        """
        Save ablation study results.

        Args:
            df: Ablation results DataFrame
            filename: Base filename

        Returns:
            Path to saved file
        """
        return self._save_dataframe(df, filename, 'ablation')

    def save_crosslingual_results(
        self,
        df: pd.DataFrame,
        filename: str = 'crosslingual_results'
    ) -> Path:
        """
        Save cross-lingual transfer results.

        Args:
            df: Cross-lingual results DataFrame
            filename: Base filename

        Returns:
            Path to saved file
        """
        return self._save_dataframe(df, filename, 'crosslingual')

    def save_theory_verification(
        self,
        df: pd.DataFrame,
        filename: str = 'theory_verification'
    ) -> Path:
        """
        Save theoretical verification results.

        Args:
            df: Theory verification DataFrame
            filename: Base filename

        Returns:
            Path to saved file
        """
        return self._save_dataframe(df, filename, 'theory')

    def _save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        category: str
    ) -> Path:
        """
        Internal method to save DataFrame with standardized columns.

        Args:
            df: DataFrame to save
            filename: Base filename
            category: Result category for organization

        Returns:
            Path to saved file
        """
        # Format column names
        formatted_df = self.format_dataframe_columns(df)

        # Validate if enabled
        if self.validate_columns:
            violations = validate_csv_columns(formatted_df)
            if violations:
                logger.warning(f"Column name violations found: {violations}")

        # Create subdirectory for category
        category_dir = self.output_dir / category
        category_dir.mkdir(exist_ok=True)

        # Generate filename
        if self.timestamp_files:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            full_filename = f"{filename}_{timestamp}.csv"
        else:
            full_filename = f"{filename}.csv"

        filepath = category_dir / full_filename

        # Save
        formatted_df.to_csv(filepath, index=False, encoding='utf-8')
        self.saved_files.append(filepath)

        logger.info(f"Saved: {filepath}")

        return filepath

    def create_summary_csv(
        self,
        results: Dict[str, Any],
        filename: str = 'summary'
    ) -> Path:
        """
        Create summary CSV from dictionary of results.

        Args:
            results: Dictionary of metric name -> value
            filename: Base filename

        Returns:
            Path to saved file
        """
        # Format keys as full-form column names
        formatted_results = {}
        for key, value in results.items():
            formatted_key = format_column_name(key)
            formatted_results[formatted_key] = [value]

        df = pd.DataFrame(formatted_results)
        return self._save_dataframe(df, filename, 'summary')

    def merge_and_save(
        self,
        dataframes: List[pd.DataFrame],
        filename: str,
        category: str = 'merged'
    ) -> Path:
        """
        Merge multiple DataFrames and save.

        Args:
            dataframes: List of DataFrames to merge
            filename: Output filename
            category: Result category

        Returns:
            Path to saved file
        """
        merged = pd.concat(dataframes, ignore_index=True)
        return self._save_dataframe(merged, filename, category)

    def get_saved_files(self) -> List[Path]:
        """Get list of all files saved by this manager."""
        return self.saved_files.copy()

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir

    def __repr__(self) -> str:
        return f"CSVOutputManager(output_dir='{self.output_dir}', files_saved={len(self.saved_files)})"
