"""
Code-mixing detection and handling for multilingual inputs.

Handles:
- Hindi-English code-mixing (Hinglish)
- Bengali-English code-mixing
- Mixed-script detection (Devanagari + Latin)
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Unicode script ranges
LATIN_PATTERN = re.compile(r'[A-Za-z]+')
DEVANAGARI_PATTERN = re.compile(r'[\u0900-\u097F]+')
BENGALI_PATTERN = re.compile(r'[\u0980-\u09FF]+')


@dataclass
class CodeMixingResult:
    """Result of code-mixing analysis."""
    is_code_mixed: bool
    primary_language: str
    secondary_language: Optional[str]
    mixing_ratio: float  # ratio of secondary language tokens
    languages_detected: List[str]
    script_counts: Dict[str, int]


class CodeMixingDetector:
    """
    Detect code-mixing in multilingual text.

    Identifies:
    - Script-level mixing (e.g., Devanagari + Latin)
    - Language-level mixing (e.g., Hindi + English words)
    """

    def __init__(
        self,
        english_words: Optional[Set[str]] = None,
        mixing_threshold: float = 0.1
    ):
        """
        Initialize code-mixing detector.

        Args:
            english_words: Set of common English words for detection
            mixing_threshold: Minimum ratio to consider as code-mixed
        """
        self.mixing_threshold = mixing_threshold

        # Common English words for detection
        self.english_words = english_words or {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'this', 'that', 'these', 'those', 'it', 'they', 'we', 'you',
            'he', 'she', 'and', 'or', 'but', 'if', 'so', 'because',
            'not', 'no', 'yes', 'all', 'some', 'any', 'each', 'every',
            'very', 'just', 'also', 'only', 'then', 'now', 'here', 'there',
        }

    def count_scripts(self, text: str) -> Dict[str, int]:
        """
        Count tokens by script type.

        Args:
            text: Input text

        Returns:
            Dict mapping script name to token count
        """
        counts = {
            'latin': 0,
            'devanagari': 0,
            'bengali': 0,
            'other': 0
        }

        # Split by whitespace and basic punctuation
        tokens = re.findall(r'\S+', text)

        for token in tokens:
            # Clean token
            clean = re.sub(r'[^\w\u0900-\u097F\u0980-\u09FF]', '', token)
            if not clean:
                continue

            # Detect script
            if DEVANAGARI_PATTERN.search(clean):
                counts['devanagari'] += 1
            elif BENGALI_PATTERN.search(clean):
                counts['bengali'] += 1
            elif LATIN_PATTERN.match(clean):
                counts['latin'] += 1
            else:
                counts['other'] += 1

        return counts

    def detect(self, text: str) -> CodeMixingResult:
        """
        Detect code-mixing in text.

        Args:
            text: Input text to analyze

        Returns:
            CodeMixingResult with detection details
        """
        script_counts = self.count_scripts(text)
        total_tokens = sum(script_counts.values())

        if total_tokens == 0:
            return CodeMixingResult(
                is_code_mixed=False,
                primary_language='unknown',
                secondary_language=None,
                mixing_ratio=0.0,
                languages_detected=[],
                script_counts=script_counts
            )

        # Determine primary and secondary languages
        sorted_scripts = sorted(
            [(k, v) for k, v in script_counts.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )

        if not sorted_scripts:
            primary = 'unknown'
            secondary = None
            mixing_ratio = 0.0
        elif len(sorted_scripts) == 1:
            primary = sorted_scripts[0][0]
            secondary = None
            mixing_ratio = 0.0
        else:
            primary = sorted_scripts[0][0]
            secondary = sorted_scripts[1][0]
            mixing_ratio = sorted_scripts[1][1] / total_tokens

        # Map scripts to languages
        script_to_lang = {
            'latin': 'en',
            'devanagari': 'hi',
            'bengali': 'bn'
        }

        primary_lang = script_to_lang.get(primary, primary)
        secondary_lang = script_to_lang.get(secondary, secondary) if secondary else None

        languages_detected = [script_to_lang.get(s, s) for s, c in sorted_scripts if c > 0]

        return CodeMixingResult(
            is_code_mixed=mixing_ratio >= self.mixing_threshold,
            primary_language=primary_lang,
            secondary_language=secondary_lang,
            mixing_ratio=mixing_ratio,
            languages_detected=languages_detected,
            script_counts=script_counts
        )

    def is_hinglish(self, text: str) -> bool:
        """Check if text is Hindi-English code-mixed (Hinglish)."""
        result = self.detect(text)
        return (
            result.is_code_mixed and
            set(result.languages_detected) == {'en', 'hi'}
        )

    def is_benglish(self, text: str) -> bool:
        """Check if text is Bengali-English code-mixed."""
        result = self.detect(text)
        return (
            result.is_code_mixed and
            set(result.languages_detected) == {'en', 'bn'}
        )


class CodeMixingHandler:
    """
    Handle code-mixed inputs for debiasing.

    Strategies:
    1. Apply steering vectors for both detected languages
    2. Weight by language proportion
    3. Use language-specific prompts
    """

    def __init__(self, detector: Optional[CodeMixingDetector] = None):
        """
        Initialize handler.

        Args:
            detector: Code-mixing detector instance
        """
        self.detector = detector or CodeMixingDetector()

    def get_steering_weights(self, text: str) -> Dict[str, float]:
        """
        Get steering vector weights based on code-mixing.

        For code-mixed text, weight steering vectors by language proportion.

        Args:
            text: Input text

        Returns:
            Dict mapping language to weight (sum to 1.0)
        """
        result = self.detector.detect(text)

        if not result.is_code_mixed:
            # Single language
            return {result.primary_language: 1.0}

        # Weight by proportion
        total = sum(result.script_counts.values())
        if total == 0:
            return {result.primary_language: 1.0}

        weights = {}
        script_to_lang = {
            'latin': 'en',
            'devanagari': 'hi',
            'bengali': 'bn'
        }

        for script, count in result.script_counts.items():
            if count > 0:
                lang = script_to_lang.get(script, script)
                weights[lang] = count / total

        return weights

    def get_prompt_prefix(self, text: str) -> str:
        """
        Generate appropriate prompt prefix for code-mixed text.

        Args:
            text: Input text

        Returns:
            Language-aware prompt prefix
        """
        result = self.detector.detect(text)

        if result.is_code_mixed:
            langs = ' and '.join(result.languages_detected)
            return f"The following text is code-mixed ({langs}). Please respond neutrally and without bias: "
        else:
            return "Please respond neutrally and without bias: "

    def should_apply_multilingual_steering(self, text: str) -> bool:
        """
        Check if multilingual steering should be applied.

        Args:
            text: Input text

        Returns:
            True if multiple languages detected and should use multi-steering
        """
        result = self.detector.detect(text)
        return result.is_code_mixed and result.mixing_ratio >= 0.2

    def __repr__(self) -> str:
        return f"CodeMixingHandler(threshold={self.detector.mixing_threshold})"
