"""
Detect demographic markers in input text.
Identifies presence of gender, race/ethnicity, religion, age indicators.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DemographicDetector:
    """
    Detect demographic markers in text.

    Supports English, Hindi, Bengali through keyword dictionaries.
    """

    def __init__(self):
        """Initialize with demographic marker dictionaries."""
        self.gender_markers = self._load_gender_markers()
        self.race_markers = self._load_race_markers()
        self.religion_markers = self._load_religion_markers()
        self.age_markers = self._load_age_markers()

    def _load_gender_markers(self) -> Dict:
        """Load gender marker dictionaries for all languages."""
        return {
            "en": {
                "male": ["he", "him", "his", "man", "men", "boy", "boys", "father", "dad", "husband", "mr", "gentleman", "brother", "son", "male"],
                "female": ["she", "her", "hers", "woman", "women", "girl", "girls", "mother", "mom", "wife", "mrs", "ms", "lady", "sister", "daughter", "female"],
                "neutral": ["they", "them", "their", "person", "people", "individual", "someone", "anyone"],
            },
            "hi": {
                "male": ["वह", "उसका", "आदमी", "लड़का", "पिता", "पति", "भाई", "बेटा"],
                "female": ["वह", "उसकी", "औरत", "लड़की", "माता", "पत्नी", "बहन", "बेटी"],
                "neutral": ["व्यक्ति", "लोग", "कोई"],
            },
            "bn": {
                "male": ["সে", "তার", "পুরুষ", "ছেলে", "বাবা", "স্বামী", "ভাই", "ছেলে"],
                "female": ["সে", "তার", "মহিলা", "মেয়ে", "মা", "স্ত্রী", "বোন", "মেয়ে"],
                "neutral": ["ব্যক্তি", "মানুষ", "কেউ"],
            },
        }

    def _load_race_markers(self) -> Dict:
        """Load race/ethnicity marker dictionaries."""
        return {
            "en": ["asian", "african", "european", "indian", "chinese", "black", "white", "hispanic", "latino", "caucasian", "ethnicity", "race"],
            "hi": ["एशियाई", "अफ्रीकी", "यूरोपीय", "भारतीय", "चीनी", "जाति", "नस्ल"],
            "bn": ["এশীয়", "আফ্রিকান", "ইউরোপীয়", "ভারতীয়", "চীনা", "জাতি", "বংশ"],
        }

    def _load_religion_markers(self) -> Dict:
        """Load religion marker dictionaries."""
        return {
            "en": ["christian", "muslim", "hindu", "buddhist", "jewish", "sikh", "church", "mosque", "temple", "synagogue", "religion", "faith"],
            "hi": ["ईसाई", "मुस्लिम", "हिंदू", "बौद्ध", "यहूदी", "सिख", "चर्च", "मस्जिद", "मंदिर", "धर्म"],
            "bn": ["খ্রিস্টান", "মুসলিম", "হিন্দু", "বৌদ্ধ", "ইহুদি", "শিখ", "চার্চ", "মসজিদ", "মন্দির", "ধর্ম"],
        }

    def _load_age_markers(self) -> Dict:
        """Load age marker dictionaries."""
        return {
            "en": {
                "young": ["young", "youth", "teenager", "child", "kid", "adolescent", "junior"],
                "old": ["old", "elderly", "senior", "aged", "elder", "retired"],
            },
            "hi": {"young": ["युवा", "किशोर", "बच्चा"], "old": ["बूढ़ा", "बुजुर्ग", "वृद्ध"]},
            "bn": {"young": ["যুবক", "কিশোর", "শিশু"], "old": ["বৃদ্ধ", "প্রবীণ", "বয়স্ক"]},
        }

    def detect(self, text: str, language: str = "en") -> Dict:
        """
        Detect all demographic markers in text.

        Args:
            text: Input text
            language: Language code (en/hi/bn)

        Returns:
            Dict with detected markers per category
        """
        text_lower = text.lower()

        # Gender detection
        gender = self._detect_gender(text_lower, language)

        # Race detection
        race_count = self._detect_race(text_lower, language)

        # Religion detection
        religion_count = self._detect_religion(text_lower, language)

        # Age detection
        age = self._detect_age(text_lower, language)

        # Has names
        has_names = self._detect_names(text)

        # Demographic density
        total_markers = gender["count"] + race_count + religion_count + age["count"]
        text_length = max(len(text.split()), 1)
        demographic_density = min(total_markers / text_length, 1.0)

        return {
            "gender": gender,
            "race_count": race_count,
            "religion_count": religion_count,
            "age": age,
            "has_names": has_names,
            "demographic_density": demographic_density,
        }

    def _detect_gender(self, text: str, language: str) -> Dict:
        """Detect gender markers."""
        markers = self.gender_markers.get(language, self.gender_markers["en"])

        male_count = sum(text.count(word) for word in markers["male"])
        female_count = sum(text.count(word) for word in markers["female"])
        neutral_count = sum(text.count(word) for word in markers["neutral"])

        total = male_count + female_count + neutral_count
        if total == 0:
            return {"male": 0.0, "female": 0.0, "neutral": 0.0, "count": 0}

        return {
            "male": male_count / total,
            "female": female_count / total,
            "neutral": neutral_count / total,
            "count": total,
        }

    def _detect_race(self, text: str, language: str) -> int:
        """Detect race/ethnicity markers."""
        markers = self.race_markers.get(language, self.race_markers["en"])
        return sum(text.count(word) for word in markers)

    def _detect_religion(self, text: str, language: str) -> int:
        """Detect religion markers."""
        markers = self.religion_markers.get(language, self.religion_markers["en"])
        return sum(text.count(word) for word in markers)

    def _detect_age(self, text: str, language: str) -> Dict:
        """Detect age markers."""
        markers = self.age_markers.get(language, self.age_markers["en"])

        young_count = sum(text.count(word) for word in markers["young"])
        old_count = sum(text.count(word) for word in markers["old"])

        total = young_count + old_count
        if total == 0:
            return {"young": 0.0, "old": 0.0, "neutral": 1.0, "count": 0}

        return {
            "young": young_count / total,
            "old": old_count / total,
            "neutral": 0.0,
            "count": total,
        }

    def _detect_names(self, text: str) -> bool:
        """Detect if text contains capitalized names (simple heuristic)."""
        # Simple check for capitalized words that might be names
        words = text.split()
        capitalized = [w for w in words if w and w[0].isupper() and len(w) > 2]
        return len(capitalized) > 0

    def get_feature_vector(self, text: str, language: str = "en") -> List[float]:
        """
        Get feature vector for bandit context.

        Returns:
            12-dim vector: [gender_male, gender_female, gender_neutral,
                           race_detected, race_count_normalized,
                           religion_detected, religion_count_normalized,
                           age_young, age_old, age_neutral,
                           has_names, demographic_density]
        """
        result = self.detect(text, language)

        # Normalize counts
        race_norm = min(result["race_count"] / 10.0, 1.0)
        religion_norm = min(result["religion_count"] / 10.0, 1.0)

        return [
            result["gender"]["male"],
            result["gender"]["female"],
            result["gender"]["neutral"],
            1.0 if result["race_count"] > 0 else 0.0,
            race_norm,
            1.0 if result["religion_count"] > 0 else 0.0,
            religion_norm,
            result["age"]["young"],
            result["age"]["old"],
            result["age"]["neutral"],
            1.0 if result["has_names"] else 0.0,
            result["demographic_density"],
        ]
