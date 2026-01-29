"""
Unified dataset loader for IndiBias and Multi-CrowS-Pairs datasets.
Converts both formats to unified MAB pipeline format.

Datasets:
- IndiBias: https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
- Multi-CrowS-Pairs: https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs
"""

import pandas as pd
import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class Language(Enum):
    ENGLISH = "en"
    HINDI = "hi"
    BENGALI = "bn"

class BiasType(Enum):
    GENDER = "gender"
    RACE = "race"
    CASTE = "caste"
    RELIGION = "religion"
    SOCIOECONOMIC = "socioeconomic"
    NATIONALITY = "nationality"
    AGE = "age"
    SEXUAL_ORIENTATION = "sexual_orientation"
    PHYSICAL_APPEARANCE = "physical_appearance"
    DISABILITY = "disability"

class SourceDataset(Enum):
    INDIBIAS = "indibias"
    CROWSPAIRS = "crowspairs"

# Mapping from original bias types to unified types
BIAS_TYPE_NORMALIZATION = {
    # IndiBias
    "caste": BiasType.CASTE,
    "gender": BiasType.GENDER,
    "religion": BiasType.RELIGION,
    "race": BiasType.RACE,
    # CrowS-Pairs
    "race-color": BiasType.RACE,
    "socioeconomic": BiasType.SOCIOECONOMIC,
    "nationality": BiasType.NATIONALITY,
    "age": BiasType.AGE,
    "sexual-orientation": BiasType.SEXUAL_ORIENTATION,
    "physical-appearance": BiasType.PHYSICAL_APPEARANCE,
    "disability": BiasType.DISABILITY,
}

# Which MAB arm is recommended for each bias type
BIAS_TO_RECOMMENDED_ARM = {
    BiasType.GENDER: 1,              # Gender steering
    BiasType.RACE: 2,                # Race steering
    BiasType.CASTE: 2,               # Use race steering (similar hierarchy bias)
    BiasType.RELIGION: 3,            # Religion steering
    BiasType.SOCIOECONOMIC: 2,       # Use race steering
    BiasType.NATIONALITY: 2,         # Use race steering
    BiasType.AGE: 4,                 # Prompt prefix
    BiasType.SEXUAL_ORIENTATION: 4,  # Prompt prefix
    BiasType.PHYSICAL_APPEARANCE: 4, # Prompt prefix
    BiasType.DISABILITY: 4,          # Prompt prefix
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class UnifiedBiasEntry:
    """Unified format for bias evaluation entries."""
    id: str
    source_dataset: str
    language: str
    sentence: str
    target_stereotypical: List[str]
    target_anti_stereotypical: List[str]
    bias_type: str
    bias_type_original: str
    stereo_direction: str  # "stereo" or "antistereo"
    mask_count: int
    recommended_arm: int
    metadata: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_target_list(target_str: str) -> List[str]:
    """
    Parse target string from CSV to list.
    Handles formats like: "['Dalits']", "['black', 'African']", etc.
    """
    if pd.isna(target_str) or target_str == "":
        return []

    try:
        # Try literal eval first
        parsed = ast.literal_eval(target_str)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
        return [str(parsed).strip()]
    except (ValueError, SyntaxError):
        # Fallback: clean string manually
        cleaned = target_str.strip("[]'\"")
        if "," in cleaned:
            return [item.strip().strip("'\"") for item in cleaned.split(",")]
        return [cleaned] if cleaned else []

def count_masks(sentence: str) -> int:
    """Count number of MASK tokens in sentence."""
    return sentence.upper().count("MASK")

def generate_entry_id(source: str, language: str, bias_type: str, sentence: str) -> str:
    """Generate unique ID for entry based on content hash."""
    content = f"{source}_{language}_{bias_type}_{sentence}"
    hash_suffix = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{source}_{language}_{bias_type}_{hash_suffix}"

def detect_language_from_path(file_path: str) -> Language:
    """Detect language from file path."""
    path_lower = file_path.lower()
    if "bengali" in path_lower or "/bn/" in path_lower:
        return Language.BENGALI
    elif "hindi" in path_lower or "/hi/" in path_lower:
        return Language.HINDI
    else:
        return Language.ENGLISH

def detect_bias_type_from_filename(filename: str) -> str:
    """Detect bias type from IndiBias filename."""
    filename_lower = filename.lower()
    if "caste" in filename_lower:
        return "caste"
    elif "gender" in filename_lower:
        return "gender"
    elif "religious" in filename_lower or "religion" in filename_lower:
        return "religion"
    elif "race" in filename_lower:
        return "race"
    return "unknown"

# ============================================================================
# INDIBIAS LOADER
# ============================================================================

class IndiBiasLoader:
    """Loader for Indian Multilingual Bias Dataset."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to IndiBias dataset root directory
        """
        self.data_dir = Path(data_dir)

        # File mappings for each language
        self.file_mappings = {
            Language.ENGLISH: {
                "caste": "english/Caste.csv",
                "gender": "english/Gender.csv",
                "religion": "english/India_Religious.csv",
                "race": "english/Race.csv",
            },
            Language.HINDI: {
                "caste": "hindi/Caste_Hindi.csv",
                "gender": "hindi/gender_hindi.csv",
                "religion": "hindi/India_Religious_hindi.csv",
                "race": "hindi/race_hindi.csv",
            },
            Language.BENGALI: {
                "caste": "bengali/Caste_Bengali.csv",
                "gender": "bengali/Gender_Bengali.csv",
                "religion": "bengali/India_Religious_Bengali.csv",
                "race": "bengali/Race_Bengali.csv",
            },
        }

    def load_single_file(
        self,
        language: Language,
        bias_type: str
    ) -> List[UnifiedBiasEntry]:
        """Load single CSV file and convert to unified format."""

        file_path = self.data_dir / self.file_mappings[language][bias_type]

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return []

        df = pd.read_csv(file_path, encoding='utf-8')
        entries = []

        for idx, row in df.iterrows():
            sentence = str(row.get('Sentence', ''))
            target_stereo = parse_target_list(row.get('Target_Stereotypical', ''))
            target_anti = parse_target_list(row.get('Target_Anti-Stereotypical', ''))

            # Skip invalid entries
            if not sentence or "MASK" not in sentence.upper():
                continue
            if not target_stereo and not target_anti:
                continue

            # Normalize bias type
            unified_bias = BIAS_TYPE_NORMALIZATION.get(bias_type, BiasType.RACE)

            entry = UnifiedBiasEntry(
                id=generate_entry_id("indibias", language.value, bias_type, sentence),
                source_dataset=SourceDataset.INDIBIAS.value,
                language=language.value,
                sentence=sentence,
                target_stereotypical=target_stereo,
                target_anti_stereotypical=target_anti,
                bias_type=unified_bias.value,
                bias_type_original=bias_type,
                stereo_direction="stereo",  # IndiBias default
                mask_count=count_masks(sentence),
                recommended_arm=BIAS_TO_RECOMMENDED_ARM.get(unified_bias, 4),
                metadata={
                    "source_file": str(file_path.name),
                    "original_index": int(idx),
                }
            )
            entries.append(entry)

        return entries

    def load_all(self) -> List[UnifiedBiasEntry]:
        """Load all IndiBias data."""
        all_entries = []

        for language in Language:
            for bias_type in ["caste", "gender", "religion", "race"]:
                entries = self.load_single_file(language, bias_type)
                all_entries.extend(entries)
                print(f"Loaded {len(entries)} entries: IndiBias/{language.value}/{bias_type}")

        print(f"Total IndiBias entries: {len(all_entries)}")
        return all_entries

    def load_by_language(self, language: Language) -> List[UnifiedBiasEntry]:
        """Load all entries for a specific language."""
        entries = []
        for bias_type in ["caste", "gender", "religion", "race"]:
            entries.extend(self.load_single_file(language, bias_type))
        return entries

    def load_by_bias_type(self, bias_type: str) -> List[UnifiedBiasEntry]:
        """Load all entries for a specific bias type across all languages."""
        entries = []
        for language in Language:
            entries.extend(self.load_single_file(language, bias_type))
        return entries

# ============================================================================
# CROWS-PAIRS LOADER
# ============================================================================

class CrowsPairsLoader:
    """Loader for Multi-CrowS-Pairs Dataset."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to Multi-CrowS-Pairs dataset root directory
        """
        self.data_dir = Path(data_dir)

        self.file_mappings = {
            Language.ENGLISH: "English/crows_pair_english.csv",
            Language.HINDI: "Hindi/crows_pair_hindi.csv",
            Language.BENGALI: "Bengali/crows_pair_bengali.csv",
        }

    def load_single_file(self, language: Language) -> List[UnifiedBiasEntry]:
        """Load single language file and convert to unified format."""

        file_path = self.data_dir / self.file_mappings[language]

        if not file_path.exists():
            print(f"Warning: File not found: {file_path}")
            return []

        df = pd.read_csv(file_path, encoding='utf-8')
        entries = []

        for idx, row in df.iterrows():
            sentence = str(row.get('Sentence', ''))
            target_stereo = parse_target_list(row.get('Target_Stereotypical', ''))
            target_anti = parse_target_list(row.get('Target_Anti-Stereotypical', ''))
            bias_type_original = str(row.get('bias_type', '')).lower().strip()
            stereo_direction = str(row.get('stereo_antistereo', 'stereo')).lower().strip()
            original_index = row.get('Index', idx)

            # Skip invalid entries
            if not sentence or "MASK" not in sentence.upper():
                continue
            if not target_stereo and not target_anti:
                continue

            # Normalize bias type
            unified_bias = BIAS_TYPE_NORMALIZATION.get(
                bias_type_original,
                BiasType.RACE
            )

            entry = UnifiedBiasEntry(
                id=generate_entry_id("crowspairs", language.value, bias_type_original, sentence),
                source_dataset=SourceDataset.CROWSPAIRS.value,
                language=language.value,
                sentence=sentence,
                target_stereotypical=target_stereo,
                target_anti_stereotypical=target_anti,
                bias_type=unified_bias.value,
                bias_type_original=bias_type_original,
                stereo_direction=stereo_direction,
                mask_count=count_masks(sentence),
                recommended_arm=BIAS_TO_RECOMMENDED_ARM.get(unified_bias, 4),
                metadata={
                    "source_file": str(file_path.name),
                    "original_index": int(original_index),
                    "annotations": str(row.get('annotations', '')),
                }
            )
            entries.append(entry)

        return entries

    def load_all(self) -> List[UnifiedBiasEntry]:
        """Load all CrowS-Pairs data."""
        all_entries = []

        for language in Language:
            entries = self.load_single_file(language)
            all_entries.extend(entries)
            print(f"Loaded {len(entries)} entries: CrowS-Pairs/{language.value}")

        print(f"Total CrowS-Pairs entries: {len(all_entries)}")
        return all_entries

    def load_by_language(self, language: Language) -> List[UnifiedBiasEntry]:
        """Load all entries for a specific language."""
        return self.load_single_file(language)

    def load_by_bias_type(self, bias_type: str) -> List[UnifiedBiasEntry]:
        """Load all entries for a specific bias type across all languages."""
        all_entries = self.load_all()
        return [e for e in all_entries if e.bias_type_original == bias_type]

# ============================================================================
# UNIFIED DATASET MANAGER
# ============================================================================

class UnifiedDatasetManager:
    """
    Manages both datasets and provides unified access.
    Creates train/val/test splits and generates contrastive pairs.
    """

    def __init__(
        self,
        indibias_dir: str,
        crowspairs_dir: str,
        output_dir: str,
        random_seed: int = 42
    ):
        """
        Args:
            indibias_dir: Path to IndiBias dataset
            crowspairs_dir: Path to Multi-CrowS-Pairs dataset
            output_dir: Path to save processed data
            random_seed: Seed for reproducible splits
        """
        self.indibias_loader = IndiBiasLoader(indibias_dir)
        self.crowspairs_loader = CrowsPairsLoader(crowspairs_dir)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.all_entries: List[UnifiedBiasEntry] = []

    def load_all_data(self) -> List[UnifiedBiasEntry]:
        """Load data from both datasets."""
        print("=" * 60)
        print("Loading IndiBias dataset...")
        print("=" * 60)
        indibias_entries = self.indibias_loader.load_all()

        print("\n" + "=" * 60)
        print("Loading Multi-CrowS-Pairs dataset...")
        print("=" * 60)
        crowspairs_entries = self.crowspairs_loader.load_all()

        self.all_entries = indibias_entries + crowspairs_entries

        print("\n" + "=" * 60)
        print(f"TOTAL ENTRIES LOADED: {len(self.all_entries)}")
        print("=" * 60)

        return self.all_entries

    def get_statistics(self) -> Dict:
        """Get statistics about loaded data."""
        if not self.all_entries:
            return {}

        stats = {
            "total_entries": len(self.all_entries),
            "by_source": {},
            "by_language": {},
            "by_bias_type": {},
            "by_language_and_bias": {},
        }

        for entry in self.all_entries:
            # By source
            src = entry.source_dataset
            stats["by_source"][src] = stats["by_source"].get(src, 0) + 1

            # By language
            lang = entry.language
            stats["by_language"][lang] = stats["by_language"].get(lang, 0) + 1

            # By bias type
            bias = entry.bias_type
            stats["by_bias_type"][bias] = stats["by_bias_type"].get(bias, 0) + 1

            # By language and bias
            key = f"{lang}_{bias}"
            stats["by_language_and_bias"][key] = stats["by_language_and_bias"].get(key, 0) + 1

        return stats

    def create_splits(
        self,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        stratify_by: str = "bias_type"  # or "language" or "both"
    ) -> Tuple[List[UnifiedBiasEntry], List[UnifiedBiasEntry], List[UnifiedBiasEntry]]:
        """
        Create train/val/test splits with stratification.

        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            stratify_by: How to stratify the split

        Returns:
            Tuple of (train_entries, val_entries, test_entries)
        """
        import random
        random.seed(self.random_seed)

        if not self.all_entries:
            self.load_all_data()

        # Group entries for stratification
        if stratify_by == "both":
            groups = {}
            for entry in self.all_entries:
                key = f"{entry.language}_{entry.bias_type}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(entry)
        elif stratify_by == "language":
            groups = {}
            for entry in self.all_entries:
                if entry.language not in groups:
                    groups[entry.language] = []
                groups[entry.language].append(entry)
        else:  # bias_type
            groups = {}
            for entry in self.all_entries:
                if entry.bias_type not in groups:
                    groups[entry.bias_type] = []
                groups[entry.bias_type].append(entry)

        train_entries = []
        val_entries = []
        test_entries = []

        for key, entries in groups.items():
            random.shuffle(entries)
            n = len(entries)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_entries.extend(entries[:n_train])
            val_entries.extend(entries[n_train:n_train + n_val])
            test_entries.extend(entries[n_train + n_val:])

        # Shuffle final lists
        random.shuffle(train_entries)
        random.shuffle(val_entries)
        random.shuffle(test_entries)

        print(f"\nSplit Statistics:")
        print(f"  Train: {len(train_entries)} ({len(train_entries)/len(self.all_entries)*100:.1f}%)")
        print(f"  Val:   {len(val_entries)} ({len(val_entries)/len(self.all_entries)*100:.1f}%)")
        print(f"  Test:  {len(test_entries)} ({len(test_entries)/len(self.all_entries)*100:.1f}%)")

        return train_entries, val_entries, test_entries

    def create_contrastive_pairs(self) -> List[Dict]:
        """
        Create contrastive sentence pairs for steering vector creation.

        Each pair has:
        - stereotypical_sentence: MASK filled with stereotypical target
        - anti_stereotypical_sentence: MASK filled with anti-stereotypical target

        Returns:
            List of contrastive pair dictionaries
        """
        if not self.all_entries:
            self.load_all_data()

        contrastive_pairs = []

        for entry in self.all_entries:
            # Skip entries with multiple MASK tokens (complex to pair)
            if entry.mask_count != 1:
                continue

            # Skip entries without both targets
            if not entry.target_stereotypical or not entry.target_anti_stereotypical:
                continue

            # Create sentences by filling MASK
            for stereo_target in entry.target_stereotypical:
                for anti_target in entry.target_anti_stereotypical:
                    # Replace MASK with targets (case-insensitive)
                    stereo_sentence = re.sub(
                        r'MASK',
                        stereo_target,
                        entry.sentence,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    anti_sentence = re.sub(
                        r'MASK',
                        anti_target,
                        entry.sentence,
                        count=1,
                        flags=re.IGNORECASE
                    )

                    pair = {
                        "biased": stereo_sentence,
                        "neutral": anti_sentence,
                        "id": f"pair_{entry.id}_{stereo_target}_{anti_target}",
                        "language": entry.language,
                        "bias_type": entry.bias_type,
                        "stereotypical_target": stereo_target,
                        "anti_stereotypical_target": anti_target,
                        "original_template": entry.sentence,
                        "source_entry_id": entry.id,
                    }
                    contrastive_pairs.append(pair)

        print(f"\nCreated {len(contrastive_pairs)} contrastive pairs for steering vectors")

        # Statistics by bias type
        pairs_by_bias = {}
        for pair in contrastive_pairs:
            bias = pair["bias_type"]
            pairs_by_bias[bias] = pairs_by_bias.get(bias, 0) + 1

        print("Contrastive pairs by bias type:")
        for bias, count in sorted(pairs_by_bias.items()):
            print(f"  {bias}: {count}")

        return contrastive_pairs

    def save_processed_data(
        self,
        train_entries: List[UnifiedBiasEntry],
        val_entries: List[UnifiedBiasEntry],
        test_entries: List[UnifiedBiasEntry],
        contrastive_pairs: List[Dict]
    ):
        """Save all processed data to output directory."""

        # Save splits as JSON
        def entries_to_json(entries: List[UnifiedBiasEntry]) -> List[Dict]:
            return [e.to_dict() for e in entries]

        # Training data
        train_path = self.output_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(entries_to_json(train_entries), f, ensure_ascii=False, indent=2)
        print(f"Saved: {train_path}")

        # Validation data
        val_path = self.output_dir / "val.json"
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(entries_to_json(val_entries), f, ensure_ascii=False, indent=2)
        print(f"Saved: {val_path}")

        # Test data
        test_path = self.output_dir / "test.json"
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(entries_to_json(test_entries), f, ensure_ascii=False, indent=2)
        print(f"Saved: {test_path}")

        # Contrastive pairs
        pairs_path = self.output_dir / "contrastive_pairs.json"
        with open(pairs_path, 'w', encoding='utf-8') as f:
            json.dump(contrastive_pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved: {pairs_path}")

        # Save contrastive pairs by bias type (for steering vector creation)
        pairs_by_bias = {}
        for pair in contrastive_pairs:
            bias = pair["bias_type"]
            if bias not in pairs_by_bias:
                pairs_by_bias[bias] = {"pairs": []}
            pairs_by_bias[bias]["pairs"].append(pair)

        steering_dir = self.output_dir / "steering_pairs"
        steering_dir.mkdir(exist_ok=True)

        for bias, data in pairs_by_bias.items():
            bias_path = steering_dir / f"{bias}_pairs.json"
            with open(bias_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved: {bias_path} ({len(data['pairs'])} pairs)")

        # Save statistics
        stats = self.get_statistics()
        stats["splits"] = {
            "train": len(train_entries),
            "val": len(val_entries),
            "test": len(test_entries),
        }
        stats["contrastive_pairs"] = {
            "total": len(contrastive_pairs),
            "by_bias_type": {k: len(v["pairs"]) for k, v in pairs_by_bias.items()}
        }

        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {stats_path}")

    def save_for_mab_pipeline(self):
        """
        Complete pipeline: load, split, create pairs, and save.
        This is the main entry point.
        """
        print("\n" + "=" * 70)
        print("UNIFIED DATASET PROCESSING FOR MAB DEBIASING PIPELINE")
        print("=" * 70)

        # Step 1: Load all data
        self.load_all_data()

        # Step 2: Print statistics
        stats = self.get_statistics()
        print("\n" + "-" * 40)
        print("DATASET STATISTICS")
        print("-" * 40)
        print(f"Total entries: {stats['total_entries']}")
        print("\nBy source:")
        for src, count in stats['by_source'].items():
            print(f"  {src}: {count}")
        print("\nBy language:")
        for lang, count in stats['by_language'].items():
            print(f"  {lang}: {count}")
        print("\nBy bias type:")
        for bias, count in sorted(stats['by_bias_type'].items()):
            print(f"  {bias}: {count}")

        # Step 3: Create splits
        print("\n" + "-" * 40)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("-" * 40)
        train, val, test = self.create_splits(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            stratify_by="both"  # Stratify by language AND bias type
        )

        # Step 4: Create contrastive pairs
        print("\n" + "-" * 40)
        print("CREATING CONTRASTIVE PAIRS FOR STEERING VECTORS")
        print("-" * 40)
        contrastive_pairs = self.create_contrastive_pairs()

        # Step 5: Save everything
        print("\n" + "-" * 40)
        print("SAVING PROCESSED DATA")
        print("-" * 40)
        self.save_processed_data(train, val, test, contrastive_pairs)

        print("\n" + "=" * 70)
        print("DATA PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nFiles created:")
        print("  - train.json (training data)")
        print("  - val.json (validation data)")
        print("  - test.json (test data)")
        print("  - contrastive_pairs.json (all contrastive pairs)")
        print("  - steering_pairs/*.json (pairs by bias type)")
        print("  - dataset_statistics.json (statistics)")

        return {
            "train": train,
            "val": val,
            "test": test,
            "contrastive_pairs": contrastive_pairs,
            "statistics": stats,
        }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process bias datasets for MAB pipeline")
    parser.add_argument(
        "--indibias_dir",
        type=str,
        required=True,
        help="Path to IndiBias dataset directory"
    )
    parser.add_argument(
        "--crowspairs_dir",
        type=str,
        required=True,
        help="Path to Multi-CrowS-Pairs dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Path to save processed data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    manager = UnifiedDatasetManager(
        indibias_dir=args.indibias_dir,
        crowspairs_dir=args.crowspairs_dir,
        output_dir=args.output_dir,
        random_seed=args.seed
    )

    manager.save_for_mab_pipeline()
