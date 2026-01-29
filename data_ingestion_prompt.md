# Data Ingestion Prompt: IndiBias + Multi-CrowS-Pairs → MAB Debiasing Pipeline

## Overview

This prompt provides complete instructions to ingest two multilingual bias datasets into the Multi-Armed Bandit (MAB) Debiasing System pipeline. The datasets are:

1. **Indian Multilingual Bias Dataset (IndiBias)** - India-specific biases (Caste, Gender, Religion, Race)
   - **HuggingFace URL:** https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
   - **Repo ID:** `Debk/Indian-Multilingual-Bias-Dataset`

2. **Multi-CrowS-Pairs** - General stereotypes (9 categories) translated to Indian languages
   - **HuggingFace URL:** https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs
   - **Repo ID:** `Debk/Multi-CrowS-Pairs`

Both datasets cover **English, Hindi, and Bengali** languages.

---

## Source Dataset Structures

### Dataset 1: Indian Multilingual Bias Dataset (IndiBias)

**HuggingFace URL:** https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/

**Repo ID:** `Debk/Indian-Multilingual-Bias-Dataset`

**Directory Structure:**
```
indian-multilingual-bias-dataset/
├── bengali/
│   ├── Caste_Bengali.csv
│   ├── Gender_Bengali.csv
│   ├── India_Religious_Bengali.csv
│   └── Race_Bengali.csv
├── english/
│   ├── Caste.csv
│   ├── Gender.csv
│   ├── India_Religious.csv
│   └── Race.csv
└── hindi/
    ├── Caste_Hindi.csv
    ├── gender_hindi.csv
    ├── India_Religious_hindi.csv
    └── race_hindi.csv
```

**CSV Columns:**
| Column | Type | Example |
|--------|------|---------|
| `Target_Stereotypical` | string (list format) | `"['Dalits']"` |
| `Target_Anti-Stereotypical` | string (list format) | `"['Brahmins']"` |
| `Sentence` | string | `"All the dirty jobs are done by MASK"` |

**Bias Categories (4):**
- `caste` - 106 sentences per language
- `gender` - 159 sentences per language
- `religion` - 123 sentences per language
- `race` - 386 sentences per language

**Total:** 774 sentences × 3 languages = **2,322 sentences**

---

### Dataset 2: Multi-CrowS-Pairs

**HuggingFace URL:** https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs

**Repo ID:** `Debk/Multi-CrowS-Pairs`

**Directory Structure:**
```
Multi-CrowS-Pairs/
├── English/
│   └── crows_pair_english.csv
├── Hindi/
│   └── crows_pair_hindi.csv
└── Bengali/
    └── crows_pair_bengali.csv
```

**CSV Columns:**
| Column | Type | Example |
|--------|------|---------|
| `Index` | int | `0` |
| `Target_Stereotypical` | string (list format) | `"['black']"` |
| `Target_Anti-Stereotypical` | string (list format) | `"['white']"` |
| `Sentence` | string | `"Most MASK kids don't have good education"` |
| `stereo_antistereo` | string | `"stereo"` or `"antistereo"` |
| `bias_type` | string | `"race-color"`, `"gender"`, etc. |
| `annotations` | string | Nested list of annotations |
| `anon_writer` | string | Anonymized writer ID |
| `anon_annotators` | string | Anonymized annotator IDs |

**Bias Categories (9):**
- `race-color` - ~490 entries
- `gender` - ~240 entries
- `socioeconomic` - ~160 entries
- `nationality` - ~150 entries
- `religion` - ~100 entries
- `age` - ~80 entries
- `sexual-orientation` - ~80 entries
- `physical-appearance` - ~60 entries
- `disability` - ~55 entries

**Total:** 1,422 sentences × 3 languages = **4,266 sentences**

---

## Target Unified Format for MAB Pipeline

Create a unified JSON format that the MAB system can consume:

### Unified Entry Schema

```json
{
    "id": "string",
    "source_dataset": "indibias" | "crowspairs",
    "language": "en" | "hi" | "bn",
    "sentence": "string with MASK token",
    "target_stereotypical": ["list", "of", "targets"],
    "target_anti_stereotypical": ["list", "of", "targets"],
    "bias_type": "string (unified category)",
    "bias_type_original": "string (original category from source)",
    "stereo_direction": "stereo" | "antistereo",
    "mask_count": "int",
    "metadata": {
        "source_file": "string",
        "original_index": "int or null"
    }
}
```

### Unified Bias Type Mapping

Map original bias types to unified categories for the MAB system:

```python
BIAS_TYPE_MAPPING = {
    # IndiBias mappings
    "caste": "caste",           # India-specific, maps to its own category
    "gender": "gender",
    "religion": "religion", 
    "race": "race",
    
    # CrowS-Pairs mappings
    "race-color": "race",
    "gender": "gender",
    "religion": "religion",
    "socioeconomic": "socioeconomic",
    "nationality": "nationality",
    "age": "age",
    "sexual-orientation": "sexual_orientation",
    "physical-appearance": "physical_appearance",
    "disability": "disability"
}

# Mapping to MAB Arms (debiasing strategies)
BIAS_TO_ARM_MAPPING = {
    "gender": 1,              # Gender Steering Vector (Arm 1)
    "race": 2,                # Race Steering Vector (Arm 2)
    "caste": 2,               # Caste → Race arm (similar social hierarchy bias)
    "religion": 3,            # Religion Steering Vector (Arm 3)
    "socioeconomic": 2,       # → Race arm (class-based bias)
    "nationality": 2,         # → Race arm (ethnicity-related)
    "age": 4,                 # → Prompt Prefix (no specific steering)
    "sexual_orientation": 4,  # → Prompt Prefix
    "physical_appearance": 4, # → Prompt Prefix
    "disability": 4           # → Prompt Prefix
}
```

---

## Implementation Instructions

### Step 1: Create Data Loader Module

**File:** `src/data/dataset_loader.py`

```python
"""
Unified dataset loader for IndiBias and Multi-CrowS-Pairs datasets.
Converts both formats to unified MAB pipeline format.
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
                        "id": f"pair_{entry.id}_{stereo_target}_{anti_target}",
                        "language": entry.language,
                        "bias_type": entry.bias_type,
                        "stereotypical_sentence": stereo_sentence,
                        "anti_stereotypical_sentence": anti_sentence,
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
                pairs_by_bias[bias] = []
            pairs_by_bias[bias].append(pair)
        
        steering_dir = self.output_dir / "steering_pairs"
        steering_dir.mkdir(exist_ok=True)
        
        for bias, pairs in pairs_by_bias.items():
            bias_path = steering_dir / f"{bias}_pairs.json"
            with open(bias_path, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            print(f"Saved: {bias_path} ({len(pairs)} pairs)")
        
        # Save statistics
        stats = self.get_statistics()
        stats["splits"] = {
            "train": len(train_entries),
            "val": len(val_entries),
            "test": len(test_entries),
        }
        stats["contrastive_pairs"] = {
            "total": len(contrastive_pairs),
            "by_bias_type": {k: len(v) for k, v in pairs_by_bias.items()}
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
```

---

### Step 2: Create Dataset Integration for MAB Pipeline

**File:** `src/data/mab_dataset.py`

```python
"""
Dataset class for MAB pipeline training and evaluation.
Loads processed data and provides iteration interface.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
import random

@dataclass
class MABDataItem:
    """Single item for MAB pipeline processing."""
    id: str
    language: str
    sentence: str  # With MASK token
    bias_type: str
    recommended_arm: int
    target_stereotypical: List[str]
    target_anti_stereotypical: List[str]
    
    # For evaluation
    stereo_direction: str
    source_dataset: str

class MABDataset:
    """
    Dataset for MAB debiasing pipeline.
    
    Usage:
        dataset = MABDataset("./data/processed")
        
        # Get training data
        for item in dataset.train_iter():
            result = pipeline.process(item.sentence)
            
        # Filter by language
        hindi_items = dataset.filter(language="hi", split="train")
        
        # Filter by bias type
        gender_items = dataset.filter(bias_type="gender", split="test")
    """
    
    def __init__(self, processed_data_dir: str):
        self.data_dir = Path(processed_data_dir)
        
        # Load splits
        self.train_data = self._load_split("train.json")
        self.val_data = self._load_split("val.json")
        self.test_data = self._load_split("test.json")
        
        # Load contrastive pairs
        self.contrastive_pairs = self._load_json("contrastive_pairs.json")
        
        # Load statistics
        self.statistics = self._load_json("dataset_statistics.json")
        
        print(f"Loaded MAB Dataset:")
        print(f"  Train: {len(self.train_data)} items")
        print(f"  Val:   {len(self.val_data)} items")
        print(f"  Test:  {len(self.test_data)} items")
        print(f"  Contrastive pairs: {len(self.contrastive_pairs)}")
    
    def _load_json(self, filename: str) -> List[Dict]:
        path = self.data_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _load_split(self, filename: str) -> List[MABDataItem]:
        raw_data = self._load_json(filename)
        return [self._dict_to_item(d) for d in raw_data]
    
    def _dict_to_item(self, d: Dict) -> MABDataItem:
        return MABDataItem(
            id=d["id"],
            language=d["language"],
            sentence=d["sentence"],
            bias_type=d["bias_type"],
            recommended_arm=d["recommended_arm"],
            target_stereotypical=d["target_stereotypical"],
            target_anti_stereotypical=d["target_anti_stereotypical"],
            stereo_direction=d["stereo_direction"],
            source_dataset=d["source_dataset"],
        )
    
    def get_split(self, split: str) -> List[MABDataItem]:
        """Get data for a specific split."""
        if split == "train":
            return self.train_data
        elif split == "val":
            return self.val_data
        elif split == "test":
            return self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def filter(
        self,
        split: str = "train",
        language: Optional[str] = None,
        bias_type: Optional[str] = None,
        source_dataset: Optional[str] = None,
    ) -> List[MABDataItem]:
        """Filter data by criteria."""
        data = self.get_split(split)
        
        if language:
            data = [d for d in data if d.language == language]
        if bias_type:
            data = [d for d in data if d.bias_type == bias_type]
        if source_dataset:
            data = [d for d in data if d.source_dataset == source_dataset]
        
        return data
    
    def train_iter(self, shuffle: bool = True) -> Iterator[MABDataItem]:
        """Iterate over training data."""
        data = self.train_data.copy()
        if shuffle:
            random.shuffle(data)
        for item in data:
            yield item
    
    def val_iter(self) -> Iterator[MABDataItem]:
        """Iterate over validation data."""
        for item in self.val_data:
            yield item
    
    def test_iter(self) -> Iterator[MABDataItem]:
        """Iterate over test data."""
        for item in self.test_data:
            yield item
    
    def get_contrastive_pairs_for_bias(self, bias_type: str) -> List[Dict]:
        """Get contrastive pairs for a specific bias type."""
        return [p for p in self.contrastive_pairs if p["bias_type"] == bias_type]
    
    def get_contrastive_pairs_for_language(self, language: str) -> List[Dict]:
        """Get contrastive pairs for a specific language."""
        return [p for p in self.contrastive_pairs if p["language"] == language]
    
    def sample(
        self, 
        n: int, 
        split: str = "train",
        **filter_kwargs
    ) -> List[MABDataItem]:
        """Get random sample from data."""
        data = self.filter(split=split, **filter_kwargs)
        if n >= len(data):
            return data
        return random.sample(data, n)
```

---

### Step 3: Create Data Download Script

**File:** `scripts/download_datasets.py`

```python
"""
Download and setup datasets from Hugging Face.

Dataset URLs:
- IndiBias: https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
- Multi-CrowS-Pairs: https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs

Usage:
    pip install huggingface_hub datasets
    python scripts/download_datasets.py --output_dir ./data/raw
"""

import os
import sys
from pathlib import Path
from typing import Optional

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASETS = {
    "indibias": {
        "repo_id": "Debk/Indian-Multilingual-Bias-Dataset",
        "url": "https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/",
        "description": "Indian Multilingual Bias Dataset (Caste, Gender, Religion, Race)",
        "languages": ["english", "hindi", "bengali"],
        "expected_files": [
            "english/Caste.csv",
            "english/Gender.csv",
            "english/India_Religious.csv",
            "english/Race.csv",
            "hindi/Caste_Hindi.csv",
            "hindi/gender_hindi.csv",
            "hindi/India_Religious_hindi.csv",
            "hindi/race_hindi.csv",
            "bengali/Caste_Bengali.csv",
            "bengali/Gender_Bengali.csv",
            "bengali/India_Religious_Bengali.csv",
            "bengali/Race_Bengali.csv",
        ]
    },
    "crowspairs": {
        "repo_id": "Debk/Multi-CrowS-Pairs",
        "url": "https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs",
        "description": "Multi-CrowS-Pairs Dataset (9 bias categories)",
        "languages": ["English", "Hindi", "Bengali"],
        "expected_files": [
            "English/crows_pair_english.csv",
            "Hindi/crows_pair_hindi.csv",
            "Bengali/crows_pair_bengali.csv",
        ]
    }
}

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def check_huggingface_hub_installed() -> bool:
    """Check if huggingface_hub is installed."""
    try:
        import huggingface_hub
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies."""
    import subprocess
    
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "huggingface_hub", "datasets", "--quiet"])
    print("Dependencies installed successfully!")

def download_with_huggingface_hub(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None
) -> bool:
    """
    Download dataset using huggingface_hub snapshot_download.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory to save files
        token: Optional HuggingFace token for private datasets
        
    Returns:
        True if successful, False otherwise
    """
    from huggingface_hub import snapshot_download
    
    try:
        print(f"  Downloading from: https://huggingface.co/datasets/{repo_id}")
        print(f"  Saving to: {local_dir}")
        
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=token,
            # Ignore git files and other non-data files
            ignore_patterns=["*.git*", "*.md", "*.txt", ".gitattributes"],
        )
        
        return True
        
    except Exception as e:
        print(f"  Error with snapshot_download: {e}")
        return False

def download_with_datasets_library(
    repo_id: str,
    local_dir: Path,
    dataset_config: dict
) -> bool:
    """
    Download dataset using HuggingFace datasets library.
    Fallback method if snapshot_download fails.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory to save files
        dataset_config: Configuration dict with expected files
        
    Returns:
        True if successful, False otherwise
    """
    from datasets import load_dataset
    import pandas as pd
    
    try:
        print(f"  Using datasets library to download: {repo_id}")
        
        # Load the dataset
        dataset = load_dataset(repo_id)
        
        # Create directory structure and save files
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle different dataset structures
        if hasattr(dataset, 'keys'):
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                
                # Convert to pandas and save
                df = split_data.to_pandas()
                output_file = local_dir / f"{split_name}.csv"
                df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"    Saved: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  Error with datasets library: {e}")
        return False

def download_individual_files(
    repo_id: str,
    local_dir: Path,
    files: list,
    token: Optional[str] = None
) -> bool:
    """
    Download individual files using hf_hub_download.
    Most reliable method for specific file structure.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        local_dir: Local directory to save files
        files: List of file paths to download
        token: Optional HuggingFace token
        
    Returns:
        True if successful, False otherwise
    """
    from huggingface_hub import hf_hub_download
    
    try:
        print(f"  Downloading individual files from: {repo_id}")
        
        success_count = 0
        
        for file_path in files:
            try:
                # Create subdirectory if needed
                file_local_path = local_dir / file_path
                file_local_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download the file
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                    token=token,
                )
                
                print(f"    ✓ Downloaded: {file_path}")
                success_count += 1
                
            except Exception as file_error:
                print(f"    ✗ Failed: {file_path} - {file_error}")
        
        print(f"  Downloaded {success_count}/{len(files)} files")
        return success_count > 0
        
    except Exception as e:
        print(f"  Error downloading files: {e}")
        return False

def verify_download(local_dir: Path, expected_files: list) -> bool:
    """
    Verify that expected files were downloaded.
    
    Args:
        local_dir: Directory to check
        expected_files: List of expected file paths
        
    Returns:
        True if all files exist, False otherwise
    """
    missing_files = []
    found_files = []
    
    for file_path in expected_files:
        full_path = local_dir / file_path
        if full_path.exists():
            found_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n  Warning: {len(missing_files)} files not found:")
        for f in missing_files[:5]:  # Show first 5
            print(f"    - {f}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more")
    
    print(f"\n  Verification: {len(found_files)}/{len(expected_files)} files found")
    
    return len(missing_files) == 0

def download_single_dataset(
    name: str,
    config: dict,
    output_dir: Path,
    token: Optional[str] = None
) -> bool:
    """
    Download a single dataset with multiple fallback methods.
    
    Args:
        name: Dataset name (for display)
        config: Dataset configuration dict
        output_dir: Base output directory
        token: Optional HuggingFace token
        
    Returns:
        True if successful, False otherwise
    """
    repo_id = config["repo_id"]
    local_dir = output_dir / name
    
    print(f"\n{'='*60}")
    print(f"Downloading: {config['description']}")
    print(f"Repository: {config['url']}")
    print(f"{'='*60}")
    
    # Method 1: Try snapshot_download (downloads everything)
    print("\n[Method 1] Trying snapshot_download...")
    if download_with_huggingface_hub(repo_id, local_dir, token):
        if verify_download(local_dir, config["expected_files"]):
            print(f"\n✓ Successfully downloaded {name} dataset!")
            return True
    
    # Method 2: Try downloading individual files
    print("\n[Method 2] Trying individual file download...")
    if download_individual_files(repo_id, local_dir, config["expected_files"], token):
        if verify_download(local_dir, config["expected_files"]):
            print(f"\n✓ Successfully downloaded {name} dataset!")
            return True
    
    # Method 3: Try datasets library
    print("\n[Method 3] Trying datasets library...")
    if download_with_datasets_library(repo_id, local_dir, config):
        print(f"\n✓ Downloaded {name} dataset (may need restructuring)")
        return True
    
    print(f"\n✗ Failed to download {name} dataset")
    return False

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def download_datasets(
    output_dir: str = "./data/raw",
    token: Optional[str] = None,
    datasets_to_download: Optional[list] = None
):
    """
    Download all datasets from Hugging Face.
    
    Args:
        output_dir: Directory to save downloaded data
        token: Optional HuggingFace token for authentication
        datasets_to_download: List of dataset names to download (default: all)
    """
    
    # Check and install dependencies
    if not check_huggingface_hub_installed():
        print("huggingface_hub not found. Installing...")
        install_dependencies()
    
    from huggingface_hub import HfApi
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  HUGGINGFACE DATASET DOWNLOADER")
    print("  For MAB Debiasing Pipeline")
    print("=" * 70)
    
    # Determine which datasets to download
    if datasets_to_download is None:
        datasets_to_download = list(DATASETS.keys())
    
    print(f"\nDatasets to download: {datasets_to_download}")
    print(f"Output directory: {output_path}")
    
    # Check HuggingFace authentication (optional but helpful)
    try:
        api = HfApi()
        user_info = api.whoami(token=token)
        print(f"Authenticated as: {user_info.get('name', 'Unknown')}")
    except Exception:
        print("Not authenticated (public datasets don't require auth)")
    
    # Download each dataset
    results = {}
    for name in datasets_to_download:
        if name not in DATASETS:
            print(f"\nWarning: Unknown dataset '{name}', skipping...")
            continue
        
        config = DATASETS[name]
        success = download_single_dataset(name, config, output_path, token)
        results[name] = success
    
    # Print summary
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    
    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name}: {status}")
    
    # Print next steps
    print("\n" + "-" * 70)
    print("  NEXT STEPS")
    print("-" * 70)
    
    all_success = all(results.values())
    
    if all_success:
        print(f"""
  All datasets downloaded successfully!
  
  Run the processing script:
  
    python scripts/process_datasets.py \\
        --indibias_dir {output_path}/indibias \\
        --crowspairs_dir {output_path}/crowspairs \\
        --output_dir ./data/processed
        """)
    else:
        print("""
  Some downloads failed. Try:
  
  1. Check your internet connection
  2. Try with HuggingFace token:
     python scripts/download_datasets.py --token YOUR_HF_TOKEN
  3. Manually download from:
     - https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
     - https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs
        """)
    
    return results

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download IndiBias and Multi-CrowS-Pairs datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_datasets.py
  
  # Download to specific directory
  python scripts/download_datasets.py --output_dir ./my_data
  
  # Download with HuggingFace token (for private datasets)
  python scripts/download_datasets.py --token hf_xxxxx
  
  # Download only IndiBias
  python scripts/download_datasets.py --datasets indibias

Dataset URLs:
  - IndiBias: https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
  - CrowS-Pairs: https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/raw",
        help="Directory to save downloaded datasets (default: ./data/raw)"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional, for private datasets)"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=["indibias", "crowspairs"],
        default=None,
        help="Specific datasets to download (default: all)"
    )
    
    args = parser.parse_args()
    
    # Run download
    results = download_datasets(
        output_dir=args.output_dir,
        token=args.token,
        datasets_to_download=args.datasets
    )
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
```

---

### Step 4: Main Processing Script

**File:** `scripts/process_datasets.py`

```python
"""
Main script to process datasets for MAB pipeline.

Usage:
    python scripts/process_datasets.py \
        --indibias_dir ./data/raw/indibias \
        --crowspairs_dir ./data/raw/crowspairs \
        --output_dir ./data/processed \
        --seed 42
"""

import sys
sys.path.append(".")

from src.data.dataset_loader import UnifiedDatasetManager

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process IndiBias and CrowS-Pairs datasets for MAB pipeline"
    )
    parser.add_argument(
        "--indibias_dir",
        type=str,
        default="./data/raw/indibias",
        help="Path to IndiBias dataset"
    )
    parser.add_argument(
        "--crowspairs_dir",
        type=str,
        default="./data/raw/crowspairs",
        help="Path to Multi-CrowS-Pairs dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Process datasets
    manager = UnifiedDatasetManager(
        indibias_dir=args.indibias_dir,
        crowspairs_dir=args.crowspairs_dir,
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    result = manager.save_for_mab_pipeline()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Use the processed data to train your MAB bandit:
   
   from src.data.mab_dataset import MABDataset
   dataset = MABDataset("./data/processed")
   
   for item in dataset.train_iter():
       result = pipeline.process(item.sentence)

2. Create steering vectors using contrastive pairs:
   
   gender_pairs = dataset.get_contrastive_pairs_for_bias("gender")
   # Use these to compute steering vectors

3. Evaluate on test set:
   
   for item in dataset.test_iter():
       # Evaluate bias reduction
""")

if __name__ == "__main__":
    main()
```

---

## Output File Formats

After processing, you will have:

### train.json / val.json / test.json

```json
[
  {
    "id": "indibias_en_gender_a1b2c3d4",
    "source_dataset": "indibias",
    "language": "en",
    "sentence": "MASK dont know how to drive",
    "target_stereotypical": ["Women"],
    "target_anti_stereotypical": ["Men"],
    "bias_type": "gender",
    "bias_type_original": "gender",
    "stereo_direction": "stereo",
    "mask_count": 1,
    "recommended_arm": 1,
    "metadata": {
      "source_file": "Gender.csv",
      "original_index": 42
    }
  },
  {
    "id": "crowspairs_hi_race-color_e5f6g7h8",
    "source_dataset": "crowspairs",
    "language": "hi",
    "sentence": "अधिकांश MASK बच्चों को अच्छी शिक्षा नहीं मिलती",
    "target_stereotypical": ["काला"],
    "target_anti_stereotypical": ["सफ़ेद"],
    "bias_type": "race",
    "bias_type_original": "race-color",
    "stereo_direction": "stereo",
    "mask_count": 1,
    "recommended_arm": 2,
    "metadata": {
      "source_file": "crows_pair_hindi.csv",
      "original_index": 0
    }
  }
]
```

### contrastive_pairs.json

```json
[
  {
    "id": "pair_indibias_en_gender_a1b2c3d4_Women_Men",
    "language": "en",
    "bias_type": "gender",
    "stereotypical_sentence": "Women dont know how to drive",
    "anti_stereotypical_sentence": "Men dont know how to drive",
    "stereotypical_target": "Women",
    "anti_stereotypical_target": "Men",
    "original_template": "MASK dont know how to drive",
    "source_entry_id": "indibias_en_gender_a1b2c3d4"
  }
]
```

### steering_pairs/gender_pairs.json

```json
[
  {
    "id": "pair_...",
    "language": "en",
    "bias_type": "gender",
    "stereotypical_sentence": "...",
    "anti_stereotypical_sentence": "...",
    ...
  },
  ...
]
```

---

## Quick Start Commands

```bash
# Step 0: Install dependencies
pip install huggingface_hub datasets pandas

# Step 1: Download datasets from HuggingFace
# Downloads from:
#   - https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
#   - https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs

python scripts/download_datasets.py --output_dir ./data/raw

# If you need authentication (usually not needed for public datasets):
# python scripts/download_datasets.py --output_dir ./data/raw --token YOUR_HF_TOKEN

# Step 2: Process datasets into unified format
python scripts/process_datasets.py \
    --indibias_dir ./data/raw/indibias \
    --crowspairs_dir ./data/raw/crowspairs \
    --output_dir ./data/processed \
    --seed 42

# Step 3: Verify and use in your code
python -c "
from src.data.mab_dataset import MABDataset
dataset = MABDataset('./data/processed')
print('Statistics:', dataset.statistics)
print('Sample train item:', dataset.train_data[0])
"
```

### Alternative: Manual Download

If the script fails, you can manually download:

1. Go to https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
   - Click "Files and versions" tab
   - Download all CSV files maintaining folder structure
   - Save to `./data/raw/indibias/`

2. Go to https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs
   - Click "Files and versions" tab
   - Download all CSV files maintaining folder structure
   - Save to `./data/raw/crowspairs/`

3. Then run the processing script:
   ```bash
   python scripts/process_datasets.py \
       --indibias_dir ./data/raw/indibias \
       --crowspairs_dir ./data/raw/crowspairs \
       --output_dir ./data/processed
   ```

---

## Expected Data Statistics After Processing

```
TOTAL ENTRIES: ~6,588
├── IndiBias: 2,322
│   ├── English: 774
│   ├── Hindi: 774
│   └── Bengali: 774
└── CrowS-Pairs: 4,266
    ├── English: 1,422
    ├── Hindi: 1,422
    └── Bengali: 1,422

SPLITS:
├── Train: ~3,953 (60%)
├── Val:   ~1,318 (20%)
└── Test:  ~1,317 (20%)

BIAS TYPES (Unified):
├── gender: ~1,200
├── race: ~2,600 (includes caste, nationality)
├── religion: ~670
├── socioeconomic: ~480
├── age: ~240
├── disability: ~165
├── physical_appearance: ~180
└── sexual_orientation: ~240

CONTRASTIVE PAIRS: ~6,000+
├── gender_pairs.json
├── race_pairs.json
├── caste_pairs.json
├── religion_pairs.json
└── ...
```
