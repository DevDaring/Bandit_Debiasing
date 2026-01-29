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
