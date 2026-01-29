"""
Download and prepare bias evaluation datasets.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_bbq():
    """Download BBQ (Bias Benchmark for QA) dataset."""
    logger.info("Downloading BBQ dataset...")

    try:
        dataset = load_dataset("heegyu/bbq", "all")

        # Convert to our format
        data = []
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                for item in dataset[split]:
                    data.append({
                        'input': item.get('context', '') + ' ' + item.get('question', ''),
                        'bias_type': item.get('category', 'unknown'),
                        'split': split,
                        'language': 'en',
                        'source': 'bbq'
                    })

        logger.info(f"BBQ: Downloaded {len(data)} samples")
        return data

    except Exception as e:
        logger.error(f"Failed to download BBQ: {e}")
        return []


def download_winobias():
    """Download WinoBias dataset."""
    logger.info("Downloading WinoBias dataset...")

    try:
        dataset = load_dataset("wino_bias", "type1_pro")

        data = []
        for split in ['train', 'validation', 'test']:
            if split in dataset:
                for item in dataset[split]:
                    data.append({
                        'input': item['text'],
                        'bias_type': 'gender',
                        'split': split,
                        'language': 'en',
                        'source': 'winobias'
                    })

        logger.info(f"WinoBias: Downloaded {len(data)} samples")
        return data

    except Exception as e:
        logger.error(f"Failed to download WinoBias: {e}")
        return []


def download_crows_pairs():
    """Download CrowS-Pairs dataset."""
    logger.info("Downloading CrowS-Pairs dataset...")

    try:
        dataset = load_dataset("crows_pairs")

        data = []
        for split in dataset:
            for item in dataset[split]:
                data.append({
                    'input': item['sent_more'],
                    'bias_type': item.get('bias_type', 'unknown'),
                    'split': 'train',  # CrowS-Pairs has no official splits
                    'language': 'en',
                    'source': 'crows_pairs'
                })

        logger.info(f"CrowS-Pairs: Downloaded {len(data)} samples")
        return data

    except Exception as e:
        logger.error(f"Failed to download CrowS-Pairs: {e}")
        return []


def create_multilingual_samples(data: List[Dict], target_lang: str) -> List[Dict]:
    """
    Create samples for Hindi/Bengali.

    Note: For actual translation, use a translation API or pre-translated datasets.
    This is a placeholder that marks samples for manual translation.
    """
    logger.info(f"Creating {target_lang} samples (marked for translation)...")

    # Take subset for translation
    subset_size = min(100, len(data))
    subset = data[:subset_size]

    multilingual_data = []
    for item in subset:
        new_item = item.copy()
        new_item['language'] = target_lang
        new_item['input'] = f"[TO_TRANSLATE_{target_lang.upper()}] {item['input']}"
        multilingual_data.append(new_item)

    logger.info(f"{target_lang}: Created {len(multilingual_data)} samples for translation")
    return multilingual_data


def split_data(data: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split data into train/val/test."""
    import random
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    for item in train_data:
        item['split'] = 'train'
    for item in val_data:
        item['split'] = 'validation'
    for item in test_data:
        item['split'] = 'test'

    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }


def save_data(data: Dict[str, List[Dict]], output_dir: Path, language: str):
    """Save data to JSON files."""
    lang_dir = output_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    for split, items in data.items():
        output_path = lang_dir / f"{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(items)} samples to {output_path}")


def main():
    """Main function to prepare all datasets."""
    logger.info("Starting dataset preparation...")

    # Output directory
    output_dir = Path('./data/bias_evaluation_sets')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download English datasets
    all_data = []

    bbq_data = download_bbq()
    all_data.extend(bbq_data)

    winobias_data = download_winobias()
    all_data.extend(winobias_data)

    crows_data = download_crows_pairs()
    all_data.extend(crows_data)

    logger.info(f"Total English samples: {len(all_data)}")

    # Split English data
    en_splits = split_data(all_data)
    save_data(en_splits, output_dir, 'en')

    # Create Hindi samples (marked for translation)
    hi_data = create_multilingual_samples(all_data, 'hi')
    hi_splits = split_data(hi_data)
    save_data(hi_splits, output_dir, 'hi')

    # Create Bengali samples (marked for translation)
    bn_data = create_multilingual_samples(all_data, 'bn')
    bn_splits = split_data(bn_data)
    save_data(bn_splits, output_dir, 'bn')

    # Save statistics
    stats = {
        'total_samples': len(all_data),
        'languages': {
            'en': {split: len(items) for split, items in en_splits.items()},
            'hi': {split: len(items) for split, items in hi_splits.items()},
            'bn': {split: len(items) for split, items in bn_splits.items()}
        },
        'sources': {
            'bbq': len(bbq_data),
            'winobias': len(winobias_data),
            'crows_pairs': len(crows_data)
        }
    }

    stats_path = output_dir / 'dataset_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset statistics saved to {stats_path}")
    logger.info("Dataset preparation complete!")

    # Print summary
    print("\n" + "="*60)
    print("DATASET PREPARATION SUMMARY")
    print("="*60)
    print(f"Total samples: {stats['total_samples']}")
    print(f"\nEnglish: {sum(en_splits[s] for s in ['train', 'validation', 'test'])}")
    print(f"  Train: {len(en_splits['train'])}")
    print(f"  Validation: {len(en_splits['validation'])}")
    print(f"  Test: {len(en_splits['test'])}")
    print(f"\nHindi: {len(hi_data)} (marked for translation)")
    print(f"Bengali: {len(bn_data)} (marked for translation)")
    print("="*60)


if __name__ == "__main__":
    main()
