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
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
