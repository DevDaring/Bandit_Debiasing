"""
Download IndiBias and Multi-CrowS-Pairs datasets from Hugging Face.

Dataset URLs:
- IndiBias: https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
- Multi-CrowS-Pairs: https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs

Usage:
    pip install huggingface_hub
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
                          "huggingface_hub", "--quiet"])
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
