#!/usr/bin/env python3
"""
Model Downloader Script
Downloads all required models from Hugging Face Hub.

Usage:
    python download_models.py              # Download all models
    python download_models.py --model Qwen2.5-1.5B-Instruct  # Download specific model
    python download_models.py --list       # List available models
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download, HfApi

# Load environment variables
load_dotenv()

# Model configurations
MODELS: Dict[str, Dict] = {
    "Qwen2.5-1.5B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "family": "Qwen",
        "params": "1.5B",
        "size_gb": 3.5,
        "gated": False,
        "trust_remote_code": True
    },
    "Llama-3.2-1B-Instruct": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "Llama",
        "params": "1B",
        "size_gb": 2.5,
        "gated": True,  # Requires license acceptance
        "trust_remote_code": False
    },
    "Gemma-2-2B-IT": {
        "hf_id": "google/gemma-2-2b-it",
        "family": "Gemma",
        "params": "2B",
        "size_gb": 5.0,
        "gated": True,  # Requires license acceptance
        "trust_remote_code": False
    },
    "mGPT-1.3B": {
        "hf_id": "ai-forever/mGPT",
        "family": "mGPT",
        "params": "1.3B",
        "size_gb": 3.0,
        "gated": False,
        "trust_remote_code": False
    },
    "BLOOMZ-7B1": {
        "hf_id": "bigscience/bloomz-7b1",
        "family": "BLOOM",
        "params": "7B",
        "size_gb": 15.0,
        "gated": False,
        "trust_remote_code": False
    },
}


def print_banner():
    """Print script banner."""
    print("=" * 60)
    print("  Multilingual JSON Evaluation - Model Downloader")
    print("=" * 60)
    print()


def list_models():
    """List all available models."""
    print("\nAvailable Models:")
    print("-" * 80)
    print(f"{'Model Name':<25} {'HF ID':<35} {'Size':<8} {'Gated'}")
    print("-" * 80)
    
    total_size = 0
    for name, config in MODELS.items():
        gated_str = "Yes" if config["gated"] else "No"
        print(f"{name:<25} {config['hf_id']:<35} {config['size_gb']:.1f} GB  {gated_str}")
        total_size += config["size_gb"]
    
    print("-" * 80)
    print(f"Total download size (approximate): {total_size:.1f} GB")
    print()
    print("Note: Gated models require accepting license on Hugging Face website:")
    print("  - Llama: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
    print("  - Gemma: https://huggingface.co/google/gemma-2-2b-it")


def check_token():
    """Check and authenticate HF token."""
    token = os.getenv("HF_TOKEN")
    
    if not token:
        print("❌ HF_TOKEN not found in environment!")
        print("\nTo fix this:")
        print("  1. Get your token from: https://huggingface.co/settings/tokens")
        print("  2. Create a .env file with: HF_TOKEN=your_token_here")
        print("  3. Or set environment variable: export HF_TOKEN=your_token_here")
        return None
    
    try:
        login(token=token)
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return token
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None


def check_model_access(hf_id: str, api: HfApi) -> bool:
    """Check if user has access to a model."""
    try:
        api.model_info(hf_id)
        return True
    except Exception as e:
        if "403" in str(e) or "401" in str(e):
            return False
        return True  # Other errors might be network issues


def download_model(model_name: str, config: Dict, cache_dir: str = None) -> bool:
    """Download a single model."""
    hf_id = config["hf_id"]
    
    print(f"\n{'='*50}")
    print(f"Downloading: {model_name}")
    print(f"HF ID: {hf_id}")
    print(f"Estimated size: {config['size_gb']:.1f} GB")
    print(f"{'='*50}")
    
    # Check access for gated models
    if config["gated"]:
        api = HfApi()
        if not check_model_access(hf_id, api):
            print(f"❌ Access denied to {hf_id}")
            print(f"   Please accept the license at: https://huggingface.co/{hf_id}")
            return False
    
    try:
        # Download model
        local_path = snapshot_download(
            repo_id=hf_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
        )
        
        print(f"✓ Downloaded to: {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def download_all(models: List[str] = None, cache_dir: str = None):
    """Download all or specified models."""
    
    # Check authentication
    token = check_token()
    if not token:
        return
    
    # Get models to download
    if models:
        models_to_download = {k: v for k, v in MODELS.items() if k in models}
    else:
        models_to_download = MODELS
    
    if not models_to_download:
        print("No valid models specified.")
        list_models()
        return
    
    # Calculate total size
    total_size = sum(m["size_gb"] for m in models_to_download.values())
    print(f"\nTotal download size: ~{total_size:.1f} GB")
    print(f"Models to download: {len(models_to_download)}")
    
    # Confirm download
    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download each model
    results = {}
    for name, config in models_to_download.items():
        success = download_model(name, config, cache_dir)
        results[name] = success
    
    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    
    for name, success in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {name}")
    
    successful = sum(results.values())
    print(f"\nCompleted: {successful}/{len(results)} models")


def verify_downloads():
    """Verify that models are downloaded and accessible."""
    from transformers import AutoTokenizer
    
    print("\nVerifying downloaded models...")
    print("-" * 50)
    
    for name, config in MODELS.items():
        try:
            # Try to load tokenizer (fast check)
            tokenizer = AutoTokenizer.from_pretrained(
                config["hf_id"],
                trust_remote_code=config.get("trust_remote_code", False),
                local_files_only=True
            )
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ❌ {name}: Not found locally")


def main():
    parser = argparse.ArgumentParser(
        description="Download models for multilingual JSON evaluation"
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        help="Specific model(s) to download"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify downloaded models"
    )
    parser.add_argument(
        "--cache-dir", "-c",
        help="Custom cache directory for downloads"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.list:
        list_models()
    elif args.verify:
        verify_downloads()
    else:
        download_all(models=args.model, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
