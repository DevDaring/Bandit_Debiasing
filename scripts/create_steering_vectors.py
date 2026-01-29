"""
Create steering vectors from contrastive pairs.
Run this ONCE before training to generate steering vectors.
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from config.model_config import ModelConfig, clear_gpu_memory
from src.llm.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_contrastive_pairs(file_path: str) -> List[Dict]:
    """Load contrastive pairs from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['pairs']


def extract_hidden_states(
    model,
    tokenizer,
    text: str,
    device: str = 'cuda'
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states from all layers.

    Returns:
        Dict mapping layer_idx -> hidden_state tensor
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states from all layers
    hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)

    # Average over sequence length for each layer
    layer_states = {}
    for layer_idx, layer_hidden in enumerate(hidden_states):
        # Average over sequence dimension
        avg_hidden = layer_hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
        layer_states[layer_idx] = avg_hidden.cpu()

    return layer_states


def compute_steering_vector(
    model,
    tokenizer,
    contrastive_pairs: List[Dict],
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Compute steering vector from contrastive pairs.

    For each layer:
        steering[layer] = mean(biased_hidden[layer]) - mean(neutral_hidden[layer])

    Returns:
        Tensor of shape (n_layers, hidden_dim)
    """
    logger.info(f"Computing steering vector from {len(contrastive_pairs)} pairs...")

    # Collect hidden states for all pairs
    biased_states_all = []
    neutral_states_all = []

    for pair in tqdm(contrastive_pairs, desc="Processing pairs"):
        biased_text = pair['biased']
        neutral_text = pair['neutral']

        # Extract hidden states
        biased_states = extract_hidden_states(model, tokenizer, biased_text, device)
        neutral_states = extract_hidden_states(model, tokenizer, neutral_text, device)

        biased_states_all.append(biased_states)
        neutral_states_all.append(neutral_states)

    # Compute mean for each layer
    n_layers = len(biased_states_all[0])
    hidden_dim = biased_states_all[0][0].shape[0]

    steering_vectors = torch.zeros(n_layers, hidden_dim)

    for layer_idx in range(n_layers):
        # Gather all hidden states for this layer
        biased_layer = torch.stack([s[layer_idx] for s in biased_states_all])  # (n_pairs, hidden_dim)
        neutral_layer = torch.stack([s[layer_idx] for s in neutral_states_all])

        # Compute mean difference
        biased_mean = biased_layer.mean(dim=0)
        neutral_mean = neutral_layer.mean(dim=0)

        steering_vectors[layer_idx] = biased_mean - neutral_mean

    logger.info(f"Steering vector computed: shape={steering_vectors.shape}")

    return steering_vectors


def main():
    """Main function to create all steering vectors."""
    logger.info("Starting steering vector creation...")

    # Paths
    data_dir = Path('./data/contrastive_pairs')
    output_dir = Path('./data/steering_vectors')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bias types to process
    bias_types = ['gender', 'race', 'religion']

    # Load model
    logger.info("Loading model...")
    model_config = ModelConfig()
    model_loader = ModelLoader()
    model, tokenizer = model_loader.load(model_config.model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Model loaded on {device}")
    logger.info(f"Current VRAM: {model_loader.current_vram_mb:.2f} MB")

    # Process each bias type
    for bias_type in bias_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {bias_type} bias")
        logger.info(f"{'='*60}")

        # Load contrastive pairs
        pairs_file = data_dir / f"{bias_type}_pairs.json"
        if not pairs_file.exists():
            logger.warning(f"File not found: {pairs_file}, skipping...")
            continue

        pairs = load_contrastive_pairs(str(pairs_file))
        logger.info(f"Loaded {len(pairs)} contrastive pairs")

        # Compute steering vector
        steering_vector = compute_steering_vector(model, tokenizer, pairs, device)

        # Save steering vector
        output_path = output_dir / f"{bias_type}_steering.pt"
        torch.save(steering_vector, output_path)
        logger.info(f"Saved steering vector to {output_path}")

        # Log statistics
        logger.info(f"Steering vector stats:")
        logger.info(f"  Shape: {steering_vector.shape}")
        logger.info(f"  Mean norm: {steering_vector.norm(dim=1).mean():.4f}")
        logger.info(f"  Min norm: {steering_vector.norm(dim=1).min():.4f}")
        logger.info(f"  Max norm: {steering_vector.norm(dim=1).max():.4f}")

    # Unload model
    logger.info("\nUnloading model...")
    model_loader.unload()
    clear_gpu_memory()
    logger.info(f"VRAM after unload: {model_loader.current_vram_mb:.2f} MB")

    logger.info("\n" + "="*60)
    logger.info("Steering vector creation complete!")
    logger.info("="*60)
    logger.info(f"Vectors saved to: {output_dir}")
    logger.info(f"Files created:")
    for bias_type in bias_types:
        vector_path = output_dir / f"{bias_type}_steering.pt"
        if vector_path.exists():
            size_mb = vector_path.stat().st_size / (1024 * 1024)
            logger.info(f"  - {vector_path.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
