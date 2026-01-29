# MAB Debiasing for Multilingual LLMs

Adaptive Multi-Armed Bandit (MAB) Debiasing Strategy Selection for Multilingual Large Language Models.

## Overview

This system dynamically selects optimal debiasing interventions from a portfolio of strategies using contextual bandit algorithms that learn from fairness and quality feedback signals.

- **Target Languages**: English, Hindi, Bengali
- **Target Model**: Qwen/Qwen2.5-7B-Instruct
- **Hardware**: Single GPU with 24GB VRAM
- **Debiasing Arms**: 6 strategies (no intervention, gender/race/religion steering, prompt prefix, output adjustment)
- **Bandit Algorithms**: LinUCB, Thompson Sampling, Neural Bandit

## Quick Start (GCP Execution)

### 1. Upload Code to GCP
```bash
gcloud compute scp --recurse Bandit_Debiasing/ instance-name:~/ --zone=your-zone
```

### 2. On GCP Instance
```bash
cd ~/Bandit_Debiasing
pip install -r requirements.txt
python setup.py develop

# Create necessary directories
mkdir -p logs results checkpoints data/steering_vectors data/bias_evaluation_sets
```

### 3. Run Complete Experiment
```bash
# Option 1: Run entire experiment automatically
python scripts/run_experiment.py --language en --n_epochs 3

# Option 2: Run individual steps
# Step 1: Prepare datasets
python scripts/prepare_evaluation_data.py

# Step 2: Create steering vectors
python scripts/create_steering_vectors.py

# Step 3: Train each bandit algorithm
python scripts/train_bandit.py \
    --bandit_type linucb \
    --train_data data/bias_evaluation_sets/en/train.json \
    --eval_data data/bias_evaluation_sets/en/validation.json \
    --n_epochs 3 \
    --warmup_samples 100

python scripts/train_bandit.py --bandit_type thompson --train_data data/bias_evaluation_sets/en/train.json --eval_data data/bias_evaluation_sets/en/validation.json --n_epochs 3
python scripts/train_bandit.py --bandit_type neural --train_data data/bias_evaluation_sets/en/train.json --eval_data data/bias_evaluation_sets/en/validation.json --n_epochs 3

# Step 4: Evaluate all trained models
python scripts/evaluate_system.py \
    --checkpoint checkpoints/bandit_linucb_final.pkl \
    --test_data data/bias_evaluation_sets/en/test.json \
    --bandit_type linucb \
    --compare_baselines

python scripts/evaluate_system.py --checkpoint checkpoints/bandit_thompson_final.pkl --test_data data/bias_evaluation_sets/en/test.json --bandit_type thompson --compare_baselines
python scripts/evaluate_system.py --checkpoint checkpoints/bandit_neural_final.pkl --test_data data/bias_evaluation_sets/en/test.json --bandit_type neural --compare_baselines
```

### 4. Download Results
```bash
# On local machine
gcloud compute scp --recurse instance-name:~/Bandit_Debiasing/results/ ./ --zone=your-zone
gcloud compute scp --recurse instance-name:~/Bandit_Debiasing/logs/ ./ --zone=your-zone
gcloud compute scp --recurse instance-name:~/Bandit_Debiasing/checkpoints/ ./ --zone=your-zone
```

### 5. Monitor Progress Remotely

With W&B logging enabled, monitor training in real-time:
```
https://wandb.ai/your-username/mab-debiasing
```

## Installation

```bash
pip install -r requirements.txt
python setup.py develop
```

## Project Structure

```
mab_debiasing/
├── config/                      # Configuration files
│   ├── model_config.py         # LLM and quantization settings
│   ├── bandit_config.py        # Bandit hyperparameters
│   └── steering_vectors.py     # Steering vector paths
├── data/                        # Data files
│   ├── bias_evaluation_sets/   # Bias benchmark datasets
│   ├── steering_vectors/       # Pre-computed steering vectors
│   └── contrastive_pairs/      # Pairs for steering vector creation
├── src/                         # Source code
│   ├── context_extractor/      # Feature extraction
│   ├── bandit/                 # Bandit algorithms
│   ├── debiasing_arms/         # Debiasing interventions
│   ├── reward/                 # Reward calculation
│   ├── llm/                    # Model loading and generation
│   └── pipeline/               # End-to-end pipelines
├── scripts/                     # Executable scripts
├── tests/                       # Unit tests
├── results/                     # Output results
├── logs/                        # Log files
└── checkpoints/                 # Model checkpoints
```

## Usage

### Training

Train a bandit algorithm with all options:
```bash
python scripts/train_bandit.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --bandit_type linucb \
    --train_data data/bias_evaluation_sets/en/train.json \
    --eval_data data/bias_evaluation_sets/en/validation.json \
    --n_epochs 3 \
    --warmup_samples 100 \
    --eval_every 100 \
    --save_every 500 \
    --max_train_samples 1000 \
    --bias_weight 0.6 \
    --quality_weight 0.4 \
    --checkpoint_dir ./checkpoints \
    --results_dir ./results \
    --wandb_project "mab-debiasing" \
    --wandb_run_name "my_experiment"
```

### Evaluation

Evaluate a trained system with baselines:
```bash
python scripts/evaluate_system.py \
    --checkpoint checkpoints/bandit_linucb_final.pkl \
    --test_data data/bias_evaluation_sets/en/test.json \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --bandit_type linucb \
    --results_dir ./results/evaluation \
    --compare_baselines \
    --max_samples 200
```

### Inference

**Interactive mode** - Test inputs interactively:
```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/bandit_linucb_final.pkl \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --bandit_type linucb \
    --mode interactive
```

**Batch mode** - Process file of inputs:
```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/bandit_linucb_final.pkl \
    --mode batch \
    --input_file inputs.txt \
    --output_file results.txt
```

## Configuration

Edit configuration files in `config/`:

- `model_config.py`: Model selection, quantization settings, memory limits
- `bandit_config.py`: Bandit hyperparameters, reward weights
- `steering_vectors.py`: Paths to pre-computed steering vectors

## Architecture

### Pipeline Flow

```
Input Text → Context Extraction → Bandit Selection → Debiasing Arm →
LLM Generation → Reward Calculation → Bandit Update → Output
```

### Components

1. **Context Extractor**: Extracts 128-dim feature vector from input
   - Language detection
   - Demographic markers
   - Topic classification
   - Bias risk scoring

2. **Bandit Algorithms**: Select optimal debiasing strategy
   - LinUCB (linear UCB)
   - Thompson Sampling (Bayesian)
   - Neural Bandit (deep learning)

3. **Debiasing Arms**: Apply interventions
   - No intervention (baseline)
   - Steering vectors (gender/race/religion)
   - Prompt prefix
   - Output adjustment

4. **Reward Calculator**: Score outputs
   - Bias scoring (embedding-based, lexical)
   - Quality scoring (coherence, length, repetition)

## Results

Results are saved to `results/` folder:

```
results/
├── training/                    # Training metrics
│   ├── linucb_metrics.json
│   ├── thompson_metrics.json
│   └── neural_metrics.json
├── evaluation/                  # Evaluation results
│   ├── comparison_report.json
│   └── figures/
└── experiment_summary.md        # Final summary
```

## Logging

Logs are written to multiple destinations:

1. **File logs**: `logs/run_{timestamp}.log`
2. **Weights & Biases**: Real-time cloud logging
3. **Progress files**: `results/progress.json`

## Memory Management

The system is designed for 24GB VRAM:

- 4-bit quantization (NF4) for LLM
- Sequential model loading only
- Aggressive memory cleanup between operations
- Memory usage logged at each step
- Neural bandit runs on CPU to avoid GPU conflicts

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_context_extractor.py

# Run tests excluding slow tests
pytest -m "not slow"

# Run tests excluding GPU tests
pytest -m "not gpu"

# Run with coverage
pytest --cov=src --cov-report=html
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors:

1. Reduce batch size in neural bandit config
2. Decrease `max_train_samples` during training
3. Use fewer warmup samples
4. Ensure previous models are unloaded: check `clear_gpu_memory()` calls

### Slow Training

To speed up training:

1. Reduce `eval_every` to evaluate less frequently
2. Use smaller evaluation set with `--max_eval_samples`
3. Start with LinUCB (fastest) before trying Neural Bandit
4. Use `--max_train_samples` for quick testing

### Missing Steering Vectors

If steering vectors are missing:

```bash
python scripts/create_steering_vectors.py
```

This requires contrastive pairs in [data/contrastive_pairs/](data/contrastive_pairs/)

### W&B Login Issues

If W&B login fails:

```bash
# Login to W&B
wandb login

# Or disable W&B
python scripts/train_bandit.py --no_wandb ...
```

## Expected Runtime (24GB GPU)

Approximate times on GCP with 24GB VRAM:

- **Dataset preparation**: 10-15 minutes
- **Steering vector creation**: 30-45 minutes
- **Training (1 epoch, 1000 samples)**: 2-3 hours per algorithm
- **Evaluation (200 samples)**: 15-20 minutes per algorithm
- **Complete experiment (3 epochs, 3 algorithms)**: 20-24 hours

## File Sizes

Expected file sizes after complete run:

- Steering vectors: ~150MB each (3 × 150MB = 450MB)
- Bandit checkpoints: ~10-50MB each
- Dataset files: ~50-100MB total
- Results and logs: ~100-200MB
- **Total disk usage**: ~1-2GB

## Citation

```bibtex
@software{mab_debiasing,
  title={Adaptive Multi-Armed Bandit Debiasing for Multilingual LLMs},
  author={Research Team},
  year={2024},
  url={https://github.com/yourusername/mab_debiasing}
}
```

## License

MIT License
