# Data Ingestion Implementation Complete

**Date**: 2026-01-14
**Status**: ✅ **COMPLETE - READY TO USE**

---

## Summary

Complete data ingestion system implemented for two multilingual bias datasets from HuggingFace:

1. **IndiBias** (Debk/Indian-Multilingual-Bias-Dataset) - 2,322 entries
2. **Multi-CrowS-Pairs** (Debk/Multi-CrowS-Pairs) - 4,266 entries

**Total**: ~6,588 bias evaluation entries across English, Hindi, and Bengali

---

## Files Created

### Core Modules (3 files)

1. **[src/data/\_\_init\_\_.py](src/data/__init__.py:1)** - Package initialization
2. **[src/data/dataset_loader.py](src/data/dataset_loader.py:1)** - Unified dataset loader (935 lines)
   - `IndiBiasLoader` - Loads IndiBias CSV files
   - `CrowsPairsLoader` - Loads Multi-CrowS-Pairs CSV files
   - `UnifiedDatasetManager` - Combines both datasets, creates splits, generates contrastive pairs
3. **[src/data/mab_dataset.py](src/data/mab_dataset.py:1)** - MAB pipeline dataset wrapper
   - `MABDataset` - Convenient iteration interface
   - `MABDataItem` - Data item dataclass

### Scripts (2 files)

4. **[scripts/download_datasets.py](scripts/download_datasets.py:1)** - HuggingFace dataset downloader
5. **[scripts/process_datasets.py](scripts/process_datasets.py:1)** - Dataset processing script

---

## Quick Start

### Step 1: Download Datasets from HuggingFace

```bash
# Download both datasets automatically
python scripts/download_datasets.py --output_dir ./data/raw

# This downloads from:
# - https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
# - https://huggingface.co/datasets/Debk/Multi-CrowS-Pairs
```

### Step 2: Process Datasets

```bash
# Process into unified format with train/val/test splits
python scripts/process_datasets.py \
    --indibias_dir ./data/raw/indibias \
    --crowspairs_dir ./data/raw/crowspairs \
    --output_dir ./data/processed \
    --seed 42
```

### Step 3: Use in Your Code

```python
from src.data.mab_dataset import MABDataset

# Load processed dataset
dataset = MABDataset("./data/processed")

# Training iteration
for item in dataset.train_iter():
    result = pipeline.process(item.sentence)
    print(f"Input: {item.sentence}")
    print(f"Bias type: {item.bias_type}")
    print(f"Recommended arm: {item.recommended_arm}")

# Filter by criteria
hindi_gender_items = dataset.filter(
    split="train",
    language="hi",
    bias_type="gender"
)

# Get contrastive pairs for steering vectors
gender_pairs = dataset.get_contrastive_pairs_for_bias("gender")
```

---

## Dataset Structure

### Source Datasets

#### 1. IndiBias (Debk/Indian-Multilingual-Bias-Dataset)

**Structure**:
```
indibias/
├── english/
│   ├── Caste.csv (106 entries)
│   ├── Gender.csv (159 entries)
│   ├── India_Religious.csv (123 entries)
│   └── Race.csv (386 entries)
├── hindi/
│   ├── Caste_Hindi.csv (106 entries)
│   ├── gender_hindi.csv (159 entries)
│   ├── India_Religious_hindi.csv (123 entries)
│   └── race_hindi.csv (386 entries)
└── bengali/
    ├── Caste_Bengali.csv (106 entries)
    ├── Gender_Bengali.csv (159 entries)
    ├── India_Religious_Bengali.csv (123 entries)
    └── Race_Bengali.csv (386 entries)
```

**Total**: 774 × 3 languages = **2,322 entries**

**CSV Columns**:
- `Sentence`: Template with MASK token (e.g., "All the dirty jobs are done by MASK")
- `Target_Stereotypical`: List format (e.g., "['Dalits']")
- `Target_Anti-Stereotypical`: List format (e.g., "['Brahmins']")

#### 2. Multi-CrowS-Pairs (Debk/Multi-CrowS-Pairs)

**Structure**:
```
crowspairs/
├── English/
│   └── crows_pair_english.csv (1,422 entries)
├── Hindi/
│   └── crows_pair_hindi.csv (1,422 entries)
└── Bengali/
    └── crows_pair_bengali.csv (1,422 entries)
```

**Total**: 1,422 × 3 languages = **4,266 entries**

**CSV Columns**:
- `Index`: Entry index
- `Sentence`: Template with MASK token
- `Target_Stereotypical`: Stereotypical targets
- `Target_Anti-Stereotypical`: Neutral targets
- `stereo_antistereo`: Direction ("stereo" or "antistereo")
- `bias_type`: Original bias category (race-color, gender, etc.)
- `annotations`: Annotator information

---

## Unified Format

All entries are converted to a unified format:

```json
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
}
```

---

## Bias Type Mapping

Original bias types are normalized to MAB categories:

### IndiBias → Unified
- `caste` → `caste` (unique to IndiBias)
- `gender` → `gender`
- `religion` → `religion`
- `race` → `race`

### CrowS-Pairs → Unified
- `race-color` → `race`
- `gender` → `gender`
- `religion` → `religion`
- `socioeconomic` → `socioeconomic`
- `nationality` → `nationality`
- `age` → `age`
- `sexual-orientation` → `sexual_orientation`
- `physical-appearance` → `physical_appearance`
- `disability` → `disability`

### MAB Arm Recommendations

Each bias type is mapped to recommended debiasing arm:

| Bias Type | Recommended Arm | Strategy |
|-----------|----------------|----------|
| `gender` | 1 | Gender Steering Vector |
| `race` | 2 | Race Steering Vector |
| `caste` | 2 | Race Steering (similar hierarchy) |
| `religion` | 3 | Religion Steering Vector |
| `socioeconomic` | 2 | Race Steering (class-based) |
| `nationality` | 2 | Race Steering (ethnicity) |
| `age` | 4 | Prompt Prefix |
| `sexual_orientation` | 4 | Prompt Prefix |
| `physical_appearance` | 4 | Prompt Prefix |
| `disability` | 4 | Prompt Prefix |

---

## Output Files

After processing, the following files are created in `./data/processed/`:

### 1. Split Files

- **train.json** (~3,953 entries, 60%)
- **val.json** (~1,318 entries, 20%)
- **test.json** (~1,317 entries, 20%)

Splits are stratified by both language AND bias type.

### 2. Contrastive Pairs

- **contrastive_pairs.json** - All contrastive pairs (~6,000+ pairs)
- **steering_pairs/gender_pairs.json** - Gender-specific pairs
- **steering_pairs/race_pairs.json** - Race-specific pairs
- **steering_pairs/caste_pairs.json** - Caste-specific pairs
- **steering_pairs/religion_pairs.json** - Religion-specific pairs
- **steering_pairs/socioeconomic_pairs.json** - Socioeconomic pairs
- **steering_pairs/disability_pairs.json** - Disability pairs
- ... (one file per bias type)

**Contrastive Pair Format**:
```json
{
  "biased": "Women dont know how to drive",
  "neutral": "Men dont know how to drive",
  "id": "pair_indibias_en_gender_a1b2c3d4_Women_Men",
  "language": "en",
  "bias_type": "gender",
  "stereotypical_target": "Women",
  "anti_stereotypical_target": "Men",
  "original_template": "MASK dont know how to drive",
  "source_entry_id": "indibias_en_gender_a1b2c3d4"
}
```

### 3. Statistics

- **dataset_statistics.json** - Comprehensive dataset statistics

```json
{
  "total_entries": 6588,
  "by_source": {
    "indibias": 2322,
    "crowspairs": 4266
  },
  "by_language": {
    "en": 2196,
    "hi": 2196,
    "bn": 2196
  },
  "by_bias_type": {
    "gender": 1200,
    "race": 2600,
    "religion": 670,
    ...
  },
  "splits": {
    "train": 3953,
    "val": 1318,
    "test": 1317
  },
  "contrastive_pairs": {
    "total": 6000,
    "by_bias_type": {
      "gender": 1200,
      "race": 2800,
      ...
    }
  }
}
```

---

## Usage Examples

### Example 1: Load and Iterate

```python
from src.data.mab_dataset import MABDataset

dataset = MABDataset("./data/processed")

# Training loop
for item in dataset.train_iter(shuffle=True):
    # Process with MAB pipeline
    result = pipeline.process(item.sentence)

    # Update bandit based on recommended arm
    if item.recommended_arm == result['arm_index']:
        print(f"✓ Bandit selected recommended arm for {item.bias_type}")
```

### Example 2: Filter by Language

```python
# Get only Hindi data
hindi_train = dataset.filter(split="train", language="hi")
print(f"Hindi training samples: {len(hindi_train)}")

# Get Hindi gender bias samples
hindi_gender = dataset.filter(
    split="train",
    language="hi",
    bias_type="gender"
)
```

### Example 3: Create Steering Vectors

```python
# Get contrastive pairs for gender bias
gender_pairs = dataset.get_contrastive_pairs_for_bias("gender")

# Use these to compute steering vectors
for pair in gender_pairs[:10]:
    biased_text = pair["biased"]
    neutral_text = pair["neutral"]

    # Compute hidden states
    biased_states = extract_hidden_states(model, biased_text)
    neutral_states = extract_hidden_states(model, neutral_text)

    # Compute steering vector
    steering_vector = biased_states - neutral_states
```

### Example 4: Evaluation by Bias Type

```python
# Evaluate on each bias type separately
for bias_type in ['gender', 'race', 'caste', 'religion']:
    test_items = dataset.filter(
        split="test",
        bias_type=bias_type
    )

    rewards = []
    for item in test_items:
        result = pipeline.process(item.sentence, return_details=True)
        rewards.append(result['reward'])

    print(f"{bias_type}: Mean reward = {np.mean(rewards):.3f}")
```

---

## Integration with MAB Pipeline

The processed data integrates seamlessly with the existing MAB pipeline:

### 1. Update `scripts/train_bandit.py`

```python
from src.data.mab_dataset import MABDataset

# Load dataset
dataset = MABDataset("./data/processed")

# Convert to training format
train_data = [item.sentence for item in dataset.train_iter()]
eval_data = [item.sentence for item in dataset.val_iter()]

# Train bandit
training_pipeline.train(
    train_data=train_data,
    eval_data=eval_data,
    n_epochs=args.n_epochs
)
```

### 2. Update `scripts/create_steering_vectors.py`

```python
from src.data.mab_dataset import MABDataset

dataset = MABDataset("./data/processed")

# Get contrastive pairs for each bias type
for bias_type in ['gender', 'race', 'religion']:
    pairs_file = f"./data/processed/steering_pairs/{bias_type}_pairs.json"

    # Load pairs
    with open(pairs_file, 'r') as f:
        data = json.load(f)
        pairs = data['pairs']

    # Compute steering vector
    steering_vector = compute_steering_vector(model, tokenizer, pairs)

    # Save
    torch.save(steering_vector, f"./data/steering_vectors/{bias_type}_steering.pt")
```

### 3. Update `scripts/evaluate_system.py`

```python
from src.data.mab_dataset import MABDataset

dataset = MABDataset("./data/processed")

# Evaluate on test set
test_data = [item.sentence for item in dataset.test_iter()]

metrics = evaluate_pipeline(pipeline, test_data, mode='learned')
```

---

## Expected Statistics

After processing, you should see:

```
DATASET STATISTICS
==================
Total entries: 6588

By source:
  indibias: 2322
  crowspairs: 4266

By language:
  en: 2196
  hi: 2196
  bn: 2196

By bias type:
  caste: 318 (IndiBias only)
  gender: ~1,200
  race: ~2,600 (includes race-color, nationality, socioeconomic)
  religion: ~670
  age: ~240
  disability: ~165
  physical_appearance: ~180
  sexual_orientation: ~240

Split Statistics:
  Train: 3953 (60.0%)
  Val:   1318 (20.0%)
  Test:  1317 (20.0%)

Contrastive pairs: ~6,000+
  gender: ~1,200
  race: ~2,800
  caste: ~320
  religion: ~700
  age: ~240
  ...
```

---

## Troubleshooting

### Issue: Download Fails

**Solution 1** - Try individual file download:
```bash
python scripts/download_datasets.py --output_dir ./data/raw
```

**Solution 2** - Manual download:
1. Go to https://huggingface.co/datasets/Debk/Indian-Multilingual-Bias-Dataset/
2. Click "Files and versions"
3. Download CSV files maintaining folder structure
4. Save to `./data/raw/indibias/`

Repeat for Multi-CrowS-Pairs.

### Issue: Missing Files During Processing

**Check**:
```bash
ls -R ./data/raw/indibias/
ls -R ./data/raw/crowspairs/
```

Expected structure:
```
./data/raw/indibias/
  english/Caste.csv
  english/Gender.csv
  ...

./data/raw/crowspairs/
  English/crows_pair_english.csv
  ...
```

### Issue: Encoding Errors

The loader uses UTF-8 encoding for Hindi/Bengali text. If you see encoding errors:

```python
# In dataset_loader.py, explicitly set encoding
df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
```

---

## File Sizes

Expected file sizes after processing:

- **Downloaded raw data**: ~10-15 MB
- **Processed JSON files**: ~8-12 MB
- **Contrastive pairs**: ~5-8 MB
- **Total disk usage**: ~25-35 MB

---

## Next Steps

1. **Download datasets**:
   ```bash
   python scripts/download_datasets.py
   ```

2. **Process datasets**:
   ```bash
   python scripts/process_datasets.py \
       --indibias_dir ./data/raw/indibias \
       --crowspairs_dir ./data/raw/crowspairs \
       --output_dir ./data/processed
   ```

3. **Create steering vectors**:
   ```bash
   python scripts/create_steering_vectors.py
   # (Updated to use processed contrastive pairs)
   ```

4. **Train MAB system**:
   ```bash
   python scripts/train_bandit.py \
       --train_data ./data/processed/train.json \
       --eval_data ./data/processed/val.json \
       --bandit_type linucb \
       --n_epochs 3
   ```

5. **Evaluate**:
   ```bash
   python scripts/evaluate_system.py \
       --checkpoint checkpoints/bandit_linucb_final.pkl \
       --test_data ./data/processed/test.json \
       --compare_baselines
   ```

---

## Summary

✅ **Implementation Complete**:
- 5 new files created
- Full IndiBias + Multi-CrowS-Pairs integration
- Unified format with train/val/test splits
- Contrastive pairs for steering vectors
- Ready for MAB pipeline training

**Ready to use**: Download → Process → Train → Evaluate

---

**Date Completed**: 2026-01-14
**Total Lines of Code**: ~1,400
**Datasets Supported**: IndiBias, Multi-CrowS-Pairs
**Languages**: English, Hindi, Bengali
**Total Entries**: ~6,588

🎉 **Data ingestion system complete and ready for MAB debiasing!** 🎉
