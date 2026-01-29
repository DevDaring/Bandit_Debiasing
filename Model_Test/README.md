# Multilingual JSON Response Evaluation Framework

A comprehensive evaluation framework for testing LLM capabilities in generating valid JSON responses across **English**, **Hindi**, and **Bengali** languages.

## 🎯 Purpose

This framework evaluates how well different LLMs can:
1. Generate **valid JSON** when asked in regional languages
2. Maintain **language consistency** (use appropriate script in keys/values)
3. Follow **structured output** instructions
4. Extract entities and format data as JSON

## 📊 Models Tested

| Model | Parameters | Family | Size | Gated |
|-------|------------|--------|------|-------|
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen | ~3.5 GB | No |
| Llama-3.2-1B-Instruct | 1B | Llama | ~2.5 GB | Yes |
| Gemma-2-2B-IT | 2B | Gemma | ~5.0 GB | Yes |
| mGPT-1.3B | 1.3B | mGPT | ~3.0 GB | No |
| BLOOMZ-7B1 | 7B | BLOOM | ~15 GB | No |

**Total Download Size:** ~29 GB

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create directory
cd multilingual_json_eval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Hugging Face Token

```bash
# Copy template
cp .env.template .env

# Edit .env and add your token
# Get token from: https://huggingface.co/settings/tokens
```

**Important:** For gated models (Llama, Gemma), you must:
1. Go to the model page on Hugging Face
2. Accept the license agreement
3. Then download will work

### 3. Download Models

```bash
# List available models
python download_models.py --list

# Download all models
python download_models.py

# Download specific model
python download_models.py --model Qwen2.5-1.5B-Instruct

# Verify downloads
python download_models.py --verify
```

### 4. Run Evaluation

```bash
# Full evaluation (all models, all languages)
python multilingual_json_evaluator.py

# Test specific models
python multilingual_json_evaluator.py --models Qwen2.5-1.5B-Instruct Gemma-2-2B-IT

# Custom output directory
python multilingual_json_evaluator.py --output my_results
```

### 5. Quick Testing

```bash
# Quick test single model
python quick_test.py --model Qwen2.5-1.5B-Instruct

# Test specific language
python quick_test.py --model Qwen2.5-1.5B-Instruct --language hindi

# Interactive mode
python quick_test.py --model Qwen2.5-1.5B-Instruct --interactive
```

## 📁 Output Files

After evaluation, you'll find in the `results/` directory:

| File | Description |
|------|-------------|
| `detailed_results_*.csv` | Per-prompt evaluation metrics |
| `summary_*.csv` | Aggregated statistics by model and language |
| `raw_responses_*.json` | Full model responses and parsed JSON |
| `evaluation_report.md` | Human-readable report |

## 📈 Evaluation Metrics

### 1. Valid JSON Rate
- Binary: Is the response valid JSON?
- Handles JSON in code blocks, markdown, etc.

### 2. Structure Score (0-1)
- 0.4: Valid JSON
- 0.3: Has expected keys
- 0.3: Non-empty values

### 3. Key Accuracy (0-1)
- Ratio of expected keys found in response
- Handles both exact and normalized (lowercase) matching

### 4. Value Accuracy (0-1)
- For prompts with expected values
- Handles string normalization and partial matches

### 5. Language Consistency (0-1)
- Does the JSON use the expected language script?
- Detects Devanagari (Hindi), Bengali, and Latin scripts

## 🧪 Test Prompts

Each language has 5 prompt types:

| Type | Description | Example |
|------|-------------|---------|
| `entity_extraction` | Extract structured info from text | Doctor's details |
| `classification` | Classify items with confidence | Sentiment analysis |
| `data_transformation` | Convert text to JSON | Student info |
| `nested_json` | Create nested structures | Company hierarchy |
| `qa_extraction` | Answer questions as JSON | Country facts |

## 🔬 Research Applications

This framework is designed for:

1. **Bias Analysis**: Compare model performance across languages
2. **Multilingual Capability**: Test Indic language understanding
3. **Structured Output**: Evaluate instruction following
4. **Debiasing Research**: Baseline measurements for your MAB framework

## 💡 Tips for Best Results

### Memory Management
```python
# For 24GB VRAM, run models sequentially
# BLOOMZ-7B1 uses 4-bit quantization automatically
```

### Custom Prompts
```python
# Add prompts to TEST_PROMPTS in multilingual_json_evaluator.py
TEST_PROMPTS["hindi"].append({
    "id": "hi_custom",
    "type": "custom",
    "prompt": "Your custom prompt here",
    "expected_keys": ["key1", "key2"]
})
```

### GCP Execution
```bash
# On GCP instance with GPU
nohup python multilingual_json_evaluator.py > eval.log 2>&1 &

# Monitor progress
tail -f eval.log
```

## 📊 Expected Results Format

```markdown
### Model × Language Matrix (Valid JSON %)

| Model | English | Hindi | Bengali |
|-------|---------|-------|---------|
| Qwen2.5-1.5B | 95.0% | 80.0% | 75.0% |
| Llama-3.2-1B | 90.0% | 70.0% | 65.0% |
| Gemma-2-2B | 92.0% | 85.0% | 78.0% |
| mGPT-1.3B | 75.0% | 60.0% | 55.0% |
| BLOOMZ-7B1 | 88.0% | 82.0% | 80.0% |
```

## 🔗 Integration with MAB Debiasing

Use these results as:
1. **Baseline**: JSON generation without debiasing
2. **Context Features**: Language capability as bandit context
3. **Evaluation**: Compare pre/post debiasing JSON quality

```python
# In your MAB framework
from multilingual_json_evaluator import JSONEvaluator, MultilingualJSONTester

evaluator = JSONEvaluator()
json_obj, error = evaluator.extract_json_from_response(model_output)
score = evaluator.calculate_structure_score(json_obj, expected_keys)
```

## 📝 Citation

If you use this framework in your research:

```bibtex
@misc{multilingual_json_eval,
  title={Multilingual JSON Response Evaluation for LLMs},
  author={Koushik},
  year={2024},
  note={Framework for evaluating structured output in Hindi, Bengali, and English}
}
```

## 🐛 Troubleshooting

### "Access denied" for Llama/Gemma
- Accept license at huggingface.co/meta-llama or google/gemma
- Ensure token has read access

### CUDA out of memory
- Models run sequentially with GPU clearing
- Use `--models` to test fewer models

### JSON extraction fails
- Check `raw_responses_*.json` for actual outputs
- Some models need more explicit JSON instructions

## 📄 License

MIT License - Feel free to use and modify for your research.
