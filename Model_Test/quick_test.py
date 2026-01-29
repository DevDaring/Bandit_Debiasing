#!/usr/bin/env python3
"""
Quick Test Script - Test a single model interactively
Useful for debugging and quick experiments.

Usage:
    python quick_test.py --model "Qwen2.5-1.5B-Instruct"
    python quick_test.py --model "Llama-3.2-1B-Instruct" --language hindi
"""

import os
import json
import argparse
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

load_dotenv()

# Sample prompts for quick testing
QUICK_PROMPTS = {
    "english": {
        "simple": """Return a JSON object with the following information:
Name: John
Age: 25
City: Mumbai

Format: {"name": "", "age": 0, "city": ""}""",
        
        "nested": """Create a JSON for this restaurant:
Name: Taj Mahal Restaurant
Rating: 4.5
Dishes: Biryani, Butter Chicken, Naan
Location: Delhi

Return as nested JSON with "restaurant" as root key."""
    },
    
    "hindi": {
        "simple": """निम्नलिखित जानकारी के साथ JSON object बनाएं:
नाम: राहुल
उम्र: 30
शहर: दिल्ली

Format: {"नाम": "", "उम्र": 0, "शहर": ""}""",
        
        "nested": """इस रेस्टोरेंट के लिए JSON बनाएं:
नाम: ताज महल रेस्टोरेंट
रेटिंग: 4.5
व्यंजन: बिरयानी, बटर चिकन, नान
स्थान: दिल्ली

"रेस्टोरेंट" को root key के रूप में nested JSON return करें।"""
    },
    
    "bengali": {
        "simple": """নিম্নলিখিত তথ্য সহ JSON object তৈরি করুন:
নাম: রাহুল
বয়স: ৩০
শহর: কলকাতা

Format: {"নাম": "", "বয়স": 0, "শহর": ""}""",
        
        "nested": """এই রেস্টুরেন্টের জন্য JSON তৈরি করুন:
নাম: তাজমহল রেস্টুরেন্ট
রেটিং: ৪.৫
খাবার: বিরিয়ানি, বাটার চিকেন, নান
অবস্থান: দিল্লি

"রেস্টুরেন্ট" root key হিসাবে nested JSON return করুন।"""
    }
}

# Model configs
MODEL_CONFIGS = {
    "Qwen2.5-1.5B-Instruct": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "trust_remote_code": True
    },
    "Llama-3.2-1B-Instruct": {
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "trust_remote_code": False
    },
    "Gemma-2-2B-IT": {
        "hf_id": "google/gemma-2-2b-it",
        "trust_remote_code": False
    },
    "mGPT-1.3B": {
        "hf_id": "ai-forever/mGPT",
        "trust_remote_code": False
    },
    "BLOOMZ-7B1": {
        "hf_id": "bigscience/bloomz-7b1",
        "trust_remote_code": False,
        "quantize": True
    }
}


def load_model(model_name: str):
    """Load model and tokenizer."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    hf_id = config["hf_id"]
    
    print(f"Loading {model_name}...")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        trust_remote_code=config.get("trust_remote_code", False),
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model kwargs
    model_kwargs = {
        "trust_remote_code": config.get("trust_remote_code", False),
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
    
    # Quantization for large models
    if config.get("quantize", False):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    
    model = AutoModelForCausalLM.from_pretrained(hf_id, **model_kwargs)
    
    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512):
    """Generate response from model."""
    # Format with chat template if available
    try:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except:
        formatted = f"User: {prompt}\n\nAssistant:"
    
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def extract_json(response: str):
    """Try to extract JSON from response."""
    import re
    
    # Direct parse
    try:
        return json.loads(response.strip()), None
    except:
        pass
    
    # Find JSON in response
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'(\{[\s\S]*\})',
        r'(\[[\s\S]*\])'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                return json.loads(match.strip()), None
            except:
                continue
    
    return None, "Could not extract JSON"


def evaluate_json(parsed_json, language: str) -> dict:
    """Evaluate JSON quality."""
    if parsed_json is None:
        return {
            "valid": False,
            "key_count": 0,
            "has_values": False,
            "language_detected": "unknown"
        }
    
    # Detect language in keys/values
    import re
    text = json.dumps(parsed_json, ensure_ascii=False)
    
    hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total = hindi_chars + bengali_chars + english_chars
    if total == 0:
        detected = "unknown"
    elif hindi_chars / total > 0.3:
        detected = "hindi"
    elif bengali_chars / total > 0.3:
        detected = "bengali"
    else:
        detected = "english"
    
    return {
        "valid": True,
        "key_count": len(parsed_json) if isinstance(parsed_json, dict) else len(parsed_json),
        "has_values": any(v for v in (parsed_json.values() if isinstance(parsed_json, dict) else parsed_json)),
        "language_detected": detected,
        "language_match": detected == language
    }


def run_interactive(model, tokenizer):
    """Run interactive testing mode."""
    print("\n" + "=" * 50)
    print("Interactive Mode - Enter prompts (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        print("\n")
        prompt = input("Prompt: ")
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        print("\nGenerating...")
        response = generate_response(model, tokenizer, prompt)
        
        print("\n--- Response ---")
        print(response)
        
        parsed, error = extract_json(response)
        if parsed:
            print("\n--- Extracted JSON ---")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        else:
            print(f"\n❌ JSON extraction failed: {error}")


def run_quick_test(model, tokenizer, language: str = "all"):
    """Run quick predefined tests."""
    print("\n" + "=" * 50)
    print("Quick Test Mode")
    print("=" * 50)
    
    languages = [language] if language != "all" else ["english", "hindi", "bengali"]
    
    results = []
    
    for lang in languages:
        if lang not in QUICK_PROMPTS:
            continue
            
        print(f"\n--- Testing {lang.upper()} ---")
        
        for prompt_type, prompt in QUICK_PROMPTS[lang].items():
            print(f"\n[{prompt_type}]")
            print(f"Prompt: {prompt[:100]}...")
            
            response = generate_response(model, tokenizer, prompt)
            print(f"Response: {response[:200]}...")
            
            parsed, error = extract_json(response)
            eval_result = evaluate_json(parsed, lang)
            
            status = "✓" if eval_result["valid"] else "❌"
            lang_match = "✓" if eval_result.get("language_match", False) else "❌"
            
            print(f"Valid JSON: {status}")
            print(f"Language Match: {lang_match} (detected: {eval_result['language_detected']})")
            
            if parsed:
                print(f"Parsed: {json.dumps(parsed, ensure_ascii=False)[:200]}")
            
            results.append({
                "language": lang,
                "type": prompt_type,
                "valid": eval_result["valid"],
                "language_match": eval_result.get("language_match", False)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    total = len(results)
    valid = sum(1 for r in results if r["valid"])
    lang_match = sum(1 for r in results if r["language_match"])
    
    print(f"Valid JSON: {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"Language Match: {lang_match}/{total} ({100*lang_match/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Quick model testing")
    parser.add_argument("--model", "-m", required=True, 
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to test")
    parser.add_argument("--language", "-l", default="all",
                       choices=["all", "english", "hindi", "bengali"],
                       help="Language to test")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Login
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    # Run tests
    if args.interactive:
        run_interactive(model, tokenizer)
    else:
        run_quick_test(model, tokenizer, args.language)


if __name__ == "__main__":
    main()
